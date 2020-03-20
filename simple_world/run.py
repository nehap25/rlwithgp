#!/usr/bin/env python3

from __future__ import print_function

import cProfile
import numpy as np
import pstats
import pybullet as p
import pybullet_data

from simple_world.constants import DIM, GRIPPER_ORI, GRIPPER_Z, ORI, STEPS
from simple_world.utils import Conf, Pose, Region, Robot, create_object, create_stack, get_conf, get_obstacles, get_pose, read_file, set_pose, step_simulation
from simple_world.primitives import get_pose_kin, get_push_conf_fn, move, push, test_cfree_push, test_feasible, test_touches

from pddlstream.algorithms.incremental import solve_incremental
from pddlstream.language.constants import print_solution
from pddlstream.language.generator import from_fn, from_gen_fn, from_test

try: input = raw_input
except NameError: pass


def main():
    solve_problem(seed=0, simulate=True)


def execute_plan(plan, rrt_info):
    action_dict = {
        'move': move,
        'push': push,
    }
    table_aabb, obstacle_objects = rrt_info
    for action in plan:
        input(f'press enter to execute action: {action.name}')
        action_dict[action.name](*action.args, simulation=True, rrt_info=(table_aabb, get_obstacles(obstacle_objects)))
        step_simulation(steps=STEPS)


def get_problem():
    domain_pddl = read_file('domain.pddl')

    constant_map = dict()

    table_aabb, robot, region, objects, poses, init_extend = load_world()

    stream_pddl = read_file('stream.pddl')
    stream_map = {
        's-pose-kin': from_fn(get_pose_kin),
        's-push-conf': from_fn(get_push_conf_fn(robot.conf.pos[2])),
        # 's-touch-region': from_gen_fn(pose_gen),
        't-cfree-push': from_test(test_cfree_push),
        't-feasible': from_test(test_feasible),
        't-touches': from_test(test_touches),
    }

    init = [
        ('Robot', robot),
        ('Conf', robot.conf),
        ('AtConf', robot, robot.conf),
        ('Region', region),
    ]
    for i in range(len(objects)):
        obj = objects[i]
        init.append(('Obj', obj))
        init.append(('WorldPose', obj, obj.pose))
        init.append(('AtWorldPose', obj, obj.pose))
        for pose in poses[i]:
            init.append(('WorldPose', obj, pose))
    init.extend(init_extend)

    both_in_region = ('and', ('Overlaps', objects[0], region), ('Overlaps', objects[1], region))
    in_region = ('and', ('Overlaps', objects[0], region))
    goal = in_region
    goal_no_side_effects = goal + tuple(('not', ('UnknownPose', obj)) for obj in objects)

    print('init:')
    for pred in init:
        print(pred)

    print('goal:', goal)

    return domain_pddl, constant_map, stream_pddl, stream_map, init, goal_no_side_effects


def load_world():
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0., 0., -10.)

    plane = p.loadURDF('plane.URDF')
    table = p.loadURDF('table/table.URDF')

    table_aabb = np.array(p.getAABB(table))
    table_top = table_aabb[1][2]

    init_conf = Conf(((table_aabb[0][0],
                 table_aabb[0][1],
                 table_top + GRIPPER_Z), GRIPPER_ORI))
    gripper = p.loadURDF('pr2_gripper.urdf',
                       basePosition=init_conf.pos,
                       baseOrientation=init_conf.ori)
    robot = Robot(gripper, init_conf)

    obj = create_object(color=(0., 0., 1., 1.))
    init_pose = Pose(((-.2, 0., table_top + DIM / 2.), ORI))
    set_pose(obj, init_pose)

    n_stack = 2
    stack_objects, init_extend = create_stack(n_stack, obj, color=(0., 0., 1., 1.))

    other_obj = create_object(color=(1., 0., 0., 1.))
    other_pose = Pose(((.2, 0., table_top + DIM / 2.), ORI))
    set_pose(other_obj, other_pose)

    step_simulation(steps=STEPS)
    conf = get_conf(robot)
    pose = get_pose(obj)
    for stack_obj in stack_objects: get_pose(stack_obj)
    other_pose = get_pose(other_obj)

    for stack_obj in stack_objects:
        rel_pose = Pose((tuple(np.subtract(stack_obj.pose.pos, obj.pose.pos)), (0., 0., 0., 0.)))
        init_extend.append(('RelPose', stack_obj, rel_pose, obj))
        init_extend.append(('AtRelPose', stack_obj, rel_pose, obj))
        init_extend.append(('PoseKin', stack_obj, stack_obj.pose, rel_pose, obj, obj.pose))

    center = table_aabb[1][0] - DIM, 0., table_aabb[1][2]
    halves = np.array([DIM, DIM/2., 0.])
    aabb = tuple(center - halves), tuple(center + halves)
    visual = p.createVisualShape(p.GEOM_BOX,
                                 halfExtents=halves,
                                 rgbaColor=(0., 1., 0., 1.))
    body = p.createMultiBody(baseVisualShapeIndex=visual,
                             basePosition=center)
    region = Region(center, aabb)

    objects = [obj, other_obj]
    objects.extend(stack_objects)

    goal_pose = Pose(((center[0]-DIM/2., center[1], pose.pos[2]), pose.ori))
    other_alt_pose = Pose((tuple(np.add(other_pose.pos, (0., .2, 0.))), other_pose.ori))
    other_goal_pose = Pose(((center[0]+DIM/2., center[1], other_pose.pos[2]), other_pose.ori))
    poses = [
        [goal_pose],
        [other_alt_pose, other_goal_pose],
        [],
        [],
    ]

    return table_aabb, robot, region, objects, poses, init_extend


def solve_problem(seed=None, simulate=False):
    p.connect(p.DIRECT)
    np.random.seed(seed)
    np_state = np.random.get_state()
    problem = get_problem()
    pr = cProfile.Profile()
    pr.enable()
    solution = solve_incremental(problem, verbose=False)
    pr.disable()
    pstats.Stats(pr).sort_stats('tottime').print_stats(10) # cumtime | tottime
    plan, cost, evaluations = solution
    print_solution(solution)
    p.disconnect()

    if simulate:
        p.connect(p.GUI)
        np.random.set_state(np_state)
        table_aabb, robot, region, objects, poses, init_extend = load_world()
        execute_plan(plan, (table_aabb, objects))
        step_simulation(steps=1000, simulate=True)
        p.disconnect()


if __name__ == '__main__':
    main()
