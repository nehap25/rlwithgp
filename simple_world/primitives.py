import numpy as np
import time
from simple_world.constants import DIM, DIM_DIAG, GAMMA, GRIPPER_ORI, GRIPPER_PITCH, GRIPPER_ROLL, GRIPPER_X, GRIPPER_Y_DIAG, MAX_ITER, STEPS
from simple_world.utils import Conf, Pose, assert_poses_close, close_enough, close_gripper, collision, eul_to_quat, get_conf, get_full_aabb, get_pose, get_straight_traj, get_yaw, quat_to_eul, rrt, rrt_to_traj, sample_aabb, set_conf, set_pose, step_simulation, touches, u_vec


def get_push_conf(robot_z, from_pose, to_pose):
    return get_push_from_conf(robot_z, from_pose, to_pose), get_push_to_conf(robot_z, from_pose, to_pose)


def get_push_from_conf(robot_z, from_pose, to_pose, margin=DIM_DIAG/2.+GRIPPER_Y_DIAG/2.):
    if from_pose.pos == to_pose.pos:
        return Conf(((from_pose.pos[0] - margin, from_pose.pos[1], robot_z), GRIPPER_ORI))
    push_vec = np.subtract(to_pose.pos, from_pose.pos)
    unit_vec = u_vec(push_vec)
    from_pos = from_pose.pos - unit_vec * margin
    from_pos[2] = robot_z
    yaw = get_yaw(unit_vec)
    ori = eul_to_quat((GRIPPER_ROLL, GRIPPER_PITCH, yaw))
    return Conf((tuple(from_pos), ori))


def get_push_to_conf(robot_z, from_pose, to_pose, margin=DIM/2.+GRIPPER_X/2.):
    if from_pose.pos == to_pose.pos:
        return get_push_from_conf(robot_z, from_pose, to_pose, margin)
    push_vec = np.subtract(to_pose.pos, from_pose.pos)
    unit_vec = u_vec(push_vec)
    to_pos = to_pose.pos - unit_vec * margin
    to_pos[2] = robot_z
    yaw = get_yaw(unit_vec)
    ori = eul_to_quat((GRIPPER_ROLL, GRIPPER_PITCH, yaw))
    return Conf((tuple(to_pos), ori))


def move(robot, from_conf=None, to_conf=None, traj=None, simulation=False, rrt_info=None):
    assert (from_conf is not None and to_conf is not None) or traj is not None
    if traj is None:
        X_aabb, obstacles = rrt_info
        rrt_out = rrt(X_aabb[:, :2].T, from_conf.pos[:2], to_conf.pos[:2], obstacles)
        traj = rrt_to_traj(rrt_out, from_conf, to_conf)
    if from_conf is None: from_conf = traj[0]
    if simulation: assert_poses_close(get_conf(robot), from_conf)
    set_conf(robot, traj[0])
    close_gripper(robot)
    for i in range(len(traj)):
        robot.constrain(traj[i])
        step_simulation(steps=STEPS, simulate=simulation)
        close_gripper(robot)
        if not close_enough(traj[i].pos, get_conf(robot).pos): 
            return robot.conf, max(0, i - 5)
    return traj[-1], len(traj)


def new_move(robot, traj, simulation):
    #set_conf(robot, traj[0])
    a=time.time()
    close_gripper(robot)
    b=time.time()
    for i in traj:
        
        robot.constrain(i)
        #print(STEPS)
        #start=time.time()
        step_simulation(steps=STEPS, simulate=False)
        #end=time.time()
        close_gripper(robot)
                
        #print(end-start) #,start3-start2,start2-start,"oneiteration")
   


def push(robot,action_list,simulation):
    #traj = get_straight_traj(from_conf, to_conf)
    #if simulation: assert_poses_close(get_pose(obj), from_pose)
    #set_pose(obj, from_pose)
    #prev_pose = get_pose(obj)
    move(robot, traj=action_list, simulation=simulation)



# streams


def get_pose_kin(obj, rel_pose, rel_obj, pose):
    world_pose = tuple(np.add(pose.pos, rel_pose.pos)), tuple(np.add(pose.ori, rel_pose.ori))
    return (Pose(world_pose),)


def get_push_conf_fn(robot_z):
    def fn(robot, obj, from_pose, to_pose):
        push_conf = get_push_conf(robot_z, from_pose, to_pose)
        return push_conf
    return fn


def pose_gen(obj, region):
    pos, ori = get_pose(obj)
    while True:
        sample = sample_aabb(np.array(region.aabb))
        sample = sample[0], sample[1], pos[2]
        yield (Pose((sample, ori)),)


def test_cfree_push(robot, from_conf, to_conf, obj, from_pose, to_pose, other_obj, pose):
    if obj == other_obj: return True
    obj_traj = get_straight_traj(from_pose, to_pose)
    for p in obj_traj[:-1]:
        set_pose(obj, p)
        set_pose(other_obj, pose)
        if collision(obj, other_obj): return False
    robot_traj = get_straight_traj(from_conf, to_conf)
    for conf in robot_traj[:-1]:
        set_conf(robot, conf)
        set_pose(other_obj, pose)
        if collision(robot, other_obj): return False
    return True


def test_feasible(robot, from_conf, to_conf, obj, from_pose, to_pose, other_obj, pose):
    return True


def test_touches(obj, pose, region):
    set_pose(obj, pose)
    obj_aabb = get_full_aabb(obj)
    return touches(obj_aabb, region.aabb)

