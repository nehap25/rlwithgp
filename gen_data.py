#!/usr/bin/env python3

import argparse
import json
import multiprocess
import numpy as np
import os
import pybullet as p
import pybullet_data
import math
import time
from simple_world.constants import DIM, ORI, FRICTION, GRIPPER_ORI, GRIPPER_X, GRIPPER_Y, GRIPPER_Z, MASS, MAX_FRICTION, MAX_MASS, MIN_FRICTION, MIN_MASS, STEPS, WLH
from simple_world.utils import Conf, Pose, Robot, close_enough, create_object, create_stack, eul_to_quat, get_full_aabb, get_pose, get_yaw, rejection_sample_aabb, rejection_sample_region, sample_aabb, set_conf, set_pose, step_simulation
from simple_world.primitives import get_push_conf, new_move, push


def conf_json(conf):
    json_dict = {
        'pos': conf.pos,
        'ori': conf.ori,
    }
    return json_dict


def data_json(data):
    robot, from_conf, to_conf, push_obj, goal_pose, region, objects, start_poses, end_poses = data
    json_dict = {
        'robot': robot_json(robot, from_conf, to_conf),
        'goal_pose': pose_json(goal_pose),
        'region': region_json(region),
        'objects': [object_json(push_obj, objects[i], start_poses[i], end_poses[i]) for i in range(len(objects))],
    }
    return json_dict

def get_conf(robot):
    conf = Conf(tuple(p.getBasePositionAndOrientation(robot.pid)))
    robot.conf = conf
    return conf

def data_to_json(data_in, fname):
    json_out = data_json(data_in)
    with open(fname, 'w') as outfile:
        json.dump(json_out, outfile)
    return json_out

def setup(use_gui):
    #TODO: get rid of random sampling
    n_obj=0
    n_stack=0
    connect_type = p.GUI if use_gui else p.DIRECT
    p.connect(connect_type)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0., 0., -10.)

    plane, table, robot = init_world()
    table_aabb = np.array(p.getAABB(table)) - np.divide(((-DIM, -DIM, 0), (DIM, DIM, 0)), 2)
    reject_aabbs = [get_full_aabb(robot)]

    objects,mass_bool = gen_objects(table_aabb, 1, reject_aabbs)
    push_obj = objects[0]
    region_wlh = (DIM, DIM, 0)
    region = rejection_sample_region(table_aabb, region_wlh, [p.getAABB(push_obj.pid)])

    #goal_pos = sample_aabb(np.array(region.aabb))
    push_vec = np.subtract((0.68,0.3,0.67), push_obj.pose.pos)
    yaw = get_yaw(push_vec)
    goal_ori = eul_to_quat((0, 0, yaw))
    goal_pose=Pose(((0.68,0.3,0.67),goal_ori))
    visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05], rgbaColor=(1, 0, 0, 0.75))
    goal_block_id = p.createMultiBody(baseVisualShapeIndex=visual_id, basePosition=list(goal_pose.pos), 
            )

    start_pose = Pose((push_obj.pose.pos, goal_ori))

    from_conf=robot.conf
 
    set_pose(push_obj, start_pose)

    objects.extend(gen_stack(push_obj, n_stack))

    

    objects.extend(gen_objects(table_aabb, n_obj, reject_aabbs))
    step_simulation(steps=20)
    end_poses = [get_pose(obj) for obj in objects]
    full_pose=end_poses+[from_conf]
    return full_pose, robot, objects,use_gui,goal_pose,mass_bool

def calc_reward(current_position,goal_pos):
    first_obj=current_position[0].pos
   
    #second_obj=current_position[1].pos
    #third_obj=current_position[2].pos
    sumy=0
    
    for i in [first_obj]:  #,second_obj,third_obj]:
        val=(goal_pos.pos[0]-i[0])**2 + (goal_pos.pos[1]-i[1])**2 #+ (goal_pos.pos[2]-first_obj[2])**2 #FIX IF ADDING MORE BLOCKS
        if val < 0.15:
          return 1-val
    return -val

def calc_action_list(angle,push_distance,current_position):
    push_distance=0.5+push_distance/2
    angle=angle*math.pi/2
    angle+=math.pi/2  #in RL model, should range from -pi/2 to pi/2
    push_distance=push_distance/2
    distance_measure=0.001
    x_coor=distance_measure*math.cos(angle)
    y_coor=distance_measure*math.sin(angle)
    z_coor=0.0
    iterations=int(push_distance//distance_measure)
    #print("ITERATIONS", iterations)
    action_list=[]
    robot_start=current_position
    for i in range(iterations):
        loc_tup=(robot_start.pos[0]+x_coor,robot_start.pos[1]+y_coor,robot_start.pos[2])
        robot_start=Conf((loc_tup,("a","b","c")))
        action_list.append(robot_start)
    return action_list

def calc_action_list_v2(x,y,current_position):    
    x_coor=x #/10
    y_coor=y #/10
    push_distance=0.7
    distance_measure = ((x_coor)**2 + (y_coor)**2)**0.5
    z_coor=0.0
   
    #iterations=int(push_distance//distance_measure)+1
    iterations=1
    #print("ITERATIONS", iterations)
    action_list=[]
    robot_start=current_position
    for i in range(iterations):
        loc_tup=(robot_start.pos[0]+x_coor,robot_start.pos[1]+y_coor,robot_start.pos[2])
        robot_start=Conf((loc_tup,("a","b","c")))
        action_list.append(robot_start)
    return action_list

def empty_steps():
    step_simulation(steps=20)

def step(angle,push_distance,robot,objects,goal_pose,use_gui):
    a=time.time()
    step_simulation(steps=STEPS)
    b=time.time()
    #print(angle, push_distance)
    #action_list=action_list_normal(angle,push_distance,get_conf(robot))
    action_list=calc_action_list_v2(angle,push_distance,get_conf(robot))
    #print(angle, push_distance)
    c=time.time()
    simulate_push(robot,use_gui, action_list)
    d=time.time()
    del action_list
    step_simulation(steps=STEPS)
    end_poses = [get_pose(obj) for obj in objects]+[get_conf(robot)]
    e=time.time()
    reward=calc_reward(end_poses,goal_pose)
    f=time.time()
    #print(f-e,e-d,d-c,c-b,b-a,"STEP")
    empty_steps()
    return robot,objects,use_gui,reward

def gen_objects(ground_aabb, n_obj, reject_aabbs):
    objects = list()
    obj_z = ground_aabb[1, 2] + DIM / 2
    ground_aabb[:, 2] = obj_z
    for i in range(n_obj):
        attributes,mass_bool = sample_attributes(ground_aabb, reject_aabbs)
        objects.append(create_object(attributes=attributes))
        reject_aabbs.append(p.getAABB(objects[-1].pid))
    if n_obj > 0:
        return objects,mass_bool
    else:  

        return objects


def gen_stack(base_obj, n_obj):
    stack_objects = list()
    aabb = np.array((base_obj.pose.pos, base_obj.pose.pos))
    below_obj = base_obj
    for i in range(n_obj):
        #if i==0:
           #attributes = sample_attributes(aabb, list(),1)
        #else:
        attributes,mass_bool=sample_attributes(aabb,list())
        (stack_obj,), _ = create_stack(1, below_obj, attributes=attributes)
        stack_objects.append(stack_obj)
        below_obj = stack_obj
    return stack_objects


def init_world():
    plane = p.loadURDF('plane.urdf')
    table = p.loadURDF('table/table.urdf')

    table_aabb = p.getAABB(table)
    table_top = table_aabb[1][2]

    conf = Conf(((table_aabb[0][0],
            table_aabb[0][1],
            table_top + GRIPPER_Z), GRIPPER_ORI))
    
    loc_tup=(-0.6662376612461023, 0.06738317807662543, 0.9510000000000001)
    conf=Conf((loc_tup,GRIPPER_ORI))
    
    gripper = p.loadURDF('pr2_gripper.urdf',
                         basePosition=conf.pos,
                         baseOrientation=conf.ori)
  
    robot = Robot(gripper, conf)
    return plane, table, robot


def json_to_data(fname):
    with open(fname) as json_file:
        return json.load(json_file)


def object_json(push_obj, obj, start_pose, end_pose):
    json_dict = {
        'pid': obj.pid,
        'friction': obj.friction,
        'mass': obj.mass,
        'wlh': obj.wlh,
        'start_pose': pose_json(start_pose),
        'end_pose': pose_json(end_pose),
        'push_obj': obj == push_obj,
    }
    return json_dict


def pose_json(pose):
    json_dict = {
        'pos': pose.pos,
        'ori': pose.ori,
    }
    return json_dict


def region_json(region):
    json_dict = {
        'center': region.center,
        'aabb': region.aabb,
    }
    return json_dict


def robot_json(robot, from_conf, to_conf):
    json_dict = {
        'pid': robot.pid,
        'from_conf': conf_json(from_conf),
        'to_conf': conf_json(to_conf),
    }
    return json_dict


def sample_attributes(ground_aabb, reject_aabbs,param=100):
    #if param ==1:
    #  friction=MAX_FRICTION/2
    #  mass=MASS
    #else:
    #  friction=MIN_FRICTION
    #  mass=MASS

    friction = FRICTION # np.random.uniform(low=MIN_FRICTION, high=MAX_FRICTION)
    a=np.random.uniform(low=0,high=1.0)
    if a > 0.5:
     mass=MIN_MASS
     mass_bool=0
    else:
     mass=MAX_MASS/2
     mass_bool=1

    mass = MIN_MASS #MAX_MASS/2 # np.random.uniform(low=MIN_MASS, high=MAX_MASS)
    ori = ORI
    pos = rejection_sample_aabb(ground_aabb, reject_aabbs)
    pos=(-0.5662376612461023, 0.06738317807662543,0.67)
    wlh = WLH
    return ((friction, mass, ori, pos, wlh),mass_bool)


def simulate_push(robot,use_gui, action_list):
    new_move(robot,action_list,use_gui)

