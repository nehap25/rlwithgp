import numpy as np
import os
import pybullet as p
import time

from simple_world.constants import COLLISION_DISTANCE, DELTA, EPSILON, FRICTION, GRIPPER_Y, MASS, MAX_FORCE, ORI, POS, SLEEP, TAU, UNIT_X, UNIT_Z, WLH

#from rrt_algorithms.src.rrt.rrt_star_bid import RRTStarBidirectional
#from rrt_algorithms.src.search_space.search_space import SearchSpace


# classes


class Conf(object):
    def __init__(self, pose):
        self.pos, self.ori = pose

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.pos == other.pos and self.ori == other.ori

    def __repr__(self):
        pos_s = '(str(round(self.pos[0],2)),str(round(self.pos[1],2)),str(round(self.pos[2],2)))'
        ori = quat_to_eul(self.ori)
        ori_s ='('+str(round(ori[0],2))+','+str(round(ori[1],2))+','+str(round(ori[2],2))+')'
        return 'conf@'+str(pos_s)+','+str(ori_s)


class Obj(object):
    def __init__(self, pid, attributes, color):
        self.pid = pid
        self.friction, self.mass, ori, pos, self.wlh = attributes
        self.pose = Pose((pos, ori))
        self.color = color

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.pid == other.pid

    def __repr__(self):
        return 'obj'+str(self.pid)


class Pose(object):
    def __init__(self, pose):
        self.pos, self.ori = pose

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.pos == other.pos and self.ori == other.ori

    def __repr__(self):
        pos_s = '(str(round(self.pos[0],2)),str(round(self.pos[1],2)),str(round(self.pos[2],2)))'
        ori = quat_to_eul(self.ori)
        ori_s = '('+str(round(ori[0],2))+','+str(round(ori[1],2))+','+str(round(ori[2],2))+')'
        return 'pose@'+str(pos_s)+','+str(ori_s)


class Region(object):
    def __init__(self, center, aabb):
        self.center = center
        self.aabb = aabb

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.center == other.center and self.aabb == other.aabb

    def __repr__(self):
        min_aabb_s = '('+str(round(self.aabb[0][0],2))+','+str(round(self.aabb[0][1],2))+','+str(round(self.aabb[0][2],2))+')'
        max_aabb_s =  '('+str(round(self.aabb[1][0],2))+','+str(round(self.aabb[1][1],2))+','+str(round(self.aabb[1][2],2))+')'
        return 'region@'+str(min_aabb_s)+','+str(max_aabb_s)


class Robot(object):
    def __init__(self, pid, conf):
        self.pid = pid
        self.conf = conf
        self.constraint = create_constraint(self.pid, self.conf)

    def constrain(self, conf):
        self.conf = conf
        change_constraint(self.constraint, self.conf)

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.pid == other.pid

    def __repr__(self):
        return 'robot'+str(self.pid)+str(self.constraint)


# helpers


def assert_poses_close(pose, other_pose, eps=EPSILON):
    assert close_enough(pose.pos, other_pose.pos, eps=eps) \
        and close_enough(quat_to_eul(pose.ori), quat_to_eul(other_pose.ori), eps=eps*10), str(pose) != str(other_pose)


def close_enough(vec, ovec, eps=EPSILON):
    diff = np.abs(np.subtract(vec, ovec))
    return np.all(np.minimum(diff, np.abs(diff-TAU)) <= eps)


def contains(aabb, pos):
    return touches(aabb, (pos, pos))


def dense_path(path, delta=DELTA):
    new_path = path
    while True:
        d = -np.inf
        dense = [new_path[0]]
        for i in range(len(new_path) - 1):
            dense.append(tuple(np.add(new_path[i + 1], new_path[i]) / 2.))
            dense.append(new_path[i + 1])
            d = max(d, np.linalg.norm(np.subtract(dense[-1], dense[-2])))
        new_path = dense
        if d <= delta: break
    return tuple(new_path)


def get_obstacles(objects):
    margin = GRIPPER_Y / 2.
    obstacles = list()
    for obj in objects:
        obj_aabb = p.getAABB(obj.pid)
        obstacles.append((obj_aabb[0][0] - margin,
                          obj_aabb[0][1] - margin,
                          obj_aabb[1][0] + margin,
                          obj_aabb[1][1] + margin))
    return tuple(obstacles)


def get_straight_traj(from_conf, to_conf, delta=DELTA):
    if from_conf.pos == to_conf.pos: return get_turn_traj(to_conf.pos, from_conf.ori, to_conf.ori)
    traj = [from_conf]
    current_pos = from_conf.pos
    step_vec = u_vec(np.subtract(to_conf.pos, current_pos)) * delta
    while np.linalg.norm(np.subtract(to_conf.pos, current_pos)) > delta:
        current_pos += step_vec
        traj.append(Conf((tuple(current_pos), from_conf.ori)))
    traj.extend(get_turn_traj(to_conf.pos, from_conf.ori, to_conf.ori))
    return tuple(traj)
    

def get_turn_traj(pos, from_ori, to_ori, delta=DELTA):
    if from_ori == to_ori: return (Conf((pos, to_ori)),)
    traj = list()
    current_ori = from_ori
    step_vec = u_vec(np.subtract(to_ori, current_ori)) * delta
    while np.linalg.norm(np.subtract(to_ori, current_ori)) > delta:
        current_ori += step_vec
        traj.append(Conf((pos, tuple(current_ori))))
    traj.append(Conf((pos, to_ori)))
    return tuple(traj)


def get_yaw(vec):
    dot = np.dot(u_vec(vec), UNIT_X)
    cross = np.cross(UNIT_X, vec)
    return np.arccos(np.clip(dot, -1., 1.)) * np.sign(np.dot(cross, UNIT_Z))


def read_file(file_name):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, 'r') as f:
        return f.read()


def rejection_sample_aabb(aabb, reject_aabbs):
    sample = sample_aabb(aabb)
    while any(contains(reject_aabb, sample) for reject_aabb in reject_aabbs):
        sample = sample_aabb(aabb)
    return sample


def rejection_sample_region(aabb, wlh, reject_aabbs):
    sample = sample_region(aabb, wlh)
    while any(touches(reject_aabb, sample.aabb) for reject_aabb in reject_aabbs):
        sample = sample_region(aabb, wlh)
    return sample


def rrt(X_aabbt, init, goal, obstacles):
    Q = np.array([(0.2, 4), (0.1, 4)], dtype=object)
    r = 0.01
    max_samples = 1024
    rewire_count = 32
    prc = 0.1
    X = SearchSpace(X_aabbt, obstacles)
    rrt = RRTStarBidirectional(X, Q, init, goal, max_samples, r, prc, rewire_count)
    path = rrt.rrt_star_bidirectional()
    return dense_path(path)


def rrt_to_traj(rrt_out, from_conf, to_conf):
    traj = [Conf(((pos[0], pos[1], from_conf.pos[2]), from_conf.ori)) for pos in rrt_out]
    traj.pop()
    traj.extend(get_turn_traj(to_conf.pos, from_conf.ori, to_conf.ori))
    return tuple(traj)


def sample_aabb(aabb):
    return tuple(np.random.uniform(low=aabb[0, :], high=aabb[1, :]))


def sample_region(aabb, wlh):
    center = sample_aabb(aabb)
    half_wlh = np.divide(wlh, 2)
    region_aabb = tuple(center - half_wlh), tuple(center + half_wlh)
    return Region(center, region_aabb)


def u_vec(vec):
    return vec / np.linalg.norm(vec)


# pybullet helpers


def change_constraint(constraint, conf, max_force=MAX_FORCE):
    p.changeConstraint(constraint, jointChildPivot=conf.pos, jointChildFrameOrientation=conf.ori, maxForce=max_force)


def close_gripper(robot):
    target_value = 0.
    gripper_left_index = 0
    gripper_right_index = 2
    p.resetJointState(robot.pid, gripper_left_index, target_value)
    p.resetJointState(robot.pid, gripper_right_index, target_value)


def collision(body, other_body, max_distance=COLLISION_DISTANCE):
    return len(p.getClosestPoints(bodyA=body.pid, bodyB=other_body.pid, distance=max_distance)) != 0


def create_constraint(robot, conf):
    return p.createConstraint(robot, -1, -1, -1, p.JOINT_FIXED,
                              (0., 0., 0.), (0., 0., 0.), conf.pos,
                              childFrameOrientation=conf.ori)


def create_object(attributes=(FRICTION, MASS, ORI, POS, WLH), color=(0., 1., 0., 1.)):
    friction, mass, ori, pos, wlh = attributes
    half_wlh = np.divide(wlh, 2.)
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_wlh)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half_wlh, rgbaColor=color)
    obj_id = p.createMultiBody(baseMass=mass,
                               baseCollisionShapeIndex=collision,
                               baseVisualShapeIndex=visual,
                               basePosition=pos, baseOrientation=ori)
    obj = Obj(obj_id, attributes, color)
    set_friction(obj, friction)
    return obj


def create_stack(n_stack, base_obj, attributes=(FRICTION, MASS, ORI, POS, WLH), color=(0., 1., 0., 1.)):
    height = attributes[-1][2]
    stack_objects = list()
    init_extend = list()
    obj_below = base_obj
    stack_pose = base_obj.pose
    for i in range(n_stack):
        stack_obj = create_object(attributes=attributes, color=color)
        stack_pose = Pose(((stack_pose.pos[0], stack_pose.pos[1], stack_pose.pos[2]+height), stack_pose.ori))
        set_pose(stack_obj, stack_pose)
        stack_objects.append(stack_obj)
        init_extend.append(('On', stack_obj, obj_below))
        obj_below = stack_obj
    return stack_objects, init_extend


def eul_to_quat(eul):
    return tuple(p.getQuaternionFromEuler(eul))


def get_conf(robot):
    conf = Conf(tuple(p.getBasePositionAndOrientation(robot.pid)))
    robot.conf = conf
    return conf

    
def get_dims(obj):
    amin, amax = get_full_aabb(obj)
    return tuple(amax - amin)


def get_full_aabb(obj):
    aabb =  np.array(p.getAABB(obj.pid))
    aabb_mins = [aabb[:1, :]]
    aabb_maxs = [aabb[1:, :]]
    for i in range(p.getNumJoints(obj.pid)):
        aabb = np.array(p.getAABB(obj.pid, i))
        aabb_mins.append(aabb[:1, :])
        aabb_maxs.append(aabb[1:, :])
    aabb_mins = np.vstack(aabb_mins)
    aabb_maxs = np.vstack(aabb_maxs)
    amin = np.amin(aabb_mins, axis=0)
    amax = np.amax(aabb_maxs, axis=0)
    return np.array([amin, amax])


def get_pose(obj):
    pose = Pose(tuple(p.getBasePositionAndOrientation(obj.pid)))
    obj.pose = pose
    return pose


def quat_to_eul(quat):
    return tuple(np.add(p.getEulerFromQuaternion(quat), TAU) % TAU)


def set_conf(robot, conf):
    p.resetBasePositionAndOrientation(robot.pid, conf.pos, conf.ori)
    robot.conf = conf


def set_friction(obj, friction):
    p.changeDynamics(obj.pid, -1, lateralFriction=friction)


def set_pose(obj, pose):
    p.resetBasePositionAndOrientation(obj.pid, pose.pos, pose.ori)
    obj.pose = pose


def step_simulation(steps=1, simulate=False):
    for i in range(steps):
        p.stepSimulation()
        if simulate: time.sleep(SLEEP)


def touches(aabb, other_aabb):
    return aabb[0][0] <= other_aabb[1][0] and other_aabb[0][0] <= aabb[1][0] \
           and aabb[0][1] <= other_aabb[1][1] and other_aabb[0][1] <= aabb[1][1] \
           and aabb[0][2] <= other_aabb[1][2] and other_aabb[0][2] <= aabb[1][2]

