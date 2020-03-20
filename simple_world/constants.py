import numpy as np


COLLISION_DISTANCE = 0
DELTA = 1e-2
DIM = 0.1
DIM_DIAG = np.sqrt(2. * np.square(DIM))
EPSILON = 5e-2
FRICTION = 0.5
GAMMA = 1e-4
GRIPPER_ORI = (0., 1., 0., 1.)
GRIPPER_X = 0.062
GRIPPER_Y = 0.117
GRIPPER_Y_DIAG = np.sqrt(2. * np.square(GRIPPER_Y))
GRIPPER_Z = 0.325
GRIPPER_ROLL = np.radians(0.)
GRIPPER_PITCH = np.radians(90.)
MASS = 1.
MAX_FORCE = 500
MAX_FRICTION = 1.
MAX_ITER = 1e4
MAX_MASS = 30.
MIN_FRICTION = 0.01
MIN_MASS = 1.
ORI = (0., 0., 0., 1.)
PI = np.pi
POS = (0., 0., 0.)
SLEEP = 5e-4
STEPS = 5
TAU = 2 * PI
UNIT_X = (1., 0., 0.)
UNIT_Y = (0., 1., 0.)
UNIT_Z = (0., 0., 1.)
WLH = (DIM, DIM, DIM)

