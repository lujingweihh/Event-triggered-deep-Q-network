''' Parameters for ETDQN '''
# highway-env
PARAM_SF = 15
PARAM_PF = 5
PARAM_VEH_COUNT = 50
PARAM_VEH_OBS_COUNT = 6
PARAM_VEH_DENSITY = 1.0

# DQN
DUELING_DQN = True
DOUBLE_DQN = True

PARAM_LEARN_START = 600
PARAM_INTERVAL_TEST = 100 # test/save the model every "PARAM_INTERVAL_TEST" steps
PARAM_MAX_STEPS = 10e4 # max step in the training

PARAM_NORMAL_MEM_SIZE = 8192
PARAM_NORMAL_BATCH_SIZE = 256
PARAM_MAX_STEPS_ONE_EP = 200 # max step in an episode
PARAM_TARGET_UPDATE_INTERVAL = 1000
PARAM_EGREEDY = 0.95
PARAM_EGREEDY_INC = PARAM_EGREEDY/(0.8 * PARAM_MAX_STEPS)
PARAM_GAMMA = 0.97
PARAM_LR = 5e-5

# neural network 
PARAM_NN_HN = 1024
PARAM_NN_HL = 3

# reward
PARAM_CRV = 0.5
PARAM_CRC = -5.0
PARAM_CRL = 0.1
PARAM_CRE = -1.5

PARAM_STEPS_BEF_COLL = 6
PARAM_COLL_DECAY = 0.8
PARAM_CRC_DISCOUNT = ([ PARAM_COLL_DECAY**5, PARAM_COLL_DECAY**4, PARAM_COLL_DECAY**3, PARAM_COLL_DECAY**2, PARAM_COLL_DECAY**1, 0 ])

# test
PARAM_TEST_EPS = 7
PARAM_TEST_MAX_STEPS = 100 