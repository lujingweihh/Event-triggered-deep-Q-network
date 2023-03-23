''' Train ETDQN '''
import gym
import time
from etdqn import ETDQN
from params import *

env = gym.make('highway-v0')

config = {
          'action': {'type': 'DiscreteMetaAction',},
          'observation':
            {
                'type': 'Kinematics',
                'vehicles_count': PARAM_VEH_OBS_COUNT,
                'features': ['presence', 'x', 'y', 'vx', 'vy'],
            },
           'manual_control': False,
           'simulation_frequency': PARAM_SF,
           'policy_frequency': PARAM_PF,
           'vehicles_count': PARAM_VEH_COUNT,
           'vehicles_density': PARAM_VEH_DENSITY,
}

env.config.update(config)
env.reset()

ad_model = ETDQN( env,
                  memory_normal_size = PARAM_NORMAL_MEM_SIZE,
                  batch_normal_size = PARAM_NORMAL_BATCH_SIZE,
                  episode_max_steps = PARAM_MAX_STEPS_ONE_EP,
                  e_greedy_increment = PARAM_EGREEDY_INC,
                  e_greedy = PARAM_EGREEDY,
                  learning_rate = PARAM_LR,
                  reward_decay = PARAM_GAMMA,
                  target_update_interval = PARAM_TARGET_UPDATE_INTERVAL,
                  DOUBLE_DQN = DOUBLE_DQN,
                  DUELING_DQN = DUELING_DQN,
                )

t0 = time.time()
ad_model.learn( learn_start = PARAM_LEARN_START, 
                max_time_steps = PARAM_MAX_STEPS, 
                interval_steps_save = PARAM_INTERVAL_TEST, 
              )

print('training time: ', time.time()-t0)   

