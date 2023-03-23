''' Test ETDQN '''
import gym
import time
import numpy as np
import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
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

# create a suitable and similar initial scenario for test
def create_initial_env(s,flag_start):
    flag_start_1 = False
    flag_start_2 = False
    j = 0    
    if np.absolute(s[0,2]-0.75) <= 0.05: # the ego-vehicle at the 4th lane
        for i in range(1,5):
            if np.absolute(s[i,2]) <= 0.001:
                j += 1
                if np.absolute(s[i,1]) <= 0.15:
                    flag_start_1 = True
        if j >= 2: # at least two vehicles in front of the ego-vehicle
            flag_start_2 = True
        flag_start = flag_start_1 and flag_start_2
    return flag_start

# begin to test
t0 = time.time()
test_episodes = PARAM_TEST_EPS
max_steps = PARAM_TEST_MAX_STEPS
gamma = PARAM_GAMMA

steps_record = []
reward_record = []
speed_record = []
epoch_record = []
triggering_frequency_record = []

num1, num2, num3 = [1], np.arange(1000,int(PARAM_MAX_STEPS)+1-1000,1000), [] # test every 1000 steps
num = [y for x in [num1, num2, num3] for y in x]
for i in num:
    print('\n------ {} epoch test ------'.format(i))
    epoch_record.append(i)
    
    ad_model.load_model(i)
        
    test_steps = []
    test_rewards = []
    test_speed = []
    test_triggers = []
    
    for _ in range(test_episodes):
        reward = 0
        steps = 0
        speed = 0
        triggers = 0
        a_last = 3
        done = False
        
        flag_start = False  
        # create an initial scenario          
        while not flag_start:
            obs, info = env.reset()
            flag_start = create_initial_env(obs,flag_start)         
         
        v = info.get('speed')
        done = info.get('crashed')  
          
        rewards = 0.0
        steps = 0
        speed = v
        triggers = 0
        
        a_prev = 3
        a_prev_inx = th.as_tensor(a_prev, dtype=th.int64)
        a_prev_oh = F.one_hot(a_prev_inx, ad_model.n_actions)
        
        sa = np.hstack((obs.flatten(), a_prev_oh))
        print('a new episode')
        while not done:
            a = ad_model.choose_action(sa, determinstic=True)
    
            trigger = ad_model.determine_trigger(a, a_prev, steps)  
            triggers += trigger       

            obs_next, _, done, _, info = ad_model.env.step(a)
               
            v = info.get('speed')
            r = ad_model.reward_function(obs_next, v, trigger, done) # obs or obs_next? 
            
            a_inx = th.as_tensor(a, dtype=th.int64)
            a_oh = F.one_hot(a_inx, ad_model.n_actions)
    
            sa = np.hstack((obs_next.flatten(), a_oh))    
    
            a_prev = a
            a_prev_inx = th.as_tensor(a_prev, dtype=th.int64)
            a_prev_oh = F.one_hot(a_prev_inx, ad_model.n_actions)
            obs = obs_next
                
            rewards += gamma**steps * r
            speed += v
            steps += 1
        
            if steps >= max_steps:
                break
            
        for i in range(PARAM_STEPS_BEF_COLL - 1):
            rewards += gamma**(steps-1-i) * PARAM_COLL_DECAY**(i+1) * PARAM_CRC
            
        test_steps.append(steps)
        test_rewards.append(rewards)
        test_speed.append(speed/(steps+1))
        test_triggers.append(triggers/steps)
        
    test_reward_re = []
    test_steps_re = []
    test_speed_re = []
    test_triggers_re = []      
        
    # remove maximum and minimum
    if len(test_steps) > 2:
        index_min = np.argmin(test_rewards)
        index_max = np.argmax(test_rewards)
        counters = test_episodes - 2
        
        # index_min and index_max may be same one        
        if index_min == index_max:
            index_max = index_max + 1
            
        for index in range(test_episodes):
            if index != index_min and index != index_max:
                test_reward_re.append(test_rewards[index])
                test_steps_re.append(test_steps[index])
                test_speed_re.append(test_speed[index])
                test_triggers_re.append(test_triggers[index])        
    else:
        test_reward_re = test_rewards
        test_steps_re = test_steps
        test_speed_re = test_speed
        test_triggers_re = test_triggers
        counters = test_episodes
        
    average_reward = sum(test_reward_re)/counters
    average_steps = sum(test_steps_re)/counters
    average_speed = sum(test_speed_re)/counters
    average_triggering_frequency = sum(test_triggers_re)/counters
    
    print('average steps: {:.6f}, average reward: {:.6f}, average speed: {:.6f}, average triggering frequency: {:.6f}'\
            .format(average_steps, average_reward, average_speed, average_triggering_frequency))
    
    steps_record.append(average_steps)
    reward_record.append(average_reward)
    speed_record.append(average_speed)
    triggering_frequency_record.append(average_triggering_frequency)


print('testing time: ', time.time()-t0)
    
# plot results
plt.scatter(epoch_record, reward_record)
plt.plot(epoch_record, reward_record)
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.savefig('reward.png')
plt.show()

plt.scatter(epoch_record, steps_record)
plt.plot(epoch_record, steps_record)
plt.xlabel('Steps')
plt.ylabel('Average steps')
plt.savefig('step.png')
plt.show()

plt.scatter(epoch_record, speed_record)
plt.plot(epoch_record, speed_record)
plt.xlabel('Steps')
plt.ylabel('Average speed')
plt.savefig('speed.png')
plt.show()

plt.scatter(epoch_record, triggering_frequency_record)
plt.plot(epoch_record, triggering_frequency_record)
plt.xlabel('Steps')
plt.ylabel('Average triggering frequency')
plt.savefig('triggering_frequency.png')
plt.show()




