import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from torch.autograd import Variable
from params import *

device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

PARAM_CRC_VEC = PARAM_CRC * np.array(PARAM_CRC_DISCOUNT)

# construction and training of neural network draw on stable baseline 3
def create_mlp_net(
    input_dim: int,   # input dimension
    output_dim: int,  # output dimension
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False):

    modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(nn.Dropout(0.3))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    return nn.Sequential(*modules)

class DuelingNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int, 
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        ):
        super(DuelingNet, self).__init__()
        self.value_net = create_mlp_net(input_dim, 1, net_arch, activation_fn)
        self.advantage_net = create_mlp_net(input_dim, output_dim, net_arch, activation_fn) 

    def forward(self, x):
        value_out = self.value_net(x)
        advantage_out = self.advantage_net(x)        
        average_advantage = advantage_out - th.mean(advantage_out,dim=1,keepdim=True)        
        q_value = value_out + average_advantage        
        return q_value

class ETDQN:
    def __init__(
            self,
            env,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            e_greedy_increment=None,
            target_update_interval=200,
            memory_normal_size=3000,
            batch_normal_size=16,
            episode_max_steps = 500,
            DOUBLE_DQN=False, 
            DUELING_DQN=False, 
    ):
        self.env = env
        self.n_actions = env.action_space.n
        self.n_features = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = target_update_interval
        self.memory_normal_size = memory_normal_size
        self.batch_normal_size = batch_normal_size
        self.episode_max_steps = episode_max_steps
        self.double_q = DOUBLE_DQN
        self.dueling_q = DUELING_DQN
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        
        self.max_grad_norm = 10
        self.learn_step_counter = 0
        
        # store [s,a,r, done, s_] in an episode to determine the last L steps before the collision
        self.memory_one_ep = np.zeros((self.episode_max_steps, self.n_features*2 + 3 + 2*self.n_actions), dtype=np.float32)
        self.memory_one_ep_counter = 0
        self.steps_before_collision = PARAM_STEPS_BEF_COLL

        # creat the memory with [s,a,r, done, s_]
        self.memory_normal = np.zeros((self.memory_normal_size, self.n_features*2 + 3 + 2*self.n_actions), dtype=np.float32)
        self.memory_normal_counter = 0
        
        self.hn = PARAM_NN_HN
        self.hl = PARAM_NN_HL
        self.hnn = []
        
        # create neural network
        for _ in range(self.hl):
            self.hnn.append(self.hn)

        # build layer
        if self.dueling_q:
            self.eval_net = DuelingNet(self.n_features+self.n_actions, self.n_actions, self.hnn, activation_fn=nn.ReLU)
            self.target_net = DuelingNet(self.n_features+self.n_actions, self.n_actions, self.hnn, activation_fn=nn.ReLU)
        else:
            self.eval_net = create_mlp_net(self.n_features+self.n_actions, self.n_actions, self.hnn, activation_fn=nn.ReLU)
            self.target_net = create_mlp_net(self.n_features+self.n_actions, self.n_actions, self.hnn, activation_fn=nn.ReLU)
            
        self.eval_net.to(device)
        self.target_net.to(device)

        print('model------->')
        print(self.eval_net)

        # define the optimizer
        self.optimizer = th.optim.Adam(self.eval_net.parameters(), learning_rate)
        # define the loss function 
        self.loss_func = nn.SmoothL1Loss
        self.loss = 0.0

    def store_transition(self, s, a, r, s_, done):       
        transition = np.hstack((s.flatten(), [a, r, done], s_.flatten()))
        self.memory_one_ep[self.memory_one_ep_counter, :] = transition
        self.memory_one_ep_counter += 1
        
        if self.memory_one_ep_counter == self.episode_max_steps and not done: # no collision in this episode (achieve 200 time steps)
            index = self.memory_normal_counter % self.memory_normal_size
            if index +  self.memory_one_ep_counter  <= self.memory_normal_size:
                self.memory_normal[index:index+self.memory_one_ep_counter, :] = self.memory_one_ep[0:self.memory_one_ep_counter,:]
            else:
                exceed_steps =  index + self.memory_one_ep_counter - self.memory_normal_size
                self.memory_normal[index:self.memory_normal_size,:] = self.memory_one_ep[0:self.memory_one_ep_counter-exceed_steps,:]
                self.memory_normal[0:exceed_steps,:] = self.memory_one_ep[self.memory_one_ep_counter-exceed_steps:self.memory_one_ep_counter,:]
                
            self.memory_one_ep = np.zeros(
                 (self.episode_max_steps, self.n_features*2 + 3 + 2*self.n_actions), dtype=np.float32)
            self.memory_normal_counter += self.memory_one_ep_counter
            self.memory_one_ep_counter = 0
            
        if done: # collision within 200 time steps
            steps = np.min([self.memory_one_ep_counter,self.steps_before_collision])
            self.memory_one_ep[self.memory_one_ep_counter-steps:self.memory_one_ep_counter, self.n_features + self.n_actions + 1] += PARAM_CRC_VEC[-steps:]
            
            index = self.memory_normal_counter % self.memory_normal_size
            if index +  self.memory_one_ep_counter  <= self.memory_normal_size:
                self.memory_normal[index:index+self.memory_one_ep_counter, :] = self.memory_one_ep[0:self.memory_one_ep_counter,:]
            else:
                exceed_steps =  index + self.memory_one_ep_counter - self.memory_normal_size
                self.memory_normal[index:self.memory_normal_size,:] = self.memory_one_ep[0:self.memory_one_ep_counter-exceed_steps,:]
                self.memory_normal[0:exceed_steps,:] = self.memory_one_ep[self.memory_one_ep_counter-exceed_steps:self.memory_one_ep_counter,:]
                
            self.memory_one_ep = np.zeros(
                 (self.episode_max_steps, self.n_features*2 + 3 + 2*self.n_actions), dtype=np.float32)
            self.memory_normal_counter += self.memory_one_ep_counter
            self.memory_one_ep_counter = 0  

    def choose_dnn_action(self, sa):
        sa = th.unsqueeze(th.FloatTensor(sa.flatten()).to(device), 0)
        q_values = self.eval_net(sa)
        action = q_values.argmax(dim=1).reshape(-1)
        return action.item()
    
    def reward_function(self, obs_next, v, trigger, done):
        v_min = 20.0
        v_max = 30.0
        w_v = np.absolute(v - v_min)/(v_max - v_min) 
           
        c_rv = PARAM_CRV
        c_rc = PARAM_CRC
        c_rl = PARAM_CRL
        c_re = PARAM_CRE 
               
        # speed reward
        r_v = w_v
        # collision reward
        r_c = done
        # lane-keeping reward      
        r_l = 0
        for i in range(4):
            if np.absolute(obs_next[0,2] - 0.25*i) <= 0.001:
                r_l = 1               
        # ETC reward
        r_e = trigger
        
        r = c_rv*r_v + c_rc*r_c + c_rl*r_l + c_re*r_e
        
        return r
    
    def determine_trigger(self, a_current, a_last, step):
        if step == 0 or a_current != a_last:
            trigger= 1
        else:
            trigger = 0
        return trigger    

    def choose_action(self, observation, determinstic=False):  
        if determinstic:            
            return self.choose_dnn_action(observation)
        else:                
            if np.random.uniform() >= self.epsilon:  
                return np.random.randint(0, self.n_actions)                                              
            else:
                return self.choose_dnn_action(observation) 

    def save_model(self,epoch):
        print('model saved')
        th.save(self.eval_net,'training_models/etdqn_'+str(epoch)+'.pkl')
        
    def load_model(self, epoch):
        print('load model')
        self.eval_net = th.load('training_models/etdqn_'+str(epoch)+'.pkl')

    def train_sample(self, sample_normal_index, interval_steps_save):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('eval_net ---> targe_net: target_net_params_updated!')             

        batch_memory = self.memory_normal[sample_normal_index, :]

        b_s = Variable(th.FloatTensor(batch_memory[:, :self.n_features + self.n_actions])).to(device)
        b_a = Variable(th.LongTensor(batch_memory[:, self.n_features + self.n_actions].astype(int).reshape(-1, 1))).to(device)
        b_r = Variable(th.FloatTensor(batch_memory[:, self.n_features + self.n_actions + 1].reshape(-1, 1))).to(device)
        b_d = Variable(th.FloatTensor(batch_memory[:, self.n_features + self.n_actions + 2]).reshape(-1, 1)).to(device)
        b_s_ = Variable(th.FloatTensor(batch_memory[:, -(self.n_features + self.n_actions):])).to(device)

        with th.no_grad():
            # compute the next Q-values using the target network
            next_q_values = self.target_net(b_s_)

            if self.double_q:
                next_eval_values = self.eval_net(b_s_)
                actions = next_eval_values.argmax(dim=1).reshape(-1, 1)
                next_q_values = next_q_values.gather(1, actions)
            else:
                # follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)

            # 1-step TD target
            target_q_values = b_r + (1 - b_d) * self.gamma * next_q_values

        current_q_values = self.eval_net(b_s).gather(1, b_a)
        
        # compute Huber loss (less sensitive to outliers) when delta =1 : huber loss = smooth loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        self.loss = loss.item()
        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        if self.learn_step_counter == 1 or (self.learn_step_counter > 0 and (self.learn_step_counter % interval_steps_save == 0)):
            print('\n------ {} epoch save ------'.format(self.learn_step_counter))
            self.save_model(self.learn_step_counter)

    def train(self, interval_steps_save):
        if self.memory_normal_counter > self.memory_normal_size:
            sample_normal_index = np.random.choice(self.memory_normal_size, size=self.batch_normal_size, replace=False)
        else:
            sample_normal_index = np.random.choice(self.memory_normal_counter, size=self.batch_normal_size, replace=False)    
        self.train_sample(sample_normal_index, interval_steps_save)

    def learn(self, learn_start, max_time_steps, interval_steps_save):
        counter_episode = 0
        print('\n------ start learning ------')    
        print('total training steps   :', max_time_steps)
        print('batch size             :', self.batch_normal_size)
        print('target update interval :', self.replace_target_iter) 
        print('gamma                  :', self.gamma)
        print('e_greedy               :', self.epsilon_max)      
        try:
            while True:
                rewards_episode = 0
                steps_episode = 0
                
                a_prev = 3 # a_{-1}
                a_prev_oh = F.one_hot(th.as_tensor(a_prev, dtype=th.int64), self.n_actions) # one-hot coding for the augmented state                
                
                obs, info = self.env.reset()
                
                v = info.get('speed')
                done = info.get('crashed')
                
                sa = np.hstack((obs.flatten(), a_prev_oh)) # augmented state
                
                print('\n------ a new eposide ------')
                while True:                 
                    a = self.choose_action(sa) # dynamic control policy 
                    a_oh = F.one_hot(th.as_tensor(a, dtype=th.int64), self.n_actions) # one-hot coding action for the augmented state 
                    
                    trigger = self.determine_trigger(a, a_prev, steps_episode)                     
                    
                    obs_next, _, done, _, info = self.env.step(a)
                    
                    v = info.get('speed')  
                    r = self.reward_function(obs_next, v, trigger, done)            
                    sa_next = np.hstack((obs_next.flatten(), a_oh))  # augmented state                              

                    self.store_transition(sa, a, r, sa_next, done)
                    
                    if self.memory_normal_counter > learn_start:
                        self.train(interval_steps_save)

                    steps_episode += 1
                    rewards_episode += self.gamma**steps_episode * r
                    
                    obs = obs_next
                    sa = sa_next
                    a_prev = a                    
                    
                    if done or self.memory_normal_counter > max_time_steps or steps_episode >= self.episode_max_steps:
                        counter_episode += 1
                        break

                if self.memory_normal_counter > learn_start:
                    print('eposides          :', counter_episode)
                    print('time steps        :', self.memory_normal_counter)
                    print('toal reward       :', rewards_episode)
                    print('epsilon           :', self.epsilon)
                    print('loss              :', self.loss)
                    print('learning progress :', float(self.memory_normal_counter) / max_time_steps)
                    
                if self.memory_normal_counter > max_time_steps:
                    print('stop learning')
                    break
                
        except KeyboardInterrupt:
            print('stop learning due to keyboardInterrupt')

        # final save
        print('\n------ final save ------')
        self.save_model(self.learn_step_counter + 1)