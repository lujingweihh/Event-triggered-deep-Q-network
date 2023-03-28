# Event-triggered deep Q-network (ETDQN) for Autonomous Driving
********
Code for paper:   
[J. Lu, L. Han, Q. Wei, X. Wang, X. Dai, F.-Y. Wang. Event-triggered deep reinforcement learning using parallel control: A case study in autonomous driving. *IEEE Transactions on Intelligent Vehicles*, early access, 2023, doi: 10.1109/TIV.2023.3262132.](https://ieeexplore.ieee.org/document/10081497)
<br/>

## Description
********
Investigate the problem of event-triggered deep reinforcement learning with discrete action space and develop an ETDQN for autonomous driving, *without training an explicit triggering condition*. The implicit triggering condition and the control policy share neural network parameters. 
<br/>

## How to run
********
- create an anaconda environment via: `conda create -n etdqn pytorch=1.12.1 python=3.9.13`
- activate the anaconda environment via: `conda activate etdqn`
- install the required packages via: `pip install -r requirements.txt`
- train ETDQN via: `etdqn_train.py`
- test ETDQN via: `etdqn_test.py`
<br/>

## Citation
********
```angular2html
@article{lu2023event,
  title={Event-Triggered Deep Reinforcement Learning Using Parallel Control: A Case Study in Autonomous Driving},
  author={Lu, Jingwei and Han, Liyuan and and Wei, Qinglai and Wang, Xiao and Dai, Xingyuan and Wang, Fei-Yue}
  journal={IEEE Transactions on Intelligent Vehicles}
  doi = {10.1109/TIV.2023.3262132}
  year={2023}
}
```
