# Team 14 Final project
## Revisiting Rapid Motor Adaptation for Quadruped Locomotion
Experimental results demonstrate that the proposed simplified RMA architecture
achieves higher robustness than a standard non-adaptive policy baseline, while
preserving training stability and significantly reducing computational overhead.
These findings indicate that RMA-inspired adaptation mechanisms remain effective
even with constrained model capacity, highlighting their practicality for educational
use and rapid prototyping.
## Environmemt
### Installing Isaac Sim
You can create the Isaac Lab environment using the following commands.
```bash
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
```
Ensure the latest pip version is installed. To update pip, run the following command from inside the virtual environment:
```bash
pip install --upgrade pip
```
Install Isaac Sim pip packages:
```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```
Install a CUDA-enabled PyTorch build that matches your system architecture:
```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```
### Installing Isaac Lab
#### Cloning Isaac lab
```bash
cd /payh/to/RL_Final_project
git clone https://github.com/isaac-sim/IsaacLab.git
```
#### Installation
Run the install command that iterates over all the extensions in source directory and installs them using pip
```bash
sudo apt install cmake build-essential
cd IsaacLab/
./isaaclab.sh --install
```
### Installing Unitree RL Lab
Use a python interpreter that has Isaac Lab installed, install the library in editable mode using:
```bash
cd ..
cd untree_rl_lab
./unitree_rl_lab.sh -i
```

## 1. Training RL policy

### 1-1. Framework
We use the unitree_rl_lab training framework. Within this framework, we incorporate the friction coefficient and restitution coefficient in the environment as observed parameters of the policy, enabling the agent to generate different strategies based on the current environmental variables.
![image](https://github.com/Jeff900804/RL/blob/main/image/framework1.png)

### 1-2. Envs setting  
First, we add a foot_friction in [velocity_env_cfg.py](./unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py) class ObservationCfg. (Just introduce what we do.)
```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        ...

        foot_friction = ObsTerm(
           func=mdp.foot_friction_4legs,
           clip=(-10.0, 10.0),
        )
        ...

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        ...
        foot_friction = ObsTerm(
           func=mdp.foot_friction_4legs,
           clip=(-10.0, 10.0),
        )
```
### 1-3. Training RL policy   
Training times: 10000  
num_envs: 4096
```python
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity --headless  
```

## 2. Training estimator
### 2-1. Framework
We designed an estimator to estimate the friction coefficient and coefficient of restitution in the current environment. The dataset is collected by randomly assigning friction coefficients (0.1-1.2) and coefficients of restitution (0.0-0.3) within a simulated environment. We collect the current ground truth and the state and action data from the past 50 records, and calculate the loss.
![image](https://github.com/Jeff900804/RL/blob/main/image/framework2-1.png)
### 2-2. Collect dataset
The computer system uses 64GB of RAM, archives up to 3 episodes at a time, and stores data as a dataset with dimensions (1495200*2850). To obtain more data, we retrieved the friction_dataset_0-3.npz three times.
```python
python scripts/rsl_rl/collect_friction_dataset.py    --task Unitree-Go2-Velocity     --device cuda:0     --episodes 3     --max_steps 1000     --history_len 50     --output friction_dataset_0.npz --num_envs 512  --checkpoint ./logs/rsl_rl/unitree_go2_velocity/2025-12-11_20-55-57_mu_0.1-1.2/model_9999.pt --headless
```
### 2-3. Training estimator
The estimator is trained using three datasets at a time and archived into a single best_model (the model with the lowest loss is retrieved each time).
```python
python scripts/rsl_rl/train_friction_estimator.py     --data_prefix friction_dataset_     --num_shards 3     --batch_size 2048     --epochs 100     --lr 1e-3     --device cuda:0
```  
## 3. Play the agent with estimator in simulation
### 3-1. Framework
To enable future real-world demonstrations, we assigned commands for the agent's linear and angular velocities. The coefficient of friction and coefficient of restitution were estimated using our designed estimator and input into the policy at a frequency set to 10Hz.
![image](https://github.com/Jeff900804/RL/blob/main/image/framework3.png)
### 3-2. Play
```python
python scripts/rsl_rl/play_with_estimator.py     --task Unitree-Go2-Velocity     --device cuda:0     --num_envs 128     --checkpoint ./logs/rsl_rl/unitree_go2_velocity/2025-12-11_20-55-57_mu_0.1-1.2/model_9999.pt 
```
