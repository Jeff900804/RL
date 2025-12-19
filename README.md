# Team 14 Final project
## Revisiting Rapid Motor Adaptation for Quadruped Locomotion
abstract...
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

## 1.Training (RL policy)

1-1.Framework
We use the unitree_rl_lab training framework. Within this framework, we incorporate the friction coefficient and restitution coefficient in the environment as observed parameters of the policy, enabling the agent to generate different strategies based on the current environmental variables.
![image](https://github.com/Jeff900804/RL/blob/main/image/framework1.png#w60)

1-2.
```python
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity --headless  
```
