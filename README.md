This repository contains an implementation of Proximal Policy Optimization (PPO) for autonomous navigation in a corridor environment with a quadcopter. There are blocks having circular opening for the drone to go through for each 4 meters. The expectation is that the agent navigates through these openings without colliding with blocks. This project currently runs only on Windows since Unreal environments were packaged for Windows.

**Libraries & Tools**
1.Unreal Engine 4
2.Microsoft AirSim

**Environment setup to run the codes**
## 1. Clone the repository 
https://github.com/shinnew9/ReinforcementLearning-BasedDroneFlightControl

## 2.From Anaconda command prompt, create a new conda environment
I recommend you to use Anaconda or Miniconda to create a virtual environment.
conda create -n ppo_drone python==3.8

## 3.Install required libraries
conda activate ppo_drone
pip install -r requirements.txt

**How to run the pretrained model?**
Make sure you followed the instructions above to setup the environment. To speed up the training, the simulation runs at 20x speed. You may consider to change the "ClockSpeed" parameter in settings.json to 1.
# 1. Download the test environment
Go to the releases and download TestEnv.zip. After downloading completed, extract it.
# 2. Now, you can open up environment's executable file and run the trained model
python policy_run.py
