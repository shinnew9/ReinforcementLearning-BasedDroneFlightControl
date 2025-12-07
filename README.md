# Vision-Based Reinforcement Learning for Autonomous Drone Navigation in AirSim

This project implements a **vision-based reinforcement learning** system for autonomous drone navigation inside **Microsoft AirSim**.  
The drone receives *only* **50×50×3 RGB images** from its front camera and must learn to:

- Detect a wall opening  
- Align itself to the center  
- Move forward through the gap  
- Avoid collisions  

No depth, LiDAR, IMU, GPS, semantic labels, or classical planning are used.  
The system learns everything **end-to-end**:

**Raw Pixels → CNN → PPO → Drone Actions**

---

#  Key Features

- **End-to-End RL**: learns navigation from vision only  
- **PPO (Proximal Policy Optimization)** with CNN backbone  
- **9-Action discrete controller** (up/down/left/right/diagonals/forward/hover)  
- **Custom AirSim Gym-like environment**  
- **Reward shaping** for alignment, progress & collision avoidance  
- **14 TensorBoard diagnostic plots** for full PPO analysis  
- Training + testing scripts with reproducible settings  
- Entire project designed for research & report writing  

---

#  1. Installation

## Step 1 — Install Anaconda
Download: https://www.anaconda.com/products/distribution

## Step 2 — Create and activate environment
```bash
conda create -n ppo_drone python=3.8 -y
conda activate ppo_drone

## Step 3 - Install dependencies
```bash
pip install airsim
pip install stable-baselines3[extra]
pip install torch torchvision torchaudio
pip install opencv-python
pip install tensorboard
pip install matplotlib


# 2. AirSim Setup

## Step 1 — Install Unreal Engine 4.27
Install UE 4.27 via Epic Games Launcher.

## Step 2 — Download AirSim
```bash
git clone https://github.com/microsoft/AirSim.git


## Step 3 - Open the simulation environment:
```bash
AirSim/Unreal/Environments/Blocks/Blocks.uproject


# 3. AirSim Settings Configuration


```bash
Documents/AirSim/settings.json
```bash
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 20.0,
  "Vehicles": {
    "drone1": {
      "VehicleType": "SimpleFlight",
      "AutoCreate": true,
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 50,
              "Height": 50
            }
          ]
        }
      }
    }
  }
}

# 4. Repository Structure
```bash
PPO-Drone-Navigation/
│
├── main.py                     # Training script
├── policy_run.py               # Testing/inference script
│
├── scripts/
│   ├── airsim_env.py           # Custom AirSim RL environment
│   ├── client.py               # AirSim API wrapper
│   ├── utils.py
│   └── config.yml              # Environment configuration
│
├── models/
│   └── best_model.zip          # Saved PPO model
│
├── logs/
│   ├── tb_logs/                # TensorBoard logs
│   ├── training_results.txt
│   └── testing_results.txt
│
├── figures/                    # All 14 training plots
│   ├── train_entropy_loss.png
│   ├── train_clip_fraction.png
│   └── ... (remaining plots)
│
└── videos/
    └── demo.mp4                # Navigation demo video

# 5. Training the PPO Agent
```bash
python main.py
```bash
models/best_model.zip

# 6. Testing PPO Policy
## Step 1 — Open Unreal Engine → Blocks → Play
## Step 2 — Run:
```bash
python policy_run.py

7. TensorBoard Visualization
```bash
tensorboard --logdir logs/tb_logs

Images/
## PPO Training Plots (All 14 Metrics)

### **Entropy Loss**
![Entropy Loss](https://github.com/<username>/<repo>/blob/main/entropy_loss.jpg?raw=true)

### **KL Divergence**
![KL Divergence](https://github.com/<username>/<repo>/blob/main/kl_divergence.jpg?raw=true)

### **Value Loss**
![Value Loss](https://github.com/<username>/<repo>/blob/main/value_loss.jpg?raw=true)

### **Policy Loss**
![Policy Loss](https://github.com/<username>/<repo>/blob/main/policy_loss.jpg?raw=true)

### **Total Loss**
![Total Loss](https://github.com/<username>/<repo>/blob/main/total_loss.jpg?raw=true)

### **Clip Fraction**
![Clip Fraction](https://github.com/<username>/<repo>/blob/main/clip_fraction.jpg?raw=true)

### **Training Reward**
![Training Reward](https://github.com/<username>/<repo>/blob/main/training_reward.jpg?raw=true)

### **Training Episode Length**
![Episode Length](https://github.com/<username>/<repo>/blob/main/episode_length.jpg?raw=true)

### **Evaluation Reward**
![Eval Reward](https://github.com/<username>/<repo>/blob/main/eval_reward.jpg?raw=true)

### **Evaluation Episode Length**
![Eval Length](https://github.com/<username>/<repo>/blob/main/eval_length.jpg?raw=true)

### **Learning Rate**
![Learning Rate](https://github.com/<username>/<repo>/blob/main/learning_rate.jpg?raw=true)

### **FPS**
![FPS](https://github.com/<username>/<repo>/blob/main/fps.jpg?raw=true)

### **Advantage Estimates**
![Advantage Estimates](https://github.com/<username>/<repo>/blob/main/advantages.jpg?raw=true)

### **Loss Components (Combined)**
![Loss Combined](https://github.com/<username>/<repo>/blob/main/loss_combined.jpg?raw=true)



Videos/

Results/





