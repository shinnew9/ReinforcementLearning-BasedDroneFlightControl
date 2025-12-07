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
```



# 3. How Training Works

The PPO agent interacts with the AirSim Blocks environment through a Gym-like API:

1. Capture 50×50×3 RGB frame from `/front_center` camera  
2. Preprocess and send observation to PPO  
3. PPO chooses one of 9 discrete velocity actions  
4. AirSim updates the drone position  
5. Environment computes reward:
   - Alignment score  
   - Forward progress  
   - Collision penalty  
6. PPO updates the actor–critic networks every 2048 steps  

All training diagnostics (KL, entropy, clip fraction, explained variance, etc.)  
are automatically logged in **TensorBoard**.



# 4. Model Architecture

The agent uses:

- **CNN** for visual feature extraction  
- **Actor network** → outputs 9 discrete body-frame velocity commands  
- **Critic network** → estimates state value \( V(s) \)  
- **PPO optimizer** → stable clipped policy updates  

High-level pipeline:
```bash
RGB Image (50×50×3)
↓
CNN Encoder
↓
Actor Head → 9 actions
Critic Head → V(s)
↓
Airsim Drone Motion
```


# 5. Results

## 5.1 Episode Length
- Converges to ~4–5 steps
- Drone achieves basic short-range stability
- Long-horizon navigation remains difficult

## 5.2 Reward Trends
- Mostly negative due to collision penalties
- Reward curve becomes structured → PPO learns stable patterns

## 5.3 PPO Diagnostics
- KL divergence stabilizes → safe updates  
- Clip fraction → near zero (convergence)  
- Entropy decreases → policy becomes deterministic  
- Explained variance → ~0.9 (critic improves)

All training plots are in `figures/`.


# 6. Demo Video

Watch the trained drone policy navigating through wall openings:

👉 **demo.mp4** (see `videos/` folder)

The agent reliably performs:
- short-range forward flight  
- coarse alignment toward the opening  
- collision avoidance in simple layouts  

While long-horizon pixel-only navigation remains difficult, this experiment demonstrates that PPO can learn meaningful visuomotor behavior from extremely limited visual input.



# 7. Limitations & Future Work

### Limitations
- No temporal modeling (agent sees only one frame at a time)
- Low-resolution 50×50 images reduce spatial precision
- Reward landscape dominated by collision penalties
- Long-horizon drift leads to early failure

### Future Work
- Frame stacking or LSTM-based policies  
- Higher-resolution cameras or multi-view inputs  
- Curriculum learning across diverse wall layouts  
- Testing SAC, TD3, or Dreamer for improved sample efficiency  
- Sim-to-real transfer on a physical quadrotor

