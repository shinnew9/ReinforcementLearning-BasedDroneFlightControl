# End-to-End Reinforcement Learning for Autonomous UAV Flight Using RGB Images

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
