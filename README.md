# Autonomous Robot Navigation via Deep Reinforcement Learning with State Estimation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Gymnasium-v0.29%2B-green.svg)](https://gymnasium.farama.org/)
[![RL-Library](https://img.shields.io/badge/Stable--Baselines3-v2.0%2B-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![Simulator](https://img.shields.io/badge/Simulator-IR--Sim-red.svg)](https://github.com/Han銳/IR-Sim)

This repository contains the source code and experimental framework developed for the Master's Thesis project on **Autonomous Mobile Robot Navigation using Deep Reinforcement Learning and State Estimation**. 

The framework integrates a 2D continuous simulator (**IR-Sim**) within a custom **Gymnasium interface**, combining range-only beacon state estimation (Particle Filtering / Least Squares) with continuous Deep RL agents trained via **Stable-Baselines3**.

---

## 📌 Project Overview

Autonomous navigation in sparse-sensor environments requires robust state estimation combined with effective policy learning. This project presents an end-to-end framework where an autonomous agent navigates complex indoor layouts (derived from the **HouseExpo** dataset) using range-only beacon signals.

### Key Features
* **Custom Gymnasium Wrapper:** Encapsulates simulator dynamics, sensor processing, state estimation updates, and reward logic into a standardized RL control loop.
* **Environment Dataset Parsing:** Tools to parse, preprocess, and load high-throughput indoor layout environments from the raw HouseExpo dataset into IR-Sim maps.
* **State Estimation (PF/LS):** Integrated state estimation modules utilizing Particle Filter (PF) and Least Squares (LS) algorithms for estimating robot pose and target beacon positions.
* **Deep RL Integration:** Native support for continuous control algorithms (e.g., SAC, TD3, PPO) implemented using **Stable-Baselines3**.

---

## 🏗️ System Architecture

The figure below outlines the modular design of the system architecture, illustrating the separation between the Gym environment wrapper dynamics and the external policy optimization loop:


## ⚙️ Installation & Environment Setup

This project uses **[Pixi](https://pixi.sh)** for reproducible environment management and task execution.

### 1. Prerequisites
Install Pixi on your machine (if not already installed):
```bash
curl -fsSL [https://pixi.sh/install.sh](https://pixi.sh/install.sh) | bash

2. Clone & Initialize
Bash

git clone [https://github.com/Maedoster/IRSIM-Range-Only-Beacon-RL.git](https://github.com/Maedoster/IRSIM-Range-Only-Beacon-RL.git)
cd IRSIM-Range-Only-Beacon-RL

# Install all dependencies and build the virtual environment
pixi install

🚀 Quickstart Guide

Running the project requires executing the data generation pipeline sequentially before initiating training or evaluation.
Step 1: Prepare the Dataset Pipeline

Before running training or testing, you must convert the raw dataset, split it into train/test subsets, and pre-generate the occupancy grid maps:
Bash

# 1. Convert raw HouseExpo dataset into IR-Sim compatible formats
pixi run convert

# 2. Split dataset into training and evaluation sets
pixi run split

# 3. Generate occupancy grid maps for state estimation and collision checking
pixi run occupancy

Step 2: Training & Monitoring
Start Training

To start training the RL agent:
Bash

pixi run train

Monitor Progress via TensorBoard

To track reward curves, episode lengths, and policy loss metrics in real-time:
Bash

pixi run tensorboard

Then open http://localhost:6006 in your web browser.
Step 3: Evaluation & Visualization
Run Policy Evaluation

To evaluate a trained model's navigation performance:
Bash

pixi run test

Visualize & Render Execution

To visualize and render the live agent trajectories in IR-Sim, load the desired checkpoint from the Best_Models/ directory (e.g., pointing to Best_Models/run_SAC_True_1/best_model.zip):
Bash

pixi run run

    Note on Model Checkpoints: Ensure that pixi run run or your evaluation script is pointing to the correct model checkpoint path within the Best_Models/ folder (or specify the path in your config/arguments).
