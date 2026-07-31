<img width="195" height="173" alt="DemoPF" src="https://github.com/user-attachments/assets/5004511a-40be-4a86-ab48-4dc70046bcad" />
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

## 🎬 Demo & Trajectory Execution

<img width="195" height="173" alt="DemoPF" src="https://github.com/user-attachments/assets/456344f9-b23e-4ae4-87d6-a56b2d82a4eb" />

## 🏗️ System Architecture

The figure below outlines the modular design of the system architecture, illustrating the separation between the Gym environment wrapper dynamics and the external policy optimization loop:

```text
                  +--------------------------+
                  |  Original HouseExpo Data |
                  +--------------------------+
                               | (Parsed)
                               v
                  +--------------------------+
                  |  IR-Sim HouseExpo Data   |
                  +--------------------------+
                               | (Loaded)
                               v
+--------------------------------------------------------------+
| Custom Gymnasium Wrapper                                     |
|                                                              |
|   +--------+      +--------------------------+               |
|   | IR-Sim | ---> | State Estimation (PF/LS) |               |
|   +--------+      +--------------------------+               |
|       |                        |                             |
|       +-----------> +----------------------+                 |
|                     |     Observations     |                 |
|                     +----------------------+                 |
|                                |                             |
|                                v                             |
|                     +----------------------+                 |
|                     |  Reward Calculation  |                 |
|                     +----------------------+                 |
+--------------------------------------------------------------+
            | (Observation, Reward)           ^ (Action)
            v                                 |
+--------------------------------------------------------------+
| Stable-Baselines3 Agent                                      |
+--------------------------------------------------------------+
```


## ⚙️ Installation & Setup

This project uses **[Pixi](https://pixi.sh)** for environment and task management.

### 1. Clone the Repository
```bash
git clone [https://github.com/Maedoster/IRSIM-Range-Only-Beacon-RL.git](https://github.com/Maedoster/IRSIM-Range-Only-Beacon-RL.git)
cd IRSIM-Range-Only-Beacon-RL
```
### 2. Preprocessing Data Pipeline

Run the preprocessing tasks sequentially to parse layouts, split datasets, and extract occupancy grids:

```bash
# 1. Convert raw dataset layouts into IR-Sim compatible formats
pixi run convert
```

```bash
# 2. Generate occupancy grids required for path planning & state estimation
pixi run occupancy
```

```bash
# 3. Split dataset into training, validation, and testing splits
pixi run split
```

## 🚀 Training & Evaluation
### 1. Training the Agent


```bash
# Launch the Deep RL training loop:
pixi run train
```

```bash
# Monitor training curves and evaluation metrics in real time:
pixi run tensorboard
```
### 2. Evaluating & Running Models

Ensure your target model checkpoint is inside the best_model/ folder in the trained models or a custom folder/, then run:
```bash
# Evaluate policy performance across test environments
pixi run test
```
```bash
# Render live trajectory execution in IR-Sim
pixi run run
```

## 🎓 Academic Context

This work was conducted as part of a Master's Thesis program in collaboration with the University of Bologna and the University of Bielefeld.

    Author: Edoardo

    First Reviewer: RA Jesus E. Aleman G.

    Second Reviewers: PD Dr.-Ing. Sven Wachsmuth, Prof. Simone Martini

## 📜 License

This repository is released under the MIT License.
