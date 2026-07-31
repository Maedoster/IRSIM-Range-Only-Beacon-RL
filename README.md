
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

<img width="600"  alt="demoPF" src="https://github.com/user-attachments/assets/f774fbfd-0895-438d-a484-bf1f566a5aa7" />

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

#### Training Arguments (`pixi run train`)

The training script supports several CLI arguments to configure the algorithm, state estimation backend, parallelism, and evaluation checkpoints (These can also be changed directly though the code):

| Category | Flag | Type / Choices | Description |
| :--- | :--- | :--- | :--- |
| **General** | `--algorithm` | `DDPG`, `TD3`, `PPO`, `SAC` | RL policy architecture to train |
| | `--pf-active` | `bool` | Enable/disable Particle Filter state estimation |
| | `--seed` | `int` | Random seed for reproducibility |
| **Training** | `--num-envs` | `int` | Number of parallel Gym vector environments |
| | `--total-timesteps` | `int` | Total environment interaction steps |
| | `--eval-episodes` | `int` | Number of episodes per evaluation cycle |
| | `--save-freq` | `int` | Frequency (in steps) to save model checkpoints |
| | `--eval-freq` | `int` | Frequency (in steps) to trigger evaluation |
| **Evaluation** | `--use-dummy-eval` | `bool` | Use single DummyVecEnv vs SubprocVecEnv during eval |
| | `--num-eval-envs` | `int` | Number of parallel evaluation environments |
| | `--save-eval-maps` | `bool` | Save visual trajectory maps generated during eval |
| **Resume** | `--run-folder-name` | `str` | Subfolder name under `models/` to resume |
| | `--checkpoint-name` | `str` | Specific `.zip` checkpoint file to resume training |

---

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
#### Testing CLI Arguments (`pixi run test`)

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--experiment-dir` | `str` | `run_DDPG_True_1` | Experiment subfolder name inside `models/` |
| `--model-name` | `str` | `best_model` | Model file name without extension (`.zip` / `.pkl`), it looks in the best_model/ subfolder |
| `--num-episodes` | `int` | `1000` | Total number of test episodes to execute |
| `--num-envs` | `int` | `14` | Number of parallel worker environments |
| `--seed` | `int` | `1` | Base random seed for reproducibility |
| `--render` | `flag` | `False` | Enable rendering (renders worker 0 only to prevent UI crashes) |

---

```bash
# Render live trajectory execution in IR-Sim
pixi run run
```

#### Interactive Trajectory Execution Arguments (`pixi run run`)
| Flag | Type | Description |
| :--- | :--- | :--- |
| `--experiment` | `str` | Experiment subfolder name inside `models/` |
| `--model` | `str` | Target model file name without extension |
| `--episodes` | `int` | Total number of episodes to render |
| `--seed` | `int` | Initial random seed |

---

## 🎓 Academic Context

This work was conducted as part of a Master's Thesis program in collaboration with the University of Bologna and the University of Bielefeld.

    Author: Edoardo

    First Reviewer: RA Jesus E. Aleman G.

    Second Reviewers: PD Dr.-Ing. Sven Wachsmuth, Prof. Simone Martini

## 📜 License

This repository is released under the MIT License.






