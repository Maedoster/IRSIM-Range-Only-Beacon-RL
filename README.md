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


---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8 or higher
* `git`

### 1. Clone the Repository
```bash
git clone [https://github.com/Maedoster/IRSIM-Range-Only-Beacon-RL.git](https://github.com/Maedoster/IRSIM-Range-Only-Beacon-RL.git)
cd IRSIM-Range-Only-Beacon-RL

2. Create and Activate Virtual Environment
Bash

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Bash

pip install --upgrade pip
pip install -r requirements.txt

🚀 Usage
Training an Agent

To start training a Soft Actor-Critic (SAC) or TD3 agent with state estimation active:
Bash

python train.py --algo sac --env houseexpo --timesteps 500000

Evaluating Trained Policies

To evaluate a trained checkpoint and visualize the trajectory in IR-Sim:
Bash

python evaluate.py --model-path Best_Models/run_SAC_True_1/best_model.zip --render

📂 Repository Structure
Plaintext

├── Best_Models/            # Saved policy weights and evaluation logs
├── datasets/               # Preprocessed HouseExpo layout files for IR-Sim
├── envs/                   # Custom Gymnasium wrapper & observation space definitions
├── state_estimation/       # Particle Filter (PF) and Least-Squares (LS) backends
├── utils/                  # Dataset parsers, metrics loggers, and plotting scripts
├── train.py                # Main script for policy training
├── evaluate.py             # Policy evaluation and trajectory rendering
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation

🎓 Academic Context

This work was conducted as part of a Master's Thesis program in collaboration with the University of Bologna and the University of Bielefeld.

    Author: Edoardo

    First Reviewer: RA Jesus E. Aleman G.

    Second Reviewers: PD Dr.-Ing. Sven Wachsmuth, Prof. Simone Martini

📜 License

This repository is released under the MIT License.
