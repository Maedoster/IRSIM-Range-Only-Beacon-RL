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
