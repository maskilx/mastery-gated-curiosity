# Phase 5: Curiosity Drone Design

After demonstrating that Mastery-Gated Curiosity successfully navigated synthetic function landscapes, we migrated the concept to a physics control problem. 

## Motivation

Standard Reinforcement Learning trains an agent from scratch to solve a specific task (e.g., "hover at this target"). However, if the agent is spawned in a chaotic state during deployment, it will likely crash because it only learned the narrow "tube" of dynamics required to solve the task from its standard starting state.

We hypothesized that if an agent was allowed to freely explore the physics of the drone *before* being assigned a task, it would build a robust internal representation of the aerodynamics.

## Environment Design

We built a custom lightweight 2D differential-thrust drone simulator in Python (`src/drone_2d_env.py`).

* **State**: `[x, z, vx, vz, theta, theta_dot]`
* **Action**: `[thrust_left, thrust_right]`
* **Dynamics**: Realistic gravity and angular torque physics. 

## Approach

1. **Open-Ended Pretraining**: The agent is placed in the environment with no extrinsic goal. Its only rewards are a small survival bonus, a curiosity bonus (based on the prediction error of a continuously trained neural world model), and a novelty bonus.
2. **World Modeling**: As the agent acts, a separate neural network tries to predict the next state. The agent is rewarded for finding transitions that surprise the world model.
3. **Downstream Fine-Tuning**: After pretraining, the agent's policy weights are loaded and fine-tuned on the actual task (hovering at `z=1.0`).

## Results

Curiosity pretraining successfully resulted in an agent that achieved a lower downstream altitude error and a lower crash rate during chaotic perturbation tests than an agent trained from scratch.
