# Mastery-Gated Curiosity

This project studies when curiosity helps, when it fails, and why naive prediction-error chasing is not enough.

## Repository Status

This repository is a research prototype. Some experiments are preliminary and some results are intentionally reported as negative results. The code is intended for exploration and reproducibility, not production RL training.

## Overview

A research prototype exploring curiosity-driven exploration, mastery/noise gating, prequential world-model learning, and downstream control pretraining.

## Core Idea

The agent should not simply chase high prediction error. It should use signals such as:

* learning progress
* uncertainty
* noise risk
* mastery detection
* blocking of unlearnable/noisy regimes
* prequential prediction error

## Project Phases

### Phase 1: Fixed Function Regions

5 hidden regions including one unlearnable noisy region. Compared Random, HighestError, UncertaintyOnly, LearningProgress, and MasteryGated.

### Phase 2: Hypothesis Discrimination

Added x-selection inside regions by choosing points that best distinguish candidate hypotheses.

### Phase 3: PPO Region Policy

Moved from hand-coded region selection to PPO. The first PPO policy learned strong noise avoidance but became lazy on difficult learnable regions.

### Phase 3B: Reward Ablation

Showed that reward shaping strongly changes behavior. A balanced reward worked better than an explicit difficulty bonus, which caused hyper-fixation.

### Phase 4: Procedural Worlds

Introduced randomized variable-size worlds. Flat PPO failed to generalize.

### Phase 4B: Set-Based Policy

A DeepSets-style policy with MaskablePPO improved procedural generalization by treating regions as an unordered set.

### Phase 4C: No Hidden Validation Signal

Replaced privileged validation error with prequential prediction error.

### Phase 4D: Stress Testing

Stress-tested budget robustness, deceptive functions, and no-status ablation. Key finding: status/tracking features were crucial.

### Phase 5: 2D Drone Control

Built a custom lightweight 2D drone environment after PyBullet-based drone environments failed to install on Apple Silicon. Tested curiosity pretraining before downstream hover control.

### Phase 5C: Negative Result

Tested safety-gated mastery curiosity for drone pretraining. The discrete hard-gated mastery approach failed in continuous control, while simpler curiosity performed better in some perturbation tests. Interpret this as an informative negative result, not as proof that the overall idea is invalid.

## Key Findings

* Mastery/noise gating helped in synthetic function exploration.
* PPO behavior was highly sensitive to reward design.
* Set-based representation was critical for procedural generalization.
* Prequential prediction error can replace privileged validation signals in the toy/procedural setting.
* Drone pretraining showed that pretraining can improve robustness compared to scratch in some settings.
* Hard discrete mastery/blocking did not transfer well to continuous drone control.
* The project suggests that hard gating works better in naturally segmented environments, while continuous control may require soft gating or learned continuous regime representations.

## Limitations

* This is not a general AI scientist.
* Most experiments are synthetic.
* The drone environment is a simplified custom 2D simulator.
* Some results may be single-seed or preliminary unless multi-seed evaluation exists.
* Phase 5C produced a negative result.
* The code is a research prototype, not production RL infrastructure.

## Known Issues

- Phase 5C is preliminary and should not be interpreted as a definitive rejection of mastery tracking in continuous control.
- The drone environment is a simplified custom 2D simulator.
- Some results may be based on limited training seeds.
- Hard discrete regime gating may require better action/state discretization or soft continuous gating.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Reproducing Experiments

```bash
PYTHONPATH=. python src/train_drone_world_model.py
PYTHONPATH=. python src/train_drone_pretrain.py
PYTHONPATH=. python src/train_drone_control.py
PYTHONPATH=. python src/evaluate_drone_phase5c.py
```

## Results

* `results/drone/phase5_summary.md`
* `results/drone/phase5c_summary.md`
