# Project Overview

Mastery-Gated Curiosity is a research prototype investigating how autonomous agents can balance exploration, noise avoidance, and mastery to efficiently learn the dynamics of their environment. 

This project explores moving from traditional "chase the highest prediction error" curiosity formulations to a more structured, safety- and progress-aware approach. 

## Key Mechanisms Explored

1. **Learning Progress**: Tracking the derivative of prediction error over time, rather than the absolute magnitude, to identify regions where learning is actively occurring.
2. **Mastery Detection**: Identifying when a region of the state space has been sufficiently learned, allowing the agent to move on to harder challenges.
3. **Noise/Chaos Avoidance**: Detecting regions where prediction error remains high but learning progress is zero (e.g. true randomness or chaotic physical dynamics) and actively blocking the agent from wasting budget there.
4. **Prequential Error**: Using a strict "predict before observing" online error metric as a proxy for true generalization error, avoiding the need for a privileged oracle validation set.
5. **Procedural Generalization**: Testing whether policies trained on randomized sets of challenges can learn generalizable exploration strategies.
6. **Continuous Physics Pretraining**: Evaluating whether curiosity-driven pretraining improves robustness and data-efficiency for downstream continuous control tasks (like hovering a drone).
