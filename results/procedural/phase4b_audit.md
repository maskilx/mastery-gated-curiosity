# Phase 4B Validation Audit

## 1. Static Leakage Audit
A formal static review was conducted, supplemented by a unit test suite (`src/test_audit.py`).
- **Observation Encapsulation**: The PPO agent only receives a strict 100-dimensional vector containing `active_mask`, `num_samples`, `val_error`, `uncertainty`, `learning_progress`, `noise_risk`, and a one-hot `status`. It is categorically blocked from accessing the true `function_family`, difficulty labels, or generative parameters.
- **Validation Integrity**: Validation `x` samples are drawn identically and independently at environment initialization and are mutually disjoint from the agent's interaction samples. Validation samples never enter the agent's internal memory buffer.
- **Reward Integrity**: The smoothed reward calculation structurally queries `is_learnable` to absolutely exclude unlearnable/noisy regions from the mean validation error improvement metric, preventing perverse incentivization.

## 2. Environment Distribution Profiling
The following metrics summarize 100 sampled environments for both ID and OOD distributions.

### ID_Medium
- Active Regions (Mean): 6.96
- Learnable Regions (Mean): 5.55
- Noisy Regions (Mean): 1.41
- Measurement Noise Range: [0.0300, 1.4988]
- Sine Frequencies Range: [3.1842, 9.2863]
- Function Family Breakdown:
  - cubic: 15.4%
  - linear: 13.4%
  - noisy_unlearnable: 20.3%
  - piecewise_linear: 17.2%
  - quadratic: 17.7%
  - sine: 16.1%

### OOD_Hard
- Active Regions (Mean): 6.96
- Learnable Regions (Mean): 5.50
- Noisy Regions (Mean): 1.46
- Measurement Noise Range: [0.0501, 1.4907]
- Sine Frequencies Range: [6.5618, 18.8439]
- Exponential (a) Range: [-1.4849, 1.4906]
- Logarithmic (a) Range: [-1.4579, 1.4994]
- Function Family Breakdown:
  - cubic: 14.9%
  - exp: 13.6%
  - log: 12.9%
  - noisy_unlearnable: 21.0%
  - piecewise_linear: 12.9%
  - quadratic: 11.1%
  - sine: 13.5%

## 3. Evaluation Audit & Oracle Baselines
Ablation of X-selection and evaluation against Oracle baselines.

|                                        |   ('ValError', 'mean') |   ('ValError', 'std') |   ('SamplesWastedInactive', 'mean') |   ('SamplesWastedNoisy', 'mean') |   ('MasteredLearnable', 'mean') |   ('FalseMastery', 'mean') |   ('FalseBlocked', 'mean') |
|:---------------------------------------|-----------------------:|----------------------:|------------------------------------:|---------------------------------:|--------------------------------:|---------------------------:|---------------------------:|
| ('ID_Medium', 'OracleBestRegion')      |                 0.028  |                0.0296 |                                   0 |                                0 |                             5.9 |                          0 |                          0 |
| ('ID_Medium', 'OracleRandomLearnable') |                 0.0581 |                0.0657 |                                   0 |                                0 |                             5.7 |                          0 |                          0 |
| ('ID_Medium', 'PPO_SetPolicy')         |                 0.0077 |                0.0039 |                                   0 |                                0 |                             7.7 |                          0 |                          0 |
| ('ID_Medium', 'PPO_SetPolicy_RandomX') |                 0.0043 |                0.0024 |                                   0 |                                0 |                             7.7 |                          0 |                          0 |
| ('OOD_Hard', 'OracleBestRegion')       |                 0.1765 |                0.2193 |                                   0 |                                0 |                             5.1 |                          0 |                          0 |
| ('OOD_Hard', 'OracleRandomLearnable')  |                 0.1591 |                0.216  |                                   0 |                                0 |                             5.1 |                          0 |                          0 |
| ('OOD_Hard', 'PPO_SetPolicy')          |                 0.007  |                0.0047 |                                   0 |                                0 |                             7.7 |                          0 |                          0 |
| ('OOD_Hard', 'PPO_SetPolicy_RandomX')  |                 0.0071 |                0.0063 |                                   0 |                                0 |                             7.7 |                          0 |                          0 |
