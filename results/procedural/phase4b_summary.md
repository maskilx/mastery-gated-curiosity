# Phase 4B Results

## Overall Performance
|                                         | Val Error       |   SamplesWastedInactive |   SamplesWastedNoisy |   MasteredLearnable |   FalseMastery |   FalseBlocked |
|:----------------------------------------|:----------------|------------------------:|---------------------:|--------------------:|---------------:|---------------:|
| ('ID_Medium', 'HighestError')           | 0.3257 ± 0.2720 |                       0 |                716.9 |                 1.1 |            0   |              0 |
| ('ID_Medium', 'LearningProgress')       | 0.0580 ± 0.0403 |                       0 |                374.8 |                 5   |            0   |              0 |
| ('ID_Medium', 'MasteryGatedHypothesis') | 0.0376 ± 0.0441 |                       0 |                 86.3 |                 5.6 |            0   |              0 |
| ('ID_Medium', 'PPO_FlatMLP')            | 0.6860 ± 0.3242 |                       0 |                103.3 |                 1.2 |            0.1 |              0 |
| ('ID_Medium', 'PPO_SetPolicy')          | 0.0054 ± 0.0036 |                       0 |                227.6 |                 6   |            1.9 |              0 |
| ('ID_Medium', 'Random')                 | 0.0828 ± 0.0838 |                       0 |                227   |                 4.9 |            0   |              0 |
| ('OOD_Hard', 'HighestError')            | 0.4576 ± 0.2782 |                       0 |                764.6 |                 0.1 |            0   |              0 |
| ('OOD_Hard', 'LearningProgress')        | 0.2111 ± 0.1910 |                       0 |                421.5 |                 4.4 |            0   |              0 |
| ('OOD_Hard', 'MasteryGatedHypothesis')  | 0.1866 ± 0.1703 |                       0 |                 75.2 |                 4.6 |            0   |              0 |
| ('OOD_Hard', 'PPO_FlatMLP')             | 0.9086 ± 0.5136 |                       0 |                195.4 |                 1   |            0.2 |              0 |
| ('OOD_Hard', 'PPO_SetPolicy')           | 0.0042 ± 0.0026 |                       0 |                232.7 |                 6.1 |            1.8 |              0 |
| ('OOD_Hard', 'Random')                  | 0.2058 ± 0.1840 |                       0 |                235   |                 4.6 |            0   |              0 |

## Performance by Number of Active Regions
|                                             |   ValError |
|:--------------------------------------------|-----------:|
| ('ID_Medium', 6, 'HighestError')            |     0.1452 |
| ('ID_Medium', 6, 'LearningProgress')        |     0.0449 |
| ('ID_Medium', 6, 'MasteryGatedHypothesis')  |     0.0098 |
| ('ID_Medium', 6, 'PPO_FlatMLP')             |     0.8118 |
| ('ID_Medium', 6, 'PPO_SetPolicy')           |     0.0054 |
| ('ID_Medium', 6, 'Random')                  |     0.056  |
| ('ID_Medium', 8, 'HighestError')            |     0.3342 |
| ('ID_Medium', 8, 'LearningProgress')        |     0.0415 |
| ('ID_Medium', 8, 'MasteryGatedHypothesis')  |     0.0126 |
| ('ID_Medium', 8, 'PPO_FlatMLP')             |     0.7663 |
| ('ID_Medium', 8, 'PPO_SetPolicy')           |     0.005  |
| ('ID_Medium', 8, 'Random')                  |     0.0234 |
| ('ID_Medium', 9, 'HighestError')            |     0.2866 |
| ('ID_Medium', 9, 'LearningProgress')        |     0.0563 |
| ('ID_Medium', 9, 'MasteryGatedHypothesis')  |     0.0838 |
| ('ID_Medium', 9, 'PPO_FlatMLP')             |     0.4701 |
| ('ID_Medium', 9, 'PPO_SetPolicy')           |     0.0068 |
| ('ID_Medium', 9, 'Random')                  |     0.1346 |
| ('ID_Medium', 10, 'HighestError')           |     0.9596 |
| ('ID_Medium', 10, 'LearningProgress')       |     0.1515 |
| ('ID_Medium', 10, 'MasteryGatedHypothesis') |     0.0572 |
| ('ID_Medium', 10, 'PPO_FlatMLP')            |     0.7158 |
| ('ID_Medium', 10, 'PPO_SetPolicy')          |     0.0025 |
| ('ID_Medium', 10, 'Random')                 |     0.1862 |
| ('OOD_Hard', 6, 'HighestError')             |     0.4223 |
| ('OOD_Hard', 6, 'LearningProgress')         |     0.1724 |
| ('OOD_Hard', 6, 'MasteryGatedHypothesis')   |     0.1241 |
| ('OOD_Hard', 6, 'PPO_FlatMLP')              |     1.2346 |
| ('OOD_Hard', 6, 'PPO_SetPolicy')            |     0.0057 |
| ('OOD_Hard', 6, 'Random')                   |     0.1491 |
| ('OOD_Hard', 8, 'HighestError')             |     0.424  |
| ('OOD_Hard', 8, 'LearningProgress')         |     0.1641 |
| ('OOD_Hard', 8, 'MasteryGatedHypothesis')   |     0.147  |
| ('OOD_Hard', 8, 'PPO_FlatMLP')              |     0.7579 |
| ('OOD_Hard', 8, 'PPO_SetPolicy')            |     0.0021 |
| ('OOD_Hard', 8, 'Random')                   |     0.1786 |
| ('OOD_Hard', 9, 'HighestError')             |     0.4937 |
| ('OOD_Hard', 9, 'LearningProgress')         |     0.2531 |
| ('OOD_Hard', 9, 'MasteryGatedHypothesis')   |     0.2485 |
| ('OOD_Hard', 9, 'PPO_FlatMLP')              |     0.76   |
| ('OOD_Hard', 9, 'PPO_SetPolicy')            |     0.0037 |
| ('OOD_Hard', 9, 'Random')                   |     0.2486 |
| ('OOD_Hard', 10, 'HighestError')            |     0.5564 |
| ('OOD_Hard', 10, 'LearningProgress')        |     0.3419 |
| ('OOD_Hard', 10, 'MasteryGatedHypothesis')  |     0.3075 |
| ('OOD_Hard', 10, 'PPO_FlatMLP')             |     0.8281 |
| ('OOD_Hard', 10, 'PPO_SetPolicy')           |     0.0075 |
| ('OOD_Hard', 10, 'Random')                  |     0.3291 |