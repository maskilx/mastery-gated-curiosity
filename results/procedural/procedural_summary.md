# Procedural World Generation (Phase 4) Results

|                                         | Val Error (Learnable Avg)   |   Samples Wasted Inactive |   Samples Wasted Noisy |   Total Mastered Learnable |   False Masteries |   False Blocked |
|:----------------------------------------|:----------------------------|--------------------------:|-----------------------:|---------------------------:|------------------:|----------------:|
| ('ID_Medium', 'HighestError')           | 0.3151 ± 0.2522             |                      17.4 |                  722.3 |                        0.9 |                 0 |               0 |
| ('ID_Medium', 'LearningProgress')       | 0.0589 ± 0.0365             |                      67.2 |                  349.6 |                        4.6 |                 0 |               0 |
| ('ID_Medium', 'MasteryGatedHypothesis') | 0.0338 ± 0.0158             |                     135.8 |                   82.2 |                        5.2 |                 0 |               0 |
| ('ID_Medium', 'PPO_Fixed')              | 0.7899 ± 0.4458             |                       0   |                  231.8 |                        2.4 |                 0 |               0 |
| ('ID_Medium', 'PPO_Procedural')         | 1.1646 ± 0.5040             |                       0   |                  267.2 |                        1.2 |                 0 |               0 |
| ('ID_Medium', 'Random')                 | 0.0974 ± 0.0790             |                     207.7 |                  192.8 |                        4.5 |                 0 |               0 |
| ('OOD_Hard', 'HighestError')            | 0.4261 ± 0.2438             |                       0.6 |                  763.8 |                        0   |                 0 |               0 |
| ('OOD_Hard', 'LearningProgress')        | 0.2225 ± 0.1746             |                      67.4 |                  393.2 |                        4.5 |                 0 |               0 |
| ('OOD_Hard', 'MasteryGatedHypothesis')  | 0.2080 ± 0.1820             |                     117.1 |                   85.3 |                        4.6 |                 0 |               0 |
| ('OOD_Hard', 'PPO_Fixed')               | 0.5860 ± 0.2961             |                       0   |                  250.7 |                        2.2 |                 0 |               0 |
| ('OOD_Hard', 'PPO_Procedural')          | 0.7622 ± 0.5198             |                       0   |                  234.5 |                        1.1 |                 0 |               0 |
| ('OOD_Hard', 'Random')                  | 0.2341 ± 0.1951             |                     207.7 |                  185   |                        4.4 |                 0 |               0 |