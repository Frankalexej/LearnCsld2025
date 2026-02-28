# Learning Consolidation
This repository contains code for running experiments on learning consolidation in neural network models. It supports controlled training regimes, multi-stage learning setups, and systematic evaluation of learning dynamics under different consolidation conditions. 

## 20251218
- In this run, we will borrow ts tc tsh tch shifted dataset from PhonGen2025. 
- We will use a simple model, say, CNN model or even non-linear FC model to as the learner. 
- The learning should better use SGD, for clarity. 
- We will first train the model on the original square, and then ask it to learn the shifted square. We expect to see some learning difference between DL and SL, and this difference will depend on whether there is L1 learning, whether there is consolidation, what kind of consolidation, etc. For this we don't really know. But at least some aspects of L1 learning should have some effect on the difference between DL and SL. 
- We will consider the following causes of change in DL-SL difference: 
    - Globally reduced learning rate compared to when learning L1; 
    - task-related learning rate tune-down -> after learning L1, those dimensions related to L1 but also in L2 are harder to learn; 
    - task-related gradient reduction: different from learning rate tuning, instead, this is about the gradient, which is rooted from loss. But I am not sure about how exactly this could ever be different from learning rate tuning; 
    - simply because of learning L1, some weights are changed, and the loss geometry or representation geometry have changed. 

- Because we are comparing DL and SL, we consider the following metrics: 
    - speed to reach the same hidden representation distance. We use this because DL does not have "performance" in terms of classification. 
    - We compare speed by checking the time needed for: (1) reaching convergence; (2) reaching the same distance. 
    - We also anticipate that different learning methods will converge at different distances, and that is also why we additionally need speed checking for reaching the same distance, instead of only reaching convergence. 
    - Because we mainly use distance metrics, we could recycle the sample test codes from PhonGen2025. 