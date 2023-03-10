# ProtSRDQN

This repository provides the code for the manuscript submitted to TKDE, which is "***Exploring Cost-Effective and Explainable Job
Skill Recommendation: A Prototype-enhanced
Deep Reinforcement Learning Approach***". 

This work is a substantially extended and revised version of our previous paper [Cost-Effective and Interpretable Job Skill Recommendation with Deep Reinforcement Learning](https://dl.acm.org/doi/abs/10.1145/3442381.3449985), which appears in the Proceedings of the Web Conference 2021 (WWW’ 2021). The code is based on [SkillRec](https://github.com/sunyinggilly/SkillRec) and [TensorFlow 1.15](https://github.com/NVIDIA/tensorflow).

## Code Instructions

### CONFIG.py
Configurations, "HOME_PATH" should be properly set before runing the code.

### Sampler.py
Code for sampling actions with different strategies.

### Prepare_code
Pre-processing codes, such as generating the skill graph and frequent skill sets.

### Environment
The code for the environment, including source code written in C++ and python packages for Linux built with [swig](https://www.swig.org).

### Model
Containing our model and the baseline methods in the paper.

### Trainers
Load the training data, train the models, save the models and evaluate the models.

