# Crossmodal Attentive Skill Learner (CASL)

Hybrid CPU-GPU implementation of Crossmodal Attentive Skill Learner (CASL)  
Codebase design is based on [GA3C](https://github.com/NVlabs/GA3C/).

![CASL Amidar Gameplay](https://github.com/shayegano/CASL/raw/master/misc/casl_amidar_gameplay.gif)

#### Paper:

S. Omidshafiei, D. K. Kim, J. Pazis, and J. P. How, "Crossmodal Attentive Skill Learner", In NIPS Deep Reinforcement Learning Symposium, 2017.  
Link: https://arxiv.org/abs/1711.10314

#### Dependencies:

[TensorFlow](https://www.tensorflow.org/) is required (tested with version 1.4.0).  
For other dependencies, please refer to src/dependencies_install.sh.  

#### Frameworks:

Two frameworks are supported:
1. Option-based crossmodal attention learning (**master branch**)
2. Action-based crossmodal attention learning (**CASL-action branch**)

#### Environments:

Three environments are supported:
1. Sequential Door Puzzle
2. 2D Minecraft-like
3. Arcade Learning Environment-Audio (ALE-Audio)

For ALE-Audio, a pull request to ALE will be sent shortly.

#### To Run:
Please refer to instruction in `src/tensorflow/CASL/README.md`.

#### Primary code maintainers:
Shayegan Omidshafiei (https://github.com/shayegano)

Dong-Ki Kim (https://github.com/dkkim93)
