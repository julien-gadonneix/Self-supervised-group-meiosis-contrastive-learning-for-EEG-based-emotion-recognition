There is a self-supervised learning framework to realize EEG-based emotion classification on DEAP and SEED dataset.
Article title： Self-supervised Group Meiosis Contrastive Learning for EEG-Based Emotion Recognition
Article link： https://arxiv.org/abs/2208.00877

#

# Improvements

This is a fork from the original [repository](https://github.com/kanhaoning/Self-supervised-group-meiosis-contrastive-learning-for-EEG-based-emotion-recognition).

Implementation of the cross-subject evaluation with this framework (original: only cross-experiment within individual subject):

- use the data processing methods ending with \_ind.py.
- select the corresponding created data file for the self-supervised learning, fully supervised learning and fine-tuning steps.

The use of the DREAMER dataset is added.

The DL framework is optimized.
