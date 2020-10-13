# Neural Networks and Learning Systems TBMI26
Course in machine learning. Uses matlab and python for excercies.
Some files were provided for the lab, the implemented files are specificed for each lab.

## Lab 1 - KNN and NN
- Implementation of a kNN classifier, training and classification.
- Implementation of a neural network, training and classification.

### Implemented files
- kNN.m: Implementation of the kNN algorithm
- runSingleLayer.m: Implementation of the forward pass on the single-layer network
- trainSingleLayer.m: Implementation of the training of the single-layer network
- runMultiLayer.m: Implementation of the forward pass on the multi-layer network
- trainMultiLayer.m: Implementation of the training of the multi-layer network
- calcConfusionMatrix.m: Function for calculating the confusion matrix of predicted results
- calcAccuracy.m: Function For calculating the accuracy of predicted results

## Lab 2 - Ada-boost
- Implementation of a fast face detection algorithM: Viola and Jones, Robust Real-time Object Detection algorithm, using ADA-boost combined with randomly generated Haar-features.

### Implemented files
- AdaBoost: Main script for the algorithm
- WeakClassifier: Implementation of a weak classifier using a decision stump
- WeakClassifierError: Implementation to get the error of a weak classifier

## Lab 3 - CNNs and Deep Learning
- Several task on image manipulations, kernels, and convolutions.
- Implementation of LeNet5 CNN and training on CIFAR10 achieveing ~80% test accuracy. 

## Lab 4 - Reinforcement Learning
- Implementation of the Q-learning algorithm for several grid worlds.

### Implemented files
- Qlearning.m: Main file for the training algorithm
- getpolicy.m:  Implementation to get the movement policy from the Q-table
- getvalue.m:  Implementation to get the V-function from the Q-table
- getepsilon.m:  Implementation for an exploration strategy
