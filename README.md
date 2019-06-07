# Neural Network visualization in Keras

This repository implements techniques to visualize neural networks in Keras. 
The library currently supports the following techniques:

* Visualization of convolutional and pooling filters [1]
* Visualization of convolutional and pooling activations
* Nearest neighbors in feature space [2]
* Maximal image generation [3]
* Vanilla saliency maps [3]
* Smooth Grad saliency maps [4]
* Integrated Gradients saliency maps [5]
* Grad-CAM saliency maps [6]

Some of these techniques are only applicable to convolutional network and layers. This is spelled out explicitly in the documentation.

The _visualization.py_ module contains functions to directly display the output of each of the above technique using matplotlib.
In alternative, the package _saliency_ and the module _activation_maximization_ can be used to obtain raw saliency maps
and  maximal images as numpy array.

This library is compatible and has been tested with python 3.7 and python 2.7.

## Usage
Please refer to the Juptyer notebook for example usage.

## Required python packages

* tensorflow
* keras 
* pillow
* matplotlib
* sklearn


## References

[1] M. D. Zeiler, R. Fergus. _Visualizing and Understanding Convolutional Networks_. https://arxiv.org/abs/1311.2901

[2] A. Krizhevsky , I. Sutskever , G. E. Hinton. _Imagenet classification with deep convolutional neural networks_. https://dl.acm.org/citation.cfm?id=3065386

[3] K. Simonyan, A. Vedaldi, A. Zisserman. _Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps. https://arxiv.org/abs/1312.6034

[4] D. Smilkov, N. Thorat, B. Kim, F. Viégas, M. Wattenberg. _SmoothGrad: removing noise by adding noise_. https://arxiv.org/abs/1706.03825

[5] M. Sundararajan, A. Taly, Q. Yan. _Axiomatic Attribution for Deep Networks_. https://arxiv.org/abs/1703.01365

[6] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, D. Batra. _Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization_.
https://arxiv.org/abs/1610.02391
