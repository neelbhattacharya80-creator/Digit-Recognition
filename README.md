# Handwritten Digit Recognition from Scratch

A complete implementation of handwritten digit recognition using three progressively more capable neural network architectures:

* **Multi-Layer Perceptron (NumPy)**
* **Convolutional Neural Network (NumPy)**
* **Convolutional Neural Network (PyTorch)**

The objective of this project was not simply to classify MNIST digits, but to understand how modern neural networks work internally by implementing forward propagation, backpropagation, optimization, and image preprocessing from scratch before comparing the results with a production deep learning framework.

---

# Overview

The project began with a fully connected neural network (MLP) implemented entirely in NumPy, including manually derived backpropagation. While the model achieved **98% test accuracy**, it struggled with real handwritten images because flattening an image removes all spatial information.

To address these limitations, I implemented a custom CNN from scratch. This required deriving convolutional backpropagation, implementing convolution efficiently using the **im2col** technique, and optimizing the code to avoid expensive Python operations wherever possible. Although the network was intentionally shallow due to implementation complexity, it generalized noticeably better to real images.

Finally, I recreated the architecture using PyTorch to compare a framework implementation against the custom models. The deeper network achieved **99% test accuracy** while producing much higher confidence on real-world inputs.

---

# Learning Outcomes

This project significantly improved both my mathematical understanding and implementation skills.

### Multi-Layer Perceptron

* Implemented forward propagation from scratch
* Derived and implemented backpropagation manually
* Built a dynamic architecture supporting any number of layers and hidden sizes
* Implemented mini-batch gradient descent
* Learned weight initialization, activation functions, regularization and optimization

### Custom CNN

* Implemented the complete convolution pipeline from scratch
* Learned why CNNs outperform fully connected networks on image data
* Implemented convolution using **im2col + matrix multiplication**
* Implemented max pooling and pooling backpropagation
* Derived convolutional backpropagation manually
* Learned how stride affects gradient computation
* Implemented gradient computation using **im2col** to avoid explicit dilated convolution
* Developed a much deeper mathematical understanding of CNNs
* Improved understanding of bias-variance tradeoffs and feature learning

### PyTorch CNN

* Learned how high-level frameworks abstract neural network implementation
* Built and trained a deeper CNN using GPU acceleration when available
* Compared framework performance against a manual implementation
* Observed the practical benefits of deeper architectures

---

# Implementation Process

## 1. Multi-Layer Perceptron (NumPy)

The project started with a fully connected neural network implemented entirely from scratch.

The network supports an arbitrary number of hidden layers and hidden units by constructing the weight matrices dynamically rather than hardcoding the architecture.

Backpropagation was derived manually and implemented using NumPy matrix operations.

### Results

* **98% MNIST test accuracy**
* Dynamic architecture
* Fully manual backpropagation

### Limitations

Although the model performed well on MNIST, it generalized poorly to real handwritten images.

The primary reason is that flattening a 28×28 image into a vector destroys spatial information. Small translations or shifts in a digit activate completely different neurons, making the network highly sensitive to positioning.

Real images also required extensive preprocessing to resemble MNIST samples. While the preprocessing pipeline performs reasonably well, difficult edge cases still exist where preprocessing itself degrades prediction quality.

---

## 2. Custom CNN (NumPy)

The CNN was built specifically to overcome the weaknesses of the MLP.

Instead of treating every pixel independently, convolution allows filters to learn local spatial features while sharing weights across the image.

The largest challenge was computational efficiency.

A straightforward implementation using nested Python loops was prohibitively slow, so convolution was rewritten using the **im2col** technique. Image patches are extracted into columns before performing convolution as a single matrix multiplication, replacing thousands of small operations with optimized linear algebra.

Patch extraction still requires looping through the image, which remains the primary performance bottleneck.

### Backpropagation Challenges

Deriving convolutional backpropagation was significantly harder than the MLP.

Initially I assumed the kernel gradient could be computed directly as:

```
ΔK = X * ΔZ
```

However, this failed because the convolution used a stride of 2, causing shape mismatches during gradient computation.

The correct solution involves dilated convolution. Instead of explicitly implementing dilation, I reused the **im2col** representation to compute the gradients efficiently through matrix multiplication.

Looking back, using a stride of 1 would have made both the mathematics and implementation considerably simpler.

Another lesson was software design. The network was implemented as a single CNN class. In retrospect, separating convolution, pooling, normalization, activation functions and fully connected layers into individual modules would have made the code cleaner and easier to extend.

### Results

* **97% MNIST test accuracy**
* Better generalization to real handwritten images
* Higher prediction confidence than the MLP
* Still limited by having only one convolutional layer

---

## 3. PyTorch CNN

The final implementation recreates the project using PyTorch.

With automatic differentiation handling gradient computation, the focus shifted from deriving mathematics to experimenting with deeper architectures.

The deeper CNN achieved **99% test accuracy** while producing significantly more confident predictions on real images.

Most correctly classified samples achieve confidence scores between **95–99%**, with remaining failures largely caused by preprocessing edge cases rather than the classifier itself.

This implementation clearly demonstrates the advantages of deeper convolutional architectures over both the custom CNN and the original MLP.

---

# Features

## MLP (NumPy)

* Dynamic architecture
* Manual forward propagation
* Manual backpropagation
* Mini-batch gradient descent
* L2 regularization
* Custom image preprocessing
* Model saving/loading
* MNIST evaluation
* Real image prediction

---

## Custom CNN (NumPy)

* Manual convolution
* Manual convolution backpropagation
* im2col optimization
* Max pooling
* Dropout
* Momentum SGD
* Softmax classifier
* Model saving/loading
* Real image prediction
* Image preprocessing pipeline

---

## PyTorch CNN

* Deep CNN architecture
* GPU support
* Automatic differentiation
* DataLoader training
* Dropout
* Model saving/loading
* Real image prediction
* Image preprocessing

---

# Sample Output

Prediction examples for the custom CNN can be found in:

```
/preds_custom_cnn

```

These examples show:

* Original handwritten image
* Processed network input
* Predicted digit
* Confidence score
* Probability distribution across all classes

---

# Results Summary

| Model         | Test Accuracy | Real Image Performance                        |
| ------------- | ------------: | --------------------------------------------- |
| MLP (NumPy)   |       **98%** | Poor due to loss of spatial information       |
| CNN (NumPy)   |       **97%** | Better generalization and confidence          |
| CNN (PyTorch) |       **99%** | Best overall performance with high confidence |

---

# Future Improvements

* Modular CNN implementation (Conv, Pooling, BatchNorm, etc. as separate classes)
* Multiple convolutional layers in the custom CNN
* Adam optimizer
* Better real-image preprocessing pipeline
* Data augmentation
* Remove remaining Python loops from im2col patch extraction using strides
* GPU implementation for the NumPy CNN using cupy
