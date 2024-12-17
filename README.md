# A Journey into Dog Breed Classification with Deep Learning

Original blog post: <https://blog.anibalhsanchez.com/en/blogging/89-a-journey-into-dog-breed-classification-with-deep-learning.html>

When I first embarked on my [Udemy Data Scientist Nanodegree](https://www.udacity.com/enrollment/nd025) Program, I never anticipated it would lead me to create an algorithm capable of recognizing dog breeds from images. Coming from a background in writing a thesis on *Particle Imaging and Tracking in Branched Electrochemical Systems*, this project reconnected me with the fascinating field of image processing.

## What the Algorithm Does - Project Highlights

The system goes beyond a simple breed classifier. The detection and recognition algorithm can:

1. Detect if an image contains a dog or a human with a high level of accuracy
2. Predict the specific dog breed

## High-level Overview

Classifying dog breeds is a complex task due to several factors. Some breeds look incredibly similar, and color variations within a breed can be dramatic. Randomly guessing the dog breed would yield less than 1% accuracy.

The final project aims to develop an algorithm to detect whether an image contains a dog or a human and then predict the dog breed. The detection is based on Convolutional Neural Networks (CNNs).

The analysis tested different CNNs to evaluate the dog breed detection algorithm based on the result's accuracy.

The expected solution was an algorithm that could:

1. Accurately detect dogs and humans in images
2. Classify dog breeds with high accuracy and suggest a resembling dog breed
3. If not, confirm if the human contains a human or not (neither case)

### Strategy for Solving the Problem

The strategy involved a multi-step approach to evaluate different techniques and results:

- **Human Detection**: Using OpenCV's Haar cascade classifier
- **Dog Detection**:
  - Using a pre-trained ResNet50 model
  - Creating a Convolutional Neural Network (CNN) from scratch
  - Using Transfer Learning with ResNet50 to enhance performance

## Description of Input Data

The input data consisted of:

1. A dataset of dog images, categorized by breed (sampleImages folder, udacity-aind/dog-project/dogImages.zip)
2. A dataset of human frontal face images categorized by public figure (lfw folder, udacity-aind/dog-project/lfw.zip)
3. Test images for general evaluation (from Wikimedia Commons).

### Exploratory Data Analysis

1. **Dog Images Dataset**: Contains 8,351 images of 133 dog breeds, separated into test, training, and validation sets.
2. **Human Images Dataset**: Contains 13,233 images of 5,749 public figures.
3. **Sample Images**: Contains 20 sample images for initial tests.

### Data Preprocessing

Preprocessing steps included:

1. Resizing images to 224x224 pixels
2. Normalizing pixel values
3. Extracting bottleneck features from pre-trained models

## Modeling

The modeling process involved:

1. Using VGG16 and ResNet50 CNNs
2. Creating a basic CNN from scratch
3. Implementing transfer learning in the ResNet50 model
4. Fine-tuning models for dog breed classification

## Metrics

The primary metric used was the **classification accuracy**. The classification accuracy is a straightforward metric for multi-class classification problems. The project targets are:

1. 1% accuracy for the initial CNN model
2. 60% accuracy for the transfer learning model
3. High accuracy in dog and human detection

## Hyperparameter Tuning

Hyperparameter tuning was performed, and the adjustments included the following:

1. Learning rate
2. Number of epochs
3. Batch size

## Results and Conclusions

The final model achieved 83.01% accuracy on the test set for dog breed classification — promising results in human and dog detection, with some limitations. The project successfully developed an algorithm capable of detecting humans and dogs and accurately classifying dog breeds. Transfer learning significantly improved performance compared to the initial CNN.

### Key Technical Achievements

1. **Accuracy**: 83.01% breed classification accuracy
2. **Flexibility**: Works with various image inputs
3. **Innovation**: Leverages cutting-edge deep learning techniques

### Challenges and Learning Moments

Some interesting challenges encountered include:

1. Handling images with ambiguous subjects
2. Managing variations within dog breeds
3. Dealing with limited training data

### Comparison Table

A comparison table would likely show the performance differences between the tested models:

Model                            | Dog Breed Detection Accuracy
---------------------------------|------------------------
Custom CNN from Scratch 0.8      | 2.04
VGG16 transfer learning model    | 73.80
ResNet50 transfer learning model | 83.01

## Future Improvements

While the current version performs well, there are several possible enhancements:

1. Implementing full-body human detection to enhance human detection accuracy
2. Increasing diversity in training data
3. Improving handling of ambiguous cases
4. Exploring more advanced deep learning approaches for detection tasks

## Acknowledgments

### The Technological Toolkit

To tackle this challenge, I assembled a robust tech stack based on open-source technologies:

1. [Keras: Deep Learning for humans](https://keras.io/)
2. [Matplotlib — Visualization with Python](https://matplotlib.org/)
3. [OpenCV - Open Source Computer Vision](https://opencv.org/)
4. [scikit-learn: machine learning in Python](https://scikit-learn.org/)
5. [TensorFlow - An end-to-end open-source machine learning platform for everyone](https://www.tensorflow.org/)
6. Programming Language: Python

## License

Sample Images from [Wikimedia Commons](https://commons.wikimedia.org/). Creative Commons CC0 License.

Attribution 4.0 International - CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>

[A Journey into Dog Breed Classification with Deep Learning](https://github.com/anibalsanchez/a-journey-into-dog-breed-classification-with-deep-learning) by [Anibal H. Sanchez Perez](https://www.linkedin.com/in/anibalsanchez/) is licensed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1)