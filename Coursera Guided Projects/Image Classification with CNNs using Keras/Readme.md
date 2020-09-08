## Project
Image Classification with CNNs using Keras

## Guidelines
In this hands-on project based course, you will learn how to create a Convolutional Neural Network (CNN) in Keras with a TensorFlow backend, and you will learn to train CNNs to solve Image Classification problems. In this project, we will create and train a CNN model on a subset of the popular CIFAR-10 dataset. The same technique can be used to solve image classification problems on your own data as well. In this course, we are going to focus on two learning objectives:

    Understand how to create convolutional neural networks in Keras.
    Be able to train convolutional neural networks to solve image classification problems.

By the end of this course, you will be able to create a CNN model in Keras from scratch, and use to solve image classification problems on your own data.

## Tasks
Task 1: Introduction

    Introduction to the problem.
    Introduction to the Rhyme interface.
    Importing the required libraries and helper functions.

Task 2: Pre-process Data

    Importing the CIFAR-10 dataset.
    Creating a subset of the dataset which has just 3 classes instead of 10. This is done for both the training and test set.
    Randomly shuffling the newly created subset.

Task 3: Visualize Examples

    Plotting randomly selected examples of a given set.
    We look at some examples from training and test set along with their labels.

Task 4: Create Model

    Creating a Keras Sequential model.
    Creating a function to add a convolutional block to the model.
    A look at the model summary.

Task 5: Train the Model

    Fit the model on the subset.
    Setting the EarlyStopping callback.
    Setting the ModelCheckpoint callback.

Task 6: Final Predictions

    Plotting the training and validation accuracy from the training.
    Loading the best model.
    Getting predictions on the test set and displaying the results.
