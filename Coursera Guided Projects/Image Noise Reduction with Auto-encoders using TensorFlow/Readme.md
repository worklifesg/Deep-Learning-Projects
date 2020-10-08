## Project
Image Noise Reduction with Auto-encoders using TensorFlow
## Guidelines
In this course, we are going to focus on two learning objectives:

    Develop an understanding of Auto-encoders.
    Be able to apply Image Noise Reduction with Auto-encoders.

By the end of this course, you will be able to apply auto-encoding to reduce noise in images.

## Tasks
**Task 1: Introduction**

  We will import the libraries and helper functions that we will need during the course of this project. We will also understand a little bit about the Rhyme interface and pre-requisites for this project.

**Task 2: Data Pre-processing**

  For this project, we are using the popular MNIST data-set. This data-set has 60,000 examples of images of handwritten digits in the training set and 10,000 examples in the test set. The examples are black and white images of 28x28. As in, 28 rows and 28 columns for each example. The labels are simply digits corresponding to the 10 classes from 0 to 9. We will create two neural network models in this project - one will be trained to perform classification of these handwritten digits. And another model will be used to de-noise input data. This is our Auto-encoder. Eventually, we will connect the two models together and have them work in conjunction as a single, composite model. In order to input the examples to our two models, we will do a little bit of processing on them.

**Task 3: Adding Noise**

  We are artificially adding some noise to our training and test examples. You may wonder - why synthesize the noise to train the Auto-encoder? This is because in real world applications, while we will often get noisy data, we will not have the corresponding _clean_ labels. Instead, when we synthesize noise on already clean images, we can train an Auto-encoder to focus on the important parts of the images and then when it's applied to real world noisy data, it knows where to focus and which features to retain.

**Task 4: Building and Training a Classifier**

  In this task, we will create a classifier and train it to classify handwritten digit images. We will use a very straight-forward neural network with two hidden layers. These are fully connected or dense layers with 256 nodes each in both the layers. The output layer has 10 nodes for the 10 classes and of course, a softmax function to get the probability scores for various classes. One tricky part here could be that we need to use sparse categorical cross-entropy loss instead of the categorical cross-entropy loss that we would have used if the labels were one-hot encoded. But since the labels are numerical values from 0 to 9 for the 10 classes in a single list with one value for each label, we would need to use the sparse categorical cross-entropy.

**Task 5: Building the Auto-encoder**

  In order to reduce noise in our data, we want to create a model - the Auto-encoder - which takes a noisy example as input and the original, corresponding example as the label. Now, if one or more hidden layers in this neural network has a lot less nodes as compared to the input and output, then the training process will force the network to learn a function similar to principal component analysis, essentially reducing dimensionality. Another thing to note is that the output layer has the sigmoid activation. The higher linear values of the last layer will become closer to the maximum normalized pixel value of 1 and the low linear values will converge towards the minimum normalized pixel value 0. This choice of activation makes sense given the examples in the input are black and white images. There's some scope for having a variety of pixel values but with sigmoid most of the values will converge to either 0 or 1 and that works well for us.

**Task 6: Training the Auto-encoder**

  We will use the noisy training set examples as our examples and the original training set examples, the ones without any noise, will be used as our labels for the Auto-encoder to learn de-noising. Let's set the epochs to a somewhat higher number, 100 in this case, because we are going to use the early stopping callback. Let's use a batch size slightly higher than usual, it will help us speed up the training. We will also use a lambda callback to log out just the validation loss for each epoch. And we will set the verbose to False because we just want to see the validation loss per epoch.

**Task 7: De-noised Images**

  Now that the Auto-encoder is trained, let's put it to use. In order to get our de-noised images, say for our test data, all we have to do is pass the noisy data through the Auto-encoder! Let's use the predict method on our model to get the results. We will also pass the de-noised images through our classifier and this time, it should perform significantly better.

**Task 8: Composite Model**

  Let's create a composite model to complete our entire prediction pipeline. What I mean is that we want a model in which we can simply feed a noisy image, and the model will first reduce noise in that image and then use this output image and run it through the Classifier to get the class prediction. Idea being that even if our incoming data in a production setting is noisy, our classifier should be able to work well because of the noise reduction from the Auto-encoder.
