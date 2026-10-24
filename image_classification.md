<!-- hide -->

# RNA for image classification - Step by step guide

<!-- endhide -->

- Understanding a new dataset.
- Model the data using an ANN.
- Analyze the results and optimize the model.

## 🌱 How to start this project

Follow the instructions below:

1. Create a new repository based on [machine learning project](https://github.com/4GeeksAcademy/machine-learning-python-template) by [clicking here](https://github.com/4GeeksAcademy/machine-learning-python-template/generate).
2. Open the newly created repository in Codespace using the [Codespace button extension](https://docs.github.com/en/codespaces/developing-in-codespaces/creating-a-codespace-for-a-repository#creating-a-codespace-for-a-repository).
3. Once the Codespace VSCode has finished opening, start your project by following the instructions below.

## 🚛 How to deliver this project

Once you have finished solving the exercises, be sure to commit your changes, push them to your repository, and go to 4Geeks.com to upload the repository link.

## 📝 Instructions

### Image classification system

The dataset is composed of dog and cat photos provided as a subset of photos from a much larger collection of 3 million manually annotated photos. This data was obtained through a collaboration between Petfinder.com and Microsoft.

The data set was originally used as a CAPTCHA, i.e., a task that a human is believed to find trivial, but that a machine cannot solve, which is used on websites to distinguish between human users and bots. The task was named "Asirra". When "Asirra" was introduced, it was mentioned "that user studies indicate that humans can solve it 99.6% of the time in less than 30 seconds." Barring a breakthrough in computer vision, we expect that computers will have no more than a 1/54,000 chance of solving it.

At the time the competition was published, the state-of-the-art result was achieved with an SVM and was described in a 2007 paper with the title "Machine Learning Attacks against Asirra's CAPTCHA" (PDF) that achieved 80% classification accuracy. It was this paper that showed that the task was no longer a suitable task for a CAPTCHA shortly after the task was proposed.

#### Step 1: Loading the dataset

The dataset is available at this [link](https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip). Download the folder and unzip the files. You will now have a folder containing the dataset and a subfolder named `train`, which includes over 25,000 image files (in `.jpg` format) of dogs and cats. The photos are labeled by their filenames, with the word `dog` or `cat`.

Download the dataset folder and unzip the files. You will now have a folder called `train` containing 25,000 image files (.jpg format) of dogs and cats. The pictures are labeled by their file name, with the word `dog` or `cat`.

#### Step 2: Visualize the input information

The first step when faced with a picture classification problem is to get as much information as possible through the pictures. Therefore, load and print the first nine pictures of dogs in a single figure. Repeat the same for cats. You can see that the pictures are in color and have different shapes and sizes.

This variety of sizes and formats must be sorted out before training the model. Make sure they all have a fixed size of 200x200 pixels.

As you can see, there are a lot of images. Make sure you stick to the following rules:

1. **If you have more than 12 gigabytes of RAM**, use the Keras image processing API to load the 25,000 photos into the training dataset and reshape them to 200×200 pixel square photos. The label must also be determined for each photo based on the file names. A tuple of photos and labels should be saved.
2. **If you have no more than 12 gigabytes of RAM**, load the images progressively using the Keras `ImageDataGenerator` class and the `flow_from_directory()` function. This will be slower to run but it will run on less capable hardware. This function prefers the data to be split into separate _train_ and _test_ directories, and under each directory to have a subdirectory for each class.

Once you have all the images processed, create an `ImageDataGenerator` object for training and test data. Then pass the folder that has training data to the `trdata` object and, similarly, pass the folder that has test data to the `tsdata` object. In this way, the images will be automatically labeled, and everything will be ready to enter the network.

#### Step 3: Build an ANN

Any classifier that fits this problem will have to be robust because some images show the cat or dog in a corner, or perhaps 2 cats or dogs in the same picture. If you have been able to research some of the winner implementations of other competitions also related to images, you will see that `EfficientNet-B0` is a CNN architecture introduced by Google in 2019. It uses compound scaling of depth, width, and resolution to reach strong ImageNet accuracy with far fewer parameters than older models, and it remains a solid, efficient baseline for vision tasks.

It uses the following test architecture:

```py
from keras.applications import EfficientNetB0
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D

model = Sequential()
model.add(EfficientNetB0(include_top = False, weights = None, input_shape = (224, 224, 3)))
model.add(GlobalAveragePooling2D())
model.add(Dense(units = 128, activation = "relu"))
model.add(Dense(units = 2, activation = "softmax"))
```

The above code loads the `EfficientNet-B0` convolutional backbone, pools its feature maps with `GlobalAveragePooling2D`, and then applies dense layers (`Dense` layers) for the final dog/cat classification.

Then add the remaining elements to form the model, train it and measure its performance.

#### Step 4: Optimize the above model

Import the `ModelCheckpoint` and `EarlyStopping` method from Keras. Create an object of both and pass them as callback functions to `fit_generator`.

Load the best model from the above and use the test set to make predictions.

#### Step 5: Save the model

Store the model in the corresponding folder.

> Note: We also incorporated the solution samples on [this link](https://github.com/4GeeksAcademy/image-classifier-project-tutorial/blob/main/solution.ipynb)that ) we strongly suggest you only use if you are stuck for more than 30 min or if you have already finished and want to compare it with your approach.
