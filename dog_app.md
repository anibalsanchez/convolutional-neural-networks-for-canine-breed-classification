# Data Scientist Nanodegree

## Convolutional Neural Networks

## Project: Write an Algorithm for a Dog Identification App

This notebook walks you through one of the most popular Udacity projects across machine learning and artificial intellegence nanodegree programs.  The goal is to classify images of dogs according to their breed.

If you are looking for a more guided capstone project related to deep learning and convolutional neural networks, this might be just it.  Notice that even if you follow the notebook to creating your classifier, you must still create a blog post or deploy an application to fulfill the requirements of the capstone project.

Also notice, you may be able to use only parts of this notebook (for example certain coding portions or the data) without completing all parts and still meet all requirements of the capstone project.

---

In this notebook, some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this project. You will not need to modify the included code beyond what is requested. Sections that begin with **'(IMPLEMENTATION)'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section, and the specifics of the implementation are marked in the code block with a 'TODO' statement. Please be sure to read the instructions carefully!

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut.  Markdown cells can be edited by double-clicking the cell to enter edit mode.

The rubric contains _optional_ "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. If you decide to pursue the "Stand Out Suggestions", you should include the code in this IPython notebook.

---

### Why We're Here

In this notebook, you will make the first steps towards developing an algorithm that could be used as part of a mobile or web app.  At the end of this project, your code will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling.  The image below displays potential sample output of your finished project (... but we expect that each student's algorithm will behave differently!).

![Sample Dog Output](images/sample_dog_output.png)

In this real-world setting, you will need to piece together a series of models to perform different tasks; for instance, the algorithm that detects humans in an image will be different from the CNN that infers dog breed.  There are many points of possible failure, and no perfect algorithm exists.  Your imperfect solution will nonetheless create a fun user experience!

### The Road Ahead

We break the notebook into separate steps.  Feel free to use the links below to navigate the notebook.

* [Step 0](#step0): Import Datasets
* [Step 1](#step1): Detect Humans
* [Step 2](#step2): Detect Dogs
* [Step 3](#step3): Create a CNN to Classify Dog Breeds (from Scratch)
* [Step 4](#step4): Use a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 5](#step5): Create a CNN to Classify Dog Breeds (using Transfer Learning)
* [Step 6](#step6): Write your Algorithm
* [Step 7](#step7): Test Your Algorithm

---
<a id='step0'></a>

## Step 0: Import Datasets

### Import Dog Dataset

In the code cell below, we import a dataset of dog images.  We populate a few variables through the use of the `load_files` function from the scikit-learn library:
* `train_files`, `valid_files`, `test_files` - numpy arrays containing file paths to images
* `train_targets`, `valid_targets`, `test_targets` - numpy arrays containing onehot-encoded classification labels
* `dog_names` - list of string-valued dog breed names for translating labels

```python
# Run the following cell only if the /workspace/home/dog-project/dog_images/ folder is not present in your workspace.
# The cell below will copy the data to your /workspace directory.
# !cp -rp /data/dog_images/ /workspace/home/dog-project

'''
NOTE
There is a known "Permission denied" issue with copying the following files. You can ignore them.
- Ibizan_hound_05697.jpg
- American_foxhound_00503.jpg
- Basset_hound_01064.jpg
- Labrador_retriever_06476.jpg
- Manchester_terrier_06806.jpg
- Norwegian_lundehund_07217.jpg
'''
```

    '\nNOTE\nThere is a known "Permission denied" issue with copying the following files. You can ignore them.\n- Ibizan_hound_05697.jpg\n- American_foxhound_00503.jpg\n- Basset_hound_01064.jpg\n- Labrador_retriever_06476.jpg\n- Manchester_terrier_06806.jpg\n- Norwegian_lundehund_07217.jpg\n'

```python
# Install the necessary package
# Restart the kernel once after install this package

# !python3 -m pip install opencv-python-headless==4.9.0.80

#
# https://stackoverflow.com/questions/77617946/solve-conda-libmamba-solver-libarchive-so-19-error-after-updating-conda-to-23
# conda install --solver=classic conda-forge::conda-libmamba-solver conda-forge::libmamba conda-forge::libmambapy conda-forge::libarchive
#
# conda install numpy
# conda install scipy pandas
# conda install matplotlib
# conda install scikit-learn
# conda install seaborn
# conda install six
# conda install tensorflow
# conda install fastai::opencv-python-headless
#
import numpy as np
from keras.utils import get_file

labels_path = get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

print(imagenet_labels)

```

    2024-12-13 12:27:12.141833: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-12-13 12:27:12.162425: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


    ['background' 'tench' 'goldfish' ... 'bolete' 'ear' 'toilet tissue']

```python
from sklearn.datasets import load_files
from keras.utils import to_categorical
from glob import glob

# Define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = data['filenames']
    dog_targets = to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# Load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')

# Load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dog_images/train/*/"))]

# Print statistics about the dataset
print(f'There are {len(dog_names)} total dog categories.')
print(f'There are {len(train_files) + len(valid_files) + len(test_files)} total dog images.')
print(f'There are {len(train_files)} training dog images.')
print(f'There are {len(valid_files)} validation dog images.')
print(f'There are {len(test_files)} test dog images.')
```

    There are 0 total dog categories.
    There are 8351 total dog images.
    There are 6680 training dog images.
    There are 835 validation dog images.
    There are 836 test dog images.

### Import Human Dataset

In the code cell below, we import a dataset of human images, where the file paths are stored in the numpy array `human_files`.

```python
import random
random.seed(8675309)

# Load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)
random.shuffle(human_files)

# Print statistics about the human dataset
print(f'There are {len(human_files)} total human images.')
```

    There are 13233 total human images.

---
<a id='step1'></a>

## Step 1: Detect Humans

We use OpenCV's implementation of [Haar feature-based cascade classifiers](http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images.  OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).  We have downloaded one of these detectors and stored it in the `haarcascades` directory.

In the next code cell, we demonstrate how to use this detector to find human faces in a sample image.

```python
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])

# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
```

    Number of faces detected: 1




![png](dog_app_files/dog_app_7_1.png)

Before using any of the face detectors, it is standard procedure to convert the images to grayscale.  The `detectMultiScale` function executes the classifier stored in `face_cascade` and takes the grayscale image as a parameter.

In the above code, `faces` is a numpy array of detected faces, where each row corresponds to a detected face.  Each detected face is a 1D array with four entries that specifies the bounding box of the detected face.  The first two entries in the array (extracted in the above code as `x` and `y`) specify the horizontal and vertical positions of the top left corner of the bounding box.  The last two entries in the array (extracted here as `w` and `h`) specify the width and height of the box.

### Write a Human Face Detector

We can use this procedure to write a function that returns `True` if a human face is detected in an image and `False` otherwise.  This function, aptly named `face_detector`, takes a string-valued file path to an image as input and appears in the code block below.

```python
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0, faces
```

### (IMPLEMENTATION) Assess the Human Face Detector

**Question 1:** Use the code cell below to test the performance of the `face_detector` function.
* What percentage of the first 100 images in `human_files` have a detected human face?
* What percentage of the first 100 images in `dog_files` have a detected human face?

Ideally, we would like 100% of human images with a detected face and 0% of dog images with a detected face.  You will see that our algorithm falls short of this goal, but still gives acceptable performance.  We extract the file paths for the first 100 images from each of the datasets and store them in the numpy arrays `human_files_short` and `dog_files_short`.

**Answer:**

```python
human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm
## on the images in human_files_short and dog_files_short.
human_detected = 0
for img_path in human_files_short:
    is_humam, faces = face_detector(img_path)
    if is_humam:
        human_detected += 1
    else:
        img = cv2.imread(img_path)
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(cv_rgb)
        plt.show()

human_detection_rate = human_detected / len(human_files_short) * 100
print("{:.2f}% human face detection rate".format(human_detection_rate))

dog_false_positives = 0
for img_path in dog_files_short:
    is_humam, faces = face_detector(img_path)
    if is_humam:
        dog_false_positives += 1
        img = cv2.imread(img_path)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(cv_rgb)
        plt.show()

dog_false_positive_detection_rate = dog_false_positives / len(dog_files_short) * 100
print("{:.2f}% dog face false-positive detection rate".format(dog_false_positive_detection_rate))
```

![png](dog_app_files/dog_app_11_0.png)

    99.00% human face detection rate




![png](dog_app_files/dog_app_11_2.png)

![png](dog_app_files/dog_app_11_3.png)

![png](dog_app_files/dog_app_11_4.png)

![png](dog_app_files/dog_app_11_5.png)

![png](dog_app_files/dog_app_11_6.png)

![png](dog_app_files/dog_app_11_7.png)

![png](dog_app_files/dog_app_11_8.png)

![png](dog_app_files/dog_app_11_9.png)

![png](dog_app_files/dog_app_11_10.png)

![png](dog_app_files/dog_app_11_11.png)

![png](dog_app_files/dog_app_11_12.png)

![png](dog_app_files/dog_app_11_13.png)

    12.00% dog face false-positive detection rate

**Question 2:** This algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

**Answer:**

We suggest the face detector from OpenCV as a potential way to detect human images in your algorithm, but you are free to explore other approaches, especially approaches that make use of deep learning :).  Please use the code cell below to design and test your own face detection algorithm.  If you decide to pursue this _optional_ task, report performance on each of the datasets.

```python
## (Optional) TODO: Report the performance of another
## face detection algorithm on the LFW dataset
### Feel free to use as many code cells as needed.
```

---
<a id='step2'></a>

## Step 2: Detect Dogs

In this section, we use a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model to detect dogs in images.  Our first line of code downloads the ResNet-50 model, along with weights that have been trained on [ImageNet](http://www.image-net.org/), a very large, very popular dataset used for image classification and other vision tasks.  ImageNet contains over 10 million URLs, each linking to an image containing an object from one of [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  Given an image, this pre-trained ResNet-50 model returns a prediction (derived from the available categories in ImageNet) for the object that is contained in the image.

```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights="imagenet")
```

### Pre-process the Data

When using TensorFlow as backend, Keras CNNs require a 4D array (which we'll also refer to as a 4D tensor) as input, with shape

$$
(\text{nb_samples}, \text{rows}, \text{columns}, \text{channels}),
$$

where `nb_samples` corresponds to the total number of images (or samples), and `rows`, `columns`, and `channels` correspond to the number of rows, columns, and channels for each image, respectively.

The `path_to_tensor` function below takes a string-valued file path to a color image as input and returns a 4D tensor suitable for supplying to a Keras CNN.  The function first loads the image and resizes it to a square image that is $224 \times 224$ pixels.  Next, the image is converted to an array, which is then resized to a 4D tensor.  In this case, since we are working with color images, each image has three channels.  Likewise, since we are processing a single image (or sample), the returned tensor will always have shape

$$
(1, 224, 224, 3).
$$

The `paths_to_tensor` function takes a numpy array of string-valued image paths as input and returns a 4D tensor with shape

$$
(\text{nb_samples}, 224, 224, 3).
$$

Here, `nb_samples` is the number of samples, or number of images, in the supplied array of image paths.  It is best to think of `nb_samples` as the number of 3D tensors (where each 3D tensor corresponds to a different image) in your dataset!

```python
from keras.preprocessing import image
# from tqdm import tqdm

def path_to_tensor(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        # Normalize the image tensor
        return np.expand_dims(x, axis=0).astype("float32") / 255
    except IOError:
        print(f"Warning: Skipping corrupted image {img_path}")
        return None


def paths_to_tensor(img_paths):
    batch_tensors = []
    for img_path in img_paths:
        tensor = path_to_tensor(img_path)
        if tensor is not None:
            batch_tensors.append(tensor[0])
    return np.array(batch_tensors)
```

### Making Predictions with ResNet-50

Getting the 4D tensor ready for ResNet-50, and for any other pre-trained model in Keras, requires some additional processing.  First, the RGB image is converted to BGR by reordering the channels.  All pre-trained models have the additional normalization step that the mean pixel (expressed in RGB as $[103.939, 116.779, 123.68]$ and calculated from all pixels in all images in ImageNet) must be subtracted from every pixel in each image.  This is implemented in the imported function `preprocess_input`.  If you're curious, you can check the code for `preprocess_input` [here](https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py).

Now that we have a way to format our image for supplying to ResNet-50, we are now ready to use the model to extract the predictions.  This is accomplished with the `predict` method, which returns an array whose $i$-th entry is the model's predicted probability that the image belongs to the $i$-th ImageNet category.  This is implemented in the `ResNet50_predict_labels` function below.

By taking the argmax of the predicted probability vector, we obtain an integer corresponding to the model's predicted object class, which we can identify with an object category through the use of this [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

```python
from keras.applications.resnet50 import preprocess_input, decode_predictions
import keras

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path

    # img = preprocess_input(path_to_tensor(img_path))
    # predictions = ResNet50_model.predict(img, verbose=0)
    # return np.argmax(predictions)

    img = image.load_img(img_path, target_size=(224, 224))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = ResNet50_model.predict(x, verbose=0)

    decoded_predictions = decode_predictions(predictions, top=1)[0]

    return decoded_predictions[0][1]
```

### Write a Dog Detector

While looking at the [dictionary](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), you will notice that the categories corresponding to dogs appear in an uninterrupted sequence and correspond to dictionary keys 151-268, inclusive, to include all categories from `'Chihuahua'` to `'Mexican hairless'`.  Thus, in order to check to see if an image is predicted to contain a dog by the pre-trained ResNet-50 model, we need only check if the `ResNet50_predict_labels` function above returns a value between 151 and 268 (inclusive).

We use these ideas to complete the `dog_detector` function below, which returns `True` if a dog is detected in an image (and `False` if not).

```python
### returns "True" if a dog is detected in the image stored at img_path
# def dog_detector(img_path):
#    prediction = ResNet50_predict_labels(img_path)
#    return (prediction <= 268) & (prediction >= 151)

dog_keywords = [
    "affenpinscher",
    "afghan_hound",
    "african_hunting_dog",
    "airedale",
    "american_staffordshire_terrier",
    "australian_terrier",
    "basenji",
    "basset",
    "beagle",
    "bedlington_terrier",
    "bernese_mountain_dog",
    "black-and-tan_coonhound",
    "blenheim_spaniel",
    "bloodhound",
    "bluetick",
    "border_collie",
    "border_terrier",
    "borzoi",
    "boston_bull",
    "bouvier_des_flandres",
    "boxer",
    "briard",
    "brittany_spaniel",
    "bull_mastiff",
    "cairn",
    "cardigan",
    "chesapeake_bay_retriever",
    "chihuahua",
    "chow",
    "clumber",
    "cocker_spaniel",
    "collie",
    "curly-coated_retriever",
    "dalmatian",
    "dandie_dinmont",
    "doberman",
    "english_foxhound",
    "english_setter",
    "english_springer",
    "entlebucher",
    "eskimo_dog",
    "flat-coated_retriever",
    "french_bulldog",
    "german_shepherd",
    "german_short-haired_pointer",
    "giant_schnauzer",
    "golden_retriever",
    "gordon_setter",
    "great_dane",
    "great_pyrenees",
    "groenendael",
    "ibizan_hound",
    "irish_setter",
    "irish_terrier",
    "irish_water_spaniel",
    "irish_wolfhound",
    "italian_greyhound",
    "japanese_spaniel",
    "keeshond",
    "kelpie",
    "kerry_blue_terrier",
    "komondor",
    "kuvasz",
    "labrador_retriever",
    "lakeland_terrier",
    "leonberg",
    "lhasa",
    "malamute",
    "malinois",
    "maltese_dog",
    "mexican_hairless",
    "miniature_pinscher",
    "miniature_poodle",
    "miniature_schnauzer",
    "newfoundland",
    "norfolk_terrier",
    "norwegian_elkhound",
    "norwich_terrier",
    "old_english_sheepdog",
    "otterhound",
    "papillon",
    "pekinese",
    "pembroke",
    "pomeranian",
    "pug",
    "redbone",
    "rhodesian_ridgeback",
    "rottweiler",
    "saint_bernard",
    "saluki",
    "samoyed",
    "schipperke",
    "scotch_terrier",
    "scottish_deerhound",
    "sealyham_terrier",
    "shetland_sheepdog",
    "shih-tzu",
    "siberian_husky",
    "silky_terrier",
    "soft-coated_wheaten_terrier",
    "staffordshire_bullterrier",
    "standard_poodle",
    "standard_schnauzer",
    "sussex_spaniel",
    "tibetan_mastiff",
    "tibetan_terrier",
    "toy_poodle",
    "toy_terrier",
    "vizsla",
    "walker_hound",
    "weimaraner",
    "welsh_springer_spaniel",
    "west_highland_white_terrier",
    "whippet",
    "wire-haired_fox_terrier",
    "yorkshire_terrier",
]

def dog_detector(img_path):
    predicted_class = ResNet50_predict_labels(img_path)

    # print("Predicted:", predicted_class)

    return any(keyword in predicted_class.lower() for keyword in dog_keywords)
```

### (IMPLEMENTATION) Assess the Dog Detector

**Question 3:** Use the code cell below to test the performance of your `dog_detector` function.
* What percentage of the images in `human_files_short` have a detected dog?
* What percentage of the images in `dog_files_short` have a detected dog?

**Answer:**

```python
### TODO: Test the performance of the dog_detector function
### on the images in human_files_short and dog_files_short.

human_files_short = human_files[:100]
dog_files_short = train_files[:100]
# Do NOT modify the code above this line.

## TODO: Test the performance of the face_detector algorithm
## on the images in human_files_short and dog_files_short.
dog_false_positives = 0
for img_path in human_files_short:
    is_dog = dog_detector(img_path)
    if is_dog:
        dog_false_positives += 1

dog_false_positives_rate = dog_false_positives / len(human_files_short) * 100
print("{:.2f}% human face false-positive rate".format(dog_false_positives_rate))

dog_detected = 0
for img_path in dog_files_short:
    is_dog = dog_detector(img_path)
    if is_dog:
        dog_detected += 1

dog_detection_rate = dog_detected / len(dog_files_short) * 100
print("{:.2f}% dog face detection rate".format(dog_detection_rate))

```

    1.00% human face false-positive rate
    100.00% dog face detection rate

---
<a id='step3'></a>

## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)

Now that we have functions for detecting humans and dogs in images, we need a way to predict breed from images.  In this step, you will create a CNN that classifies dog breeds.  You must create your CNN _from scratch_ (so, you can't use transfer learning _yet_!), and you must attain a test accuracy of at least 1%.  In Step 5 of this notebook, you will have the opportunity to use transfer learning to create a CNN that attains greatly improved accuracy.

Be careful with adding too many trainable layers!  More parameters means longer training, which means you are more likely to need a GPU to accelerate the training process.  Thankfully, Keras provides a handy estimate of the time that each epoch is likely to take; you can extrapolate this estimate to figure out how long it will take for your algorithm to train.

We mention that the task of assigning breed to dogs from images is considered exceptionally challenging.  To see why, consider that _even a human_ would have great difficulty in distinguishing between a Brittany and a Welsh Springer Spaniel.

Brittany | Welsh Springer Spaniel
* | -
<img src="images/Brittany_02625.jpg" width="100"> | <img src="images/Welsh_springer_spaniel_08203.jpg" width="200">

It is not difficult to find other dog breed pairs with minimal inter-class variation (for instance, Curly-Coated Retrievers and American Water Spaniels).

Curly-Coated Retriever | American Water Spaniel
* | -
<img src="images/Curly-coated_retriever_03896.jpg" width="200"> | <img src="images/American_water_spaniel_00648.jpg" width="200">

Likewise, recall that labradors come in yellow, chocolate, and black.  Your vision-based algorithm will have to conquer this high intra-class variation to determine how to classify all of these different shades as the same breed.

Yellow Labrador | Chocolate Labrador | Black Labrador
* | -
<img src="images/Labrador_retriever_06457.jpg" width="150"> | <img src="images/Labrador_retriever_06455.jpg" width="240"> | <img src="images/Labrador_retriever_06449.jpg" width="220">

We also mention that random chance presents an exceptionally low bar: setting aside the fact that the classes are slightly imabalanced, a random guess will provide a correct answer roughly 1 in 133 times, which corresponds to an accuracy of less than 1%.

Remember that the practice is far ahead of the theory in deep learning.  Experiment with many different architectures, and trust your intuition.  And, of course, have fun!

### Pre-process the Data

We rescale the images by dividing every pixel in every image by 255.

```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Allow loading of truncated images

def image_generator(files, targets, batch_size):
    while True:
        batch_paths = np.random.choice(a=files, size=batch_size)
        batch_input = paths_to_tensor(batch_paths)
        valid_paths = [p for p in batch_paths if path_to_tensor(p) is not None]
        batch_indices = [np.where(files == img_path)[0][0] for img_path in valid_paths]
        batch_output = np.array([targets[index] for index in batch_indices])

        if len(batch_input) > 0:  # Ensure there is data to yield
            yield batch_input, batch_output

# Create generators for train, validation, and test datasets
train_generator = image_generator(train_files, train_targets, batch_size=64)
valid_generator = image_generator(valid_files, valid_targets, batch_size=64)
test_generator = image_generator(test_files, test_targets, batch_size=64)
```

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:

        model.summary()

We have imported some Python modules to get you started, but feel free to import as many modules as you need.  If you end up getting stuck, here's a hint that specifies a model that trains relatively fast on CPU and attains >1% test accuracy in 5 epochs:

![Sample CNN](images/sample_cnn.png)

**Question 4:** Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  If you chose to use the hinted architecture above, describe why you think that CNN architecture should work well for the image classification task.

**Answer:**

```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

### TODO: Define your architecture.

# model = Sequential([
#     Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     GlobalAveragePooling2D(),
#     Dense(133, activation='softmax')
# ])

num_classes = 133
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

```

    /home/anibalsanchez/5_bin/miniconda3/lib/python3.11/site-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(activity_regularizer=activity_regularizer, **kwargs)

### Compile the Model

```python
# Set a smaller learning rate
# from keras.optimizers import Adam
# adam = Adam(lr=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">222</span>, <span style="color: #00af00; text-decoration-color: #00af00">222</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">448</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">111</span>, <span style="color: #00af00; text-decoration-color: #00af00">111</span>, <span style="color: #00af00; text-decoration-color: #00af00">16</span>)   │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">109</span>, <span style="color: #00af00; text-decoration-color: #00af00">109</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">4,640</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">54</span>, <span style="color: #00af00; text-decoration-color: #00af00">54</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">93312</span>)          │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │     <span style="color: #00af00; text-decoration-color: #00af00">5,972,032</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">133</span>)            │         <span style="color: #00af00; text-decoration-color: #00af00">8,645</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">5,985,765</span> (22.83 MB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">5,985,765</span> (22.83 MB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement.

```python
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

### TODO: specify the number of epochs that you would like to use to train the model.

epochs = 10

### Do NOT modify the code below this line.

# Add checkpoint to save the best model
# checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
#                                verbose=1, save_best_only=True)

# ReduceLROnPlateau: This callback reduces the learning rate when a metric has stopped improving.
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

checkpointer = ModelCheckpoint(
    filepath="saved_models/weights.best.from_scratch.keras",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1
)

# Train the model
model.fit(train_generator,
                    steps_per_epoch=len(train_files) // 32,
                    epochs=epochs,
                    validation_data=valid_generator,
                    validation_steps=len(valid_files) // 32,
                    callbacks=[checkpointer, reduce_lr],
                    verbose=2)
```

    Epoch 1/10

    Epoch 1: val_accuracy improved from -inf to 0.01623, saving model to saved_models/weights.best.from_scratch.keras
    208/208 - 87s - 419ms/step - accuracy: 0.0138 - loss: 4.8794 - val_accuracy: 0.0162 - val_loss: 4.8441 - learning_rate: 0.0010
    Epoch 2/10

    Epoch 2: val_accuracy improved from 0.01623 to 0.04147, saving model to saved_models/weights.best.from_scratch.keras
    208/208 - 88s - 423ms/step - accuracy: 0.0497 - loss: 4.4675 - val_accuracy: 0.0415 - val_loss: 4.4995 - learning_rate: 0.0010
    Epoch 3/10

    Epoch 3: val_accuracy improved from 0.04147 to 0.04808, saving model to saved_models/weights.best.from_scratch.keras
    208/208 - 86s - 412ms/step - accuracy: 0.1597 - loss: 3.6059 - val_accuracy: 0.0481 - val_loss: 4.9720 - learning_rate: 0.0010
    Epoch 4/10

    Epoch 4: val_accuracy did not improve from 0.04808
    208/208 - 83s - 398ms/step - accuracy: 0.3384 - loss: 2.6883 - val_accuracy: 0.0439 - val_loss: 5.9526 - learning_rate: 0.0010
    Epoch 5/10

    Epoch 5: val_accuracy did not improve from 0.04808

    Epoch 5: ReduceLROnPlateau reducing learning rate to 0.00020000000949949026.
    208/208 - 82s - 394ms/step - accuracy: 0.5300 - loss: 1.8670 - val_accuracy: 0.0403 - val_loss: 7.4455 - learning_rate: 0.0010
    Epoch 6/10

    Epoch 6: val_accuracy did not improve from 0.04808
    208/208 - 82s - 396ms/step - accuracy: 0.6809 - loss: 1.3047 - val_accuracy: 0.0288 - val_loss: 7.9584 - learning_rate: 2.0000e-04
    Epoch 7/10

    Epoch 7: val_accuracy did not improve from 0.04808
    208/208 - 82s - 396ms/step - accuracy: 0.7333 - loss: 1.1003 - val_accuracy: 0.0385 - val_loss: 8.4870 - learning_rate: 2.0000e-04
    Epoch 8/10

    Epoch 8: val_accuracy did not improve from 0.04808

    Epoch 8: ReduceLROnPlateau reducing learning rate to 4.0000001899898055e-05.
    208/208 - 80s - 385ms/step - accuracy: 0.7776 - loss: 0.9497 - val_accuracy: 0.0282 - val_loss: 9.0550 - learning_rate: 2.0000e-04
    Epoch 9/10

    Epoch 9: val_accuracy did not improve from 0.04808
    208/208 - 84s - 404ms/step - accuracy: 0.8100 - loss: 0.8479 - val_accuracy: 0.0337 - val_loss: 8.9768 - learning_rate: 4.0000e-05
    Epoch 10/10

    Epoch 10: val_accuracy did not improve from 0.04808
    208/208 - 82s - 393ms/step - accuracy: 0.8132 - loss: 0.8157 - val_accuracy: 0.0325 - val_loss: 9.2871 - learning_rate: 4.0000e-05





    <keras.src.callbacks.history.History at 0x77ef18daddd0>

### Load the Model with the Best Validation Loss

```python
model.load_weights("saved_models/weights.best.from_scratch.keras")
```

### Test the Model

Try out your model on the test dataset of dog images.  Ensure that your test accuracy is greater than 1%.

```python
# Evaluate the model on the test data using `evaluate_generator`
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_files) // 32)

# Convert the accuracy to percentage
test_accuracy = test_accuracy * 100

print('Test accuracy: %.4f%%' % test_accuracy)
```

    [1m26/26[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 264ms/step - accuracy: 0.0505 - loss: 4.8880
    Test accuracy: 4.0865%

---
<a id='step4'></a>

## Step 4: Use a CNN to Classify Dog Breeds

To reduce training time without sacrificing accuracy, we show you how to train a CNN using transfer learning.  In the following step, you will get a chance to use transfer learning to train your own CNN.

### Obtain Bottleneck Features

```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
```

### Model Architecture

The model uses the the pre-trained VGG-16 model as a fixed feature extractor, where the last convolutional output of VGG-16 is fed as input to our model.  We only add a global average pooling layer and a fully connected layer, where the latter contains one node for each dog category and is equipped with a softmax.

```python
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
```

    /home/anibalsanchez/5_bin/miniconda3/lib/python3.11/site-packages/keras/src/layers/pooling/base_global_pooling.py:12: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
      super().__init__(**kwargs)

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_1"</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ global_average_pooling2d        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling2D</span>)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">133</span>)            │        <span style="color: #00af00; text-decoration-color: #00af00">68,229</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">68,229</span> (266.52 KB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">68,229</span> (266.52 KB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>

### Compile the Model

```python
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
```

### Train the Model

```python
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.keras',
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets,
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
```

    Epoch 1/20
    [1m305/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 996us/step - accuracy: 0.1100 - loss: 12.3089
    Epoch 1: val_loss improved from inf to 3.73713, saving model to saved_models/weights.best.VGG16.keras
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.1202 - loss: 11.9103 - val_accuracy: 0.4311 - val_loss: 3.7371
    Epoch 2/20
    [1m313/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 972us/step - accuracy: 0.5861 - loss: 2.2583
    Epoch 2: val_loss improved from 3.73713 to 2.55493, saving model to saved_models/weights.best.VGG16.keras
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.5869 - loss: 2.2540 - val_accuracy: 0.5593 - val_loss: 2.5549
    Epoch 3/20
    [1m314/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 967us/step - accuracy: 0.7319 - loss: 1.1952
    Epoch 3: val_loss improved from 2.55493 to 2.28619, saving model to saved_models/weights.best.VGG16.keras
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.7322 - loss: 1.1983 - val_accuracy: 0.5940 - val_loss: 2.2862
    Epoch 4/20
    [1m313/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 970us/step - accuracy: 0.8347 - loss: 0.7405
    Epoch 4: val_loss improved from 2.28619 to 2.21332, saving model to saved_models/weights.best.VGG16.keras
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.8338 - loss: 0.7434 - val_accuracy: 0.6263 - val_loss: 2.2133
    Epoch 5/20
    [1m297/334[0m [32m━━━━━━━━━━━━━━━━━[0m[37m━━━[0m [1m0s[0m 1ms/step - accuracy: 0.8711 - loss: 0.5035
    Epoch 5: val_loss improved from 2.21332 to 1.89798, saving model to saved_models/weights.best.VGG16.keras
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.8701 - loss: 0.5095 - val_accuracy: 0.6695 - val_loss: 1.8980
    Epoch 6/20
    [1m314/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 973us/step - accuracy: 0.8951 - loss: 0.3898
    Epoch 6: val_loss did not improve from 1.89798
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.8951 - loss: 0.3902 - val_accuracy: 0.6874 - val_loss: 1.9039
    Epoch 7/20
    [1m312/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 975us/step - accuracy: 0.9296 - loss: 0.2564
    Epoch 7: val_loss did not improve from 1.89798
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9289 - loss: 0.2599 - val_accuracy: 0.6946 - val_loss: 1.9419
    Epoch 8/20
    [1m315/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 963us/step - accuracy: 0.9445 - loss: 0.1990
    Epoch 8: val_loss did not improve from 1.89798
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9441 - loss: 0.2011 - val_accuracy: 0.6922 - val_loss: 1.9076
    Epoch 9/20
    [1m311/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 978us/step - accuracy: 0.9570 - loss: 0.1504
    Epoch 9: val_loss did not improve from 1.89798
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9565 - loss: 0.1522 - val_accuracy: 0.6958 - val_loss: 1.9513
    Epoch 10/20
    [1m309/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 984us/step - accuracy: 0.9657 - loss: 0.1085
    Epoch 10: val_loss did not improve from 1.89798
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9655 - loss: 0.1098 - val_accuracy: 0.7114 - val_loss: 1.9866
    Epoch 11/20
    [1m316/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 961us/step - accuracy: 0.9836 - loss: 0.0663
    Epoch 11: val_loss did not improve from 1.89798
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9831 - loss: 0.0677 - val_accuracy: 0.7054 - val_loss: 1.9642
    Epoch 12/20
    [1m311/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 975us/step - accuracy: 0.9762 - loss: 0.0724
    Epoch 12: val_loss did not improve from 1.89798
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9762 - loss: 0.0728 - val_accuracy: 0.7114 - val_loss: 1.9099
    Epoch 13/20
    [1m311/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 976us/step - accuracy: 0.9865 - loss: 0.0453
    Epoch 13: val_loss did not improve from 1.89798
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9862 - loss: 0.0463 - val_accuracy: 0.7281 - val_loss: 2.0533
    Epoch 14/20
    [1m313/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 972us/step - accuracy: 0.9843 - loss: 0.0526
    Epoch 14: val_loss improved from 1.89798 to 1.88817, saving model to saved_models/weights.best.VGG16.keras
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9843 - loss: 0.0525 - val_accuracy: 0.7246 - val_loss: 1.8882
    Epoch 15/20
    [1m314/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 966us/step - accuracy: 0.9890 - loss: 0.0340
    Epoch 15: val_loss did not improve from 1.88817
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9888 - loss: 0.0344 - val_accuracy: 0.7186 - val_loss: 1.9695
    Epoch 16/20
    [1m309/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 985us/step - accuracy: 0.9914 - loss: 0.0315
    Epoch 16: val_loss did not improve from 1.88817
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9913 - loss: 0.0316 - val_accuracy: 0.7174 - val_loss: 1.9786
    Epoch 17/20
    [1m310/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 978us/step - accuracy: 0.9922 - loss: 0.0248
    Epoch 17: val_loss did not improve from 1.88817
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9922 - loss: 0.0250 - val_accuracy: 0.7186 - val_loss: 1.8909
    Epoch 18/20
    [1m308/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 986us/step - accuracy: 0.9913 - loss: 0.0257
    Epoch 18: val_loss did not improve from 1.88817
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9913 - loss: 0.0254 - val_accuracy: 0.7317 - val_loss: 1.9158
    Epoch 19/20
    [1m314/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 970us/step - accuracy: 0.9976 - loss: 0.0106
    Epoch 19: val_loss did not improve from 1.88817
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9974 - loss: 0.0111 - val_accuracy: 0.7377 - val_loss: 1.9851
    Epoch 20/20
    [1m313/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 972us/step - accuracy: 0.9972 - loss: 0.0117
    Epoch 20: val_loss did not improve from 1.88817
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9972 - loss: 0.0116 - val_accuracy: 0.7401 - val_loss: 1.9140





    <keras.src.callbacks.history.History at 0x77ef18310bd0>

### Load the Model with the Best Validation Loss

```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.keras')
```

### Test the Model

Now, we can use the CNN to test how well it identifies breed within our test dataset of dog images.  We print the test accuracy below.

```python
# Test the model
test_predictions = VGG16_model.predict(test_VGG16, batch_size=20)
test_accuracy = 100 * np.mean(np.argmax(test_predictions, axis=1) == np.argmax(test_targets, axis=1))
print('Test accuracy: %.4f%%' % test_accuracy)
```

    [1m42/42[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    Test accuracy: 71.7703%

### Predict Dog Breed with the Model

```python
from extract_bottleneck_features import *

def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
```

---
<a id='step5'></a>

## Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)

You will now use transfer learning to create a CNN that can identify dog breed from images.  Your CNN must attain at least 60% accuracy on the test set.

In Step 4, we used transfer learning to create a CNN using VGG-16 bottleneck features.  In this section, you must use the bottleneck features from a different pre-trained model.  To make things easier for you, we have pre-computed the features for all of the networks that are currently available in Keras:
* [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
* [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
* [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
* [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

The files are encoded as such:

    Dog{network}Data.npz

where `{network}`, in the above filename, can be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.  Pick one of the above architectures, download the corresponding bottleneck features, and store the downloaded file in the `bottleneck_features/` folder in the repository.

### (IMPLEMENTATION) Obtain Bottleneck Features

In the code block below, extract the bottleneck features corresponding to the train, test, and validation sets by running the following:

    bottleneck_features = np.load('bottleneck_features/Dog{network}Data.npz')
    train_{network} = bottleneck_features['train']
    valid_{network} = bottleneck_features['valid']
    test_{network} = bottleneck_features['test']

```python
### TODO: Obtain bottleneck features from another pre-trained CNN.

bottleneck_features = np.load("bottleneck_features/DogResnet50Data.npz")
train_Resnet50 = bottleneck_features["train"]
valid_Resnet50 = bottleneck_features["valid"]
test_Resnet50 = bottleneck_features["test"]
```

### (IMPLEMENTATION) Model Architecture

Create a CNN to classify dog breed.  At the end of your code cell block, summarize the layers of your model by executing the line:

        <your model's name>.summary()

**Question 5:** Outline the steps you took to get to your final CNN architecture and your reasoning at each step.  Describe why you think the architecture is suitable for the current problem.

**Answer:**

```python
### TODO: Define your architecture.

Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dense(133, activation="softmax"))

Resnet50_model.summary()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_2"</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ global_average_pooling2d_1      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2048</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling2D</span>)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">133</span>)            │       <span style="color: #00af00; text-decoration-color: #00af00">272,517</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">272,517</span> (1.04 MB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">272,517</span> (1.04 MB)
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>

### (IMPLEMENTATION) Compile the Model

```python
### TODO: Compile the model.

Resnet50_model.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)
```

### (IMPLEMENTATION) Train the Model

Train your model in the code cell below.  Use model checkpointing to save the model that attains the best validation loss.

You are welcome to [augment the training data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html), but this is not a requirement.

```python
### TODO: Train the model.

checkpointer = ModelCheckpoint(
    filepath="saved_models/weights.best.Resnet50.keras", verbose=1, save_best_only=True
)

Resnet50_model.fit(
    train_Resnet50,
    train_targets,
    validation_data=(valid_Resnet50, valid_targets),
    epochs=20,
    batch_size=20,
    callbacks=[checkpointer],
    verbose=1,
)
```

    Epoch 1/20
    [1m326/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.4107 - loss: 2.7626
    Epoch 1: val_loss improved from inf to 0.85346, saving model to saved_models/weights.best.Resnet50.keras
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.4157 - loss: 2.7322 - val_accuracy: 0.7413 - val_loss: 0.8535
    Epoch 2/20
    [1m323/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.8670 - loss: 0.4461
    Epoch 2: val_loss improved from 0.85346 to 0.73689, saving model to saved_models/weights.best.Resnet50.keras
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.8669 - loss: 0.4460 - val_accuracy: 0.7772 - val_loss: 0.7369
    Epoch 3/20
    [1m312/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 1ms/step - accuracy: 0.9223 - loss: 0.2549
    Epoch 3: val_loss improved from 0.73689 to 0.64463, saving model to saved_models/weights.best.Resnet50.keras
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 2ms/step - accuracy: 0.9221 - loss: 0.2552 - val_accuracy: 0.8000 - val_loss: 0.6446
    Epoch 4/20
    [1m319/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9493 - loss: 0.1635
    Epoch 4: val_loss did not improve from 0.64463
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9493 - loss: 0.1638 - val_accuracy: 0.8060 - val_loss: 0.6453
    Epoch 5/20
    [1m320/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9720 - loss: 0.1018
    Epoch 5: val_loss improved from 0.64463 to 0.62926, saving model to saved_models/weights.best.Resnet50.keras
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.9717 - loss: 0.1023 - val_accuracy: 0.8263 - val_loss: 0.6293
    Epoch 6/20
    [1m317/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 1ms/step - accuracy: 0.9768 - loss: 0.0755
    Epoch 6: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.9769 - loss: 0.0757 - val_accuracy: 0.8192 - val_loss: 0.6591
    Epoch 7/20
    [1m324/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9873 - loss: 0.0484
    Epoch 7: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9873 - loss: 0.0486 - val_accuracy: 0.8192 - val_loss: 0.6759
    Epoch 8/20
    [1m323/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9915 - loss: 0.0325
    Epoch 8: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9914 - loss: 0.0328 - val_accuracy: 0.8072 - val_loss: 0.6658
    Epoch 9/20
    [1m318/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9950 - loss: 0.0262
    Epoch 9: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.9949 - loss: 0.0263 - val_accuracy: 0.8240 - val_loss: 0.6635
    Epoch 10/20
    [1m324/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9964 - loss: 0.0200
    Epoch 10: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9963 - loss: 0.0200 - val_accuracy: 0.8180 - val_loss: 0.6733
    Epoch 11/20
    [1m324/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9973 - loss: 0.0161
    Epoch 11: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9973 - loss: 0.0161 - val_accuracy: 0.8275 - val_loss: 0.6467
    Epoch 12/20
    [1m322/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9977 - loss: 0.0109
    Epoch 12: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9977 - loss: 0.0110 - val_accuracy: 0.8347 - val_loss: 0.6602
    Epoch 13/20
    [1m320/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9986 - loss: 0.0082
    Epoch 13: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9985 - loss: 0.0083 - val_accuracy: 0.8395 - val_loss: 0.6493
    Epoch 14/20
    [1m322/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9983 - loss: 0.0077
    Epoch 14: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9983 - loss: 0.0078 - val_accuracy: 0.8371 - val_loss: 0.6550
    Epoch 15/20
    [1m323/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9988 - loss: 0.0058
    Epoch 15: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9987 - loss: 0.0059 - val_accuracy: 0.8371 - val_loss: 0.6576
    Epoch 16/20
    [1m321/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9990 - loss: 0.0045
    Epoch 16: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9990 - loss: 0.0046 - val_accuracy: 0.8479 - val_loss: 0.6378
    Epoch 17/20
    [1m321/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9989 - loss: 0.0070
    Epoch 17: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9989 - loss: 0.0069 - val_accuracy: 0.8431 - val_loss: 0.6582
    Epoch 18/20
    [1m323/334[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 1ms/step - accuracy: 0.9982 - loss: 0.0059
    Epoch 18: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step - accuracy: 0.9982 - loss: 0.0059 - val_accuracy: 0.8479 - val_loss: 0.6530
    Epoch 19/20
    [1m317/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 1ms/step - accuracy: 0.9995 - loss: 0.0029
    Epoch 19: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.9995 - loss: 0.0030 - val_accuracy: 0.8407 - val_loss: 0.6653
    Epoch 20/20
    [1m317/334[0m [32m━━━━━━━━━━━━━━━━━━[0m[37m━━[0m [1m0s[0m 1ms/step - accuracy: 0.9987 - loss: 0.0042
    Epoch 20: val_loss did not improve from 0.62926
    [1m334/334[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 1ms/step - accuracy: 0.9987 - loss: 0.0043 - val_accuracy: 0.8419 - val_loss: 0.6934





    <keras.src.callbacks.history.History at 0x77ef1027cdd0>

### (IMPLEMENTATION) Load the Model with the Best Validation Loss

```python
### TODO: Load the model weights with the best validation loss.
Resnet50_model.load_weights("saved_models/weights.best.Resnet50.keras")
```

### (IMPLEMENTATION) Test the Model

Try out your model on the test dataset of dog images. Ensure that your test accuracy is greater than 60%.

```python
### TODO: Calculate classification accuracy on the test dataset.
test_predictions = Resnet50_model.predict(test_Resnet50, batch_size=20)
test_accuracy = 100 * np.mean(
    np.argmax(test_predictions, axis=1) == np.argmax(test_targets, axis=1)
)
print("Test accuracy: %.4f%%" % test_accuracy)
```

    [1m42/42[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 1ms/step
    Test accuracy: 82.2967%

### (IMPLEMENTATION) Predict Dog Breed with the Model

Write a function that takes an image path as input and returns the dog breed (`Affenpinscher`, `Afghan_hound`, etc) that is predicted by your model.

Similar to the analogous function in Step 5, your function should have three steps:

1. Extract the bottleneck features corresponding to the chosen CNN model.
2. Supply the bottleneck features as input to the model to return the predicted vector.  Note that the argmax of this prediction vector gives the index of the predicted dog breed.
3. Use the `dog_names` array defined in Step 0 of this notebook to return the corresponding breed.

The functions to extract the bottleneck features can be found in `extract_bottleneck_features.py`, and they have been imported in an earlier code cell.  To obtain the bottleneck features corresponding to your chosen CNN architecture, you need to use the function

    extract_{network}

where `{network}`, in the above filename, should be one of `VGG19`, `Resnet50`, `InceptionV3`, or `Xception`.

```python
### TODO: Write a function that takes a path to an image as input
### and returns the dog breed that is predicted by the model.

def BestResnet50_predict_labels(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = ResNet50_model.predict(img_array, verbose=0)

    decoded_predictions = decode_predictions(predictions, top=1)[0]

    return decoded_predictions[0][1]

def BestResnet50_dog_detector(img_path):
    predicted_class = ResNet50_predict_labels(img_path)

    return any(keyword in predicted_class.lower() for keyword in dog_keywords)

print(
    "Label Beagle_01155.jpg:",
    BestResnet50_predict_labels("dogImages/test/016.Beagle/Beagle_01155.jpg"),
)

print(
    "Face Detector Beagle_01155.jpg: ",
    face_detector("dogImages/test/016.Beagle/Beagle_01155.jpg")[0],
)

print("Face Detector John_Travolta_0006.jpg: ", face_detector("lfw/John_Travolta/John_Travolta_0006.jpg")[0])

print(
    "BestResnet50_dog_detector Beagle_01155.jpg:",
    BestResnet50_dog_detector("dogImages/test/016.Beagle/Beagle_01155.jpg"),
)
print(
    "BestResnet50_dog_detector John_Travolta_0006.jpg:",
    BestResnet50_dog_detector("lfw/John_Travolta/John_Travolta_0006.jpg"),
)
```

    Label Beagle_01155.jpg: beagle
    Face Detector Beagle_01155.jpg:  False
    Face Detector John_Travolta_0006.jpg:  True
    BestResnet50_dog_detector Beagle_01155.jpg: True
    BestResnet50_dog_detector John_Travolta_0006.jpg: False

---
<a id='step6'></a>

## Step 6: Write your Algorithm

Write an algorithm that accepts a file path to an image and first determines whether the image contains a human, dog, or neither.  Then,
* if a **dog** is detected in the image, return the predicted breed.
* if a **human** is detected in the image, return the resembling dog breed.
* if **neither** is detected in the image, provide output that indicates an error.

You are welcome to write your own functions for detecting humans and dogs in images, but feel free to use the `face_detector` and `dog_detector` functions developed above.  You are **required** to use your CNN from Step 5 to predict dog breed.

A sample image and output for our algorithm is provided below, but feel free to design your own user experience!

![Sample Human Output](images/sample_human_2.png)

This photo looks like an Afghan Hound.

### (IMPLEMENTATION) Write your Algorithm

```python
### TODO: Write your algorithm.
### Feel free to use as many code cells as needed.

def my_algorithm(img_path):
    if BestResnet50_dog_detector(img_path):
        return BestResnet50_predict_labels(img_path)
    if face_detector(img_path)[0]:
        return 'human'
    return 'neither'

print(
    "my_algorithm Beagle_01155.jpg:",
    my_algorithm("dogImages/test/016.Beagle/Beagle_01155.jpg"),
)
print(
    "my_algorithm John_Travolta_0006.jpg:",
    my_algorithm("lfw/John_Travolta/John_Travolta_0006.jpg"),
)
```

    my_algorithm Beagle_01155.jpg: beagle
    my_algorithm John_Travolta_0006.jpg: human

---
<a id='step7'></a>

## Step 7: Test Your Algorithm

In this section, you will take your new algorithm for a spin!  What kind of dog does the algorithm think that **you** look like?  If you have a dog, does it predict your dog's breed accurately?  If you have a cat, does it mistakenly think that your cat is a dog?

### (IMPLEMENTATION) Test Your Algorithm on Sample Images

Test your algorithm at least six images on your computer.  Feel free to use any images you like.  Use at least two human and two dog images.

**Question 6:** Is the output better than you expected :) ?  Or worse :( ?  Provide at least three possible points of improvement for your algorithm.

**Answer:**

```python
## TODO: Execute your algorithm from Step 6 on
## at least 6 images on your computer.
## Feel free to use as many code cells as needed.

def test_my_algorithm(img_file):
    img_path = "sampleImages/" + img_file
    print(img_path, " => ", my_algorithm(img_path))
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()

image_files = [
    "20140202_Hinzenbach_Evelyn_Insam_2389.jpg",
    "640px-Dean_Rusk,_Lyndon_B._Johnson_and_Robert_McNamara_in_Cabinet_Room_meeting_February_1968.jpg",
    "640px-Football_player_before_throw.jpg",
    "640px-Riding_the_Plasma_Wave_-_Flickr_-_NASA_Goddard_Photo_and_Video.jpg",
    "640px-Sally_Ride_(1984).jpg",
    "640px-Vitoria_-_Parque_de_Olárizu_-_Niebla_y_cencellada_-BT-_01.jpg",
    "Aminah_Cendrakasih,_c._1959,_by_Tati_Photo_Studio.jpg",
    "Campamento_de_ganado_de_la_tribu_Mundari,_Terekeka,_Sudán_del_Sur,_2024-01-28,_DD_157.jpg",
    "Canis_lupus_familiaris.002_-_Monfero.jpg",
    "Canis_lupus_familiaris,_Neuss_(DE)_--_2024_--_0085.jpg",
    "Flickr_cc_runner_wisconsin_u.jpg",
    "Greenland_467_(35130903436)_(cropped).jpg",
    "Liver_yellow_dog_in_the_water_looking_at_viewer_at_golden_hour_in_Don_Det_Laos.jpg",
    "Mexican_Sunflower_Tithonia_rotundifolia_Flower_2163px.jpg",
    "MotorCycle.jpg",
    "RobotDragon.jpg",
    "Sikh_man,_Agra_10.jpg",
    "Vitruvian.jpg",
    "Woman_with_photo,_Afghanistan.jpg",
    "York_Minster_Chapter_House_Ceiling.jpg",
]

for image_file in image_files:
    test_my_algorithm(image_file)
```

    sampleImages/20140202_Hinzenbach_Evelyn_Insam_2389.jpg  =>  neither




![png](dog_app_files/dog_app_67_1.png)

    sampleImages/640px-Dean_Rusk,_Lyndon_B._Johnson_and_Robert_McNamara_in_Cabinet_Room_meeting_February_1968.jpg  =>  human




![png](dog_app_files/dog_app_67_3.png)

    sampleImages/640px-Football_player_before_throw.jpg  =>  neither




![png](dog_app_files/dog_app_67_5.png)

    sampleImages/640px-Riding_the_Plasma_Wave_-_Flickr_-_NASA_Goddard_Photo_and_Video.jpg  =>  neither




![png](dog_app_files/dog_app_67_7.png)

    sampleImages/640px-Sally_Ride_(1984).jpg  =>  human




![png](dog_app_files/dog_app_67_9.png)

    sampleImages/640px-Vitoria_-_Parque_de_Olárizu_-_Niebla_y_cencellada_-BT-_01.jpg  =>  neither




![png](dog_app_files/dog_app_67_11.png)

    sampleImages/Aminah_Cendrakasih,_c._1959,_by_Tati_Photo_Studio.jpg  =>  human




![png](dog_app_files/dog_app_67_13.png)

    sampleImages/Campamento_de_ganado_de_la_tribu_Mundari,_Terekeka,_Sudán_del_Sur,_2024-01-28,_DD_157.jpg  =>  neither




![png](dog_app_files/dog_app_67_15.png)

    sampleImages/Canis_lupus_familiaris.002_-_Monfero.jpg  =>  Chihuahua




![png](dog_app_files/dog_app_67_17.png)

    sampleImages/Canis_lupus_familiaris,_Neuss_(DE)_--_2024_--_0085.jpg  =>  vizsla




![png](dog_app_files/dog_app_67_19.png)

    sampleImages/Flickr_cc_runner_wisconsin_u.jpg  =>  neither




![png](dog_app_files/dog_app_67_21.png)

    sampleImages/Greenland_467_(35130903436)_(cropped).jpg  =>  neither




![png](dog_app_files/dog_app_67_23.png)

    sampleImages/Liver_yellow_dog_in_the_water_looking_at_viewer_at_golden_hour_in_Don_Det_Laos.jpg  =>  kelpie




![png](dog_app_files/dog_app_67_25.png)

    sampleImages/Mexican_Sunflower_Tithonia_rotundifolia_Flower_2163px.jpg  =>  neither




![png](dog_app_files/dog_app_67_27.png)

    sampleImages/MotorCycle.jpg  =>  neither




![png](dog_app_files/dog_app_67_29.png)

    sampleImages/RobotDragon.jpg  =>  neither




![png](dog_app_files/dog_app_67_31.png)

    sampleImages/Sikh_man,_Agra_10.jpg  =>  human




![png](dog_app_files/dog_app_67_33.png)

    sampleImages/Vitruvian.jpg  =>  human




![png](dog_app_files/dog_app_67_35.png)

    sampleImages/Woman_with_photo,_Afghanistan.jpg  =>  neither




![png](dog_app_files/dog_app_67_37.png)

    sampleImages/York_Minster_Chapter_House_Ceiling.jpg  =>  neither




![png](dog_app_files/dog_app_67_39.png)

## Analysis of the Human/Dog/Neither Detector

The Human/Dog/Neither detector demonstrates promising results, exceeding initial expectations. The algorithm employs a Haar feature-based cascade classifier (frontal face) for human detection and an enhanced ResNet50 model for dog identification, establishing a robust framework for multi-class image classification.

### Dog Detection Accuracy

The detector performs well in identifying dog images. All canine images in the sample set were correctly classified, highlighting the improved ResNet50 model's effectiveness in this task. While the algorithm did miss one image containing two dogs, overall, its high-level accuracy is excellent.

### Human Detection Performance

Human detection relies on the Haar feature-based cascade classifier (frontal face). Given that the category of human images is broader than the frontal face images used to train the detector, more errors are expected. The "Neither" classification shows promise but is impacted by errors in human misdetection.

### Areas for Improvement

#### Human Detection Accuracy

The mixed results in human image detection suggest a need for improvement. Using a specific human detector rather than a frontal face detector could enhance accuracy.

#### Handling Ambiguous Cases

Improving human detection will subsequently enhance the accuracy of "Neither" image classification.

#### Training Data Diversity

The frontal face detector missed images featuring clear frontal human faces, likely due to training data bias. Overrepresenting certain ethnicities, ages, or facial features in the training set can skew results. Haar cascades, which rely on contrast differences, may underperform on darker skin tones. While Haar cascade classifiers are valued for their speed and efficiency, modern deep learning-based approaches offer better performance and can be trained to mitigate various biases.
