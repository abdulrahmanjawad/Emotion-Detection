# Emotion Detection
An emotion detection CNN model created using TensorFlow that can classify 7 emotions from facial expressions.

# Getting Started
Make sure [Python 3](https://www.python.org/downloads/) is already installed.

## Setting up the environment
 1. Clone or download the repository on your local machine.
 2. Within the Facial-Emotion-Recognition directory create a Virtual Python Environment with command:
      ```bash
      python -m venv emotion
      ```
    where `emotion` is the name of the environment.
 4. Activate the enviroment using the command:
      ```bash
      emotion\scripts\activate
      ```
 4. Install the required packages using:
      ```bash
      pip install -r requirements.txt
      ```
      
## Using the the detector on images
 1. Use `cd` command to move into the `detection` directory.
 2. To detect emotion, use the following command:
    ```bash
    python detector.py path/to/image
    ```
<br>
The images folder contains sample images to try on the detector.
<br>

# Model
Model was trained on [Google Colab](https://colab.research.google.com/) to make use of GPU for computation.
## Data Augmentation
 - To compensate for the small size of dataset, we made use of data augmentation techniques.
 - They include: flip horizontal, rescaling (1./255), rotation etc.
 - Was applied only on training set.

## Architecture 
 - 4 layered Convolutional Network with Batch Normalization and Max Pooling.
 - 'Relu' activation and dropout rate of 0.2 used in Conv Nets
 - After Conv Nets, flattening is done to create Fully Connected Layers.
 -  FC consists of 2 dense layers. Hidden layer has 1024 neurons with 'Relu' activation. After hidden layer, batch normalization is done.
 - Final output layer has 7 neurons based on the 7 emotions and uses 'Softmax' activation.

## Hyperparameters
 - Batch Size: 128
 - Epochs: 50
 - SGD optimizer
 - Learning rate: 0.01
 - Learning rate decay: 0.0001

## Accuracy
 Final validation accuracy achieved was 62.72%. Test set was evaluated and accuracy achieved was 64%. There is some chance of overfitting that can be improved.

# Detector
- Uses `Haar Cascade Classifiers` in OpenCV to detect faces.
- Preprocesses the detected face and resizes the face into 48x48 sized image
- Passes the resized image into the model for emotion prediction
- Predicted emotion is then displayed along with the bounding boxes on the face.

# Dataset
Facial Emotion Recognition [fer2013](https://www.kaggle.com/msambare/fer2013) was used to train the model.
The data consists of 48x48 pixel grayscale images of faces and has classified 7 emotions (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
    
# Tools used:
1. [Python](https://www.python.org/downloads/) 
2. [TensorFlow](https://www.tensorflow.org/)
3. [OpenCV](https://opencv.org/)
4. [Numpy](https://numpy.org/)
5. [Argparse](https://docs.python.org/3/library/argparse.html)
