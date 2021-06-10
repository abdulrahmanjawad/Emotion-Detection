# Emotion Detection
An emotion detection CNN model created using TensorFlow that can classify 7 emotions from facial expression images.

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
 1. Use `cd` command to move into the `detection` directory
 2. To detect emotion, use the following command:
    ```bash
    python detector.py path/to/image.png
    ```
<br>
The images folder contains sample images to try on the detector.
<br>

# Dataset
Facial Emotion Recognition [fer2013](https://www.kaggle.com/msambare/fer2013) was used to train the model.
The data consists of 48x48 pixel grayscale images of faces and has classified 7 emotions (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
    
# Tools used:
1. [Python](https://www.python.org/downloads/) 
2. [TensorFlow](https://www.tensorflow.org/)
3. [OpenCV](https://opencv.org/)
4. [Numpy](https://numpy.org/)
5. [Argparse](https://docs.python.org/3/library/argparse.html)
    
