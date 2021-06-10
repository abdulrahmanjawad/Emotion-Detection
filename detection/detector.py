import cv2
import numpy as np
import argparse
import tensorflow as tf 
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

def load_model():
	with open("../models/model.json", 'r') as f:
		model_json = f.read()

	model = model_from_json(model_json)
	model.load_weights('..\\models\\model_weights.h5')
	
	return model

def load_classifier():
	classifier = cv2.CascadeClassifier('..\\models\\haarcascade_frontalface_default.xml')

	return classifier

def _byte_to_img(contents):
	nparr = np.fromstring(contents, np.uint8)
	img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

	return img

def emotion_predictor(contents, args, cmd = False):

	if cmd == False:
		img = _byte_to_img(contents)
	else:
		img = cv2.imread(args['image'])
	
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	model = load_model()
	classifier = load_classifier()

	faces_detected = classifier.detectMultiScale(gray_img, 1.18, 5)

	for (x, y, w, h) in faces_detected:
	    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
	    roi_gray = gray_img[y:y + w, x:x + h]
	    roi_gray = cv2.resize(roi_gray, (48, 48))
	    img_pixels = image.img_to_array(roi_gray)
	    img_pixels = np.expand_dims(img_pixels, axis=0)
	    img_pixels /= 255.0

	    predictions = model.predict(img_pixels)
	    max_index = int(np.argmax(predictions))

	    emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral","Sad","Surprised"]
	    predicted_emotion = emotions[max_index]

	    # return predicted_emotion

	    cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

	resized_img = cv2.resize(img, (600, 400))
	cv2.imshow('Facial Emotion Recognition', resized_img)
	cv2.waitKey(0)

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument('image', help='path to input image file')
	args = vars(ap.parse_args())
	emotion_predictor(0, args, True)