import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #cascade classifier
model = FacialExpressionModel("model.json", "model_weights.h5") #new model for predictions
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object): #class for the video
    def __init__(self):
        self.video = cv2.VideoCapture("/home/rhyme/Desktop/Project/videos/facial_exp.mkv") #change path to zero for source as input source

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self): #load video and check expression in each frame
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5) #gray scale

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48)) # predict new emotion
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2) #create expression text by opencv
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2) # rectangular where expression is using opencv

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
