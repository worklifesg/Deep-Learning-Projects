from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf

#resource allocation (memeory allocation) on GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15 # bit

session=tf.compat.v1.Session(config=config)

class FacialExpressionModel(object):
    #list of emotions
    EMOTIONS_LIST=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

    def __init__(self,model_json_file,model_weights_file):
        with open('model_from_json', "r") as json_file:
            loaded_model_json=json_file.read()
            self.loaded_model=model_from_json(loaded_model_json)

        #loading weights in new models
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_fucntion()

    def predict_emotion(self,img):
        self.preds=self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
