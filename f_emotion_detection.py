import config as cfg
import cv2
import face_recognition
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

class predict_emotions():
    def __init__(self):
        self.model = load_model(cfg.path_model)

    def preprocess_img(self,face_image,rgb=True,w=48,h=48):
        face_image = cv2.resize(face_image, (w,h))
        if rgb == False:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = face_image.astype("float") / 255.0
        face_image= img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        return face_image

    def get_emotion(self,boxes_face,img):
        emotions = []
        for box in boxes_face:
            x0,y1,x1,y0 = box
            face_image = img[x0:x1,y0:y1]
            # preprocesar data
            face_image = self.preprocess_img(face_image ,cfg.rgb, cfg.w, cfg.h)
            # predecir imagen
            prediction = self.model.predict(face_image)
            emotion = cfg.labels[prediction.argmax()]
            emotions.append(emotion)
        return emotions


def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        y0,x1,y1,x0 = box[i]
        img = cv2.rectangle(img,
                    (x0,y0),
                    (x1,y1),
                    (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img
