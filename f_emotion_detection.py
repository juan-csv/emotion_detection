import config as cfg
import cv2
import face_recognition
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

class predict_emotions():
    def __init__(self):
        # cargo modelo de deteccion de emociones
        self.model = load_model(cfg.path_model)
        # cargo modelo de deteccion de rostros frontales
        self.detect_frontal_face = cv2.CascadeClassifier(cfg.detect_frontal_face)

    def preprocess_img(self,face_image,rgb=True,w=48,h=48):
        face_image = cv2.resize(face_image, (w,h))
        if rgb == False:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        face_image = face_image.astype("float") / 255.0
        face_image= img_to_array(face_image)
        face_image = np.expand_dims(face_image, axis=0)
        return face_image

    def detect_faces(self,img):
        rects,_,confidence = self.detect_frontal_face.detectMultiScale3(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE, outputRejectLevels = True)
        #rects = cascade.detectMultiScale(img,minNeighbors=10, scaleFactor=1.05)
        if len(rects) == 0:
            return [],[]
        rects[:,2:] += rects[:,:2]
        return rects,confidence

    def get_emotion(self,img):
        emotions = []
        # detectar_rostro
        boxes_face,_= self.detect_faces(img)
        if len(boxes_face)!=0:
            for box in boxes_face:
                y0,x0,y1,x1 = box
                face_image = img[x0:x1,y0:y1]
                # preprocesar data
                face_image = self.preprocess_img(face_image ,cfg.rgb, cfg.w, cfg.h)
                # predecir imagen
                prediction = self.model.predict(face_image)
                emotion = cfg.labels[prediction.argmax()]
                emotions.append(emotion)
        else:
            emotions = []
            boxes_face = []
        return emotions,boxes_face


def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        x0,y0,x1,y1 = box[i]
        img = cv2.rectangle(img,
                    (x0,y0),
                    (x1,y1),
                    (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img
