'''
python Emotion_detection.py --input webcam
python Emotion_detection.py --input image --path data_test/friends1.jpg
'''
import f_emotion_detection as fed
import cv2
from imutils.video import VideoStream
import imutils
import argparse

# instanciar detector
Detector = fed.predict_emotions()
type_input = 'image'

if type_input == 'image':
    # ----------------------------- image -----------------------------
    #ingestar data
    im = cv2.imread('data_test/friends1.jpg')
    # detectar_rostro
    emotions,boxes_face = Detector.get_emotion(im)
    # visualizacion
    if len(emotions)!=0:
        img_post = fed.bounding_box(im,boxes_face,emotions)
    else:
        img_post = im
    cv2.imshow('emotion_detection',img_post)
    cv2.imwrite('result.jpg', img_post) 
    cv2.waitKey(0)