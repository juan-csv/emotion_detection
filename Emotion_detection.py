'''
python Emotion_detection.py --input webcam
python Emotion_detection.py --input image --path data_test/friends1.jpg
'''
import f_emotion_detection as fed
import cv2
from imutils.video import VideoStream
import imutils
import argparse

parser = argparse.ArgumentParser(description="Emotions detection")
parser.add_argument('--input', type=str, default= 'webcam',
                    help="webcam or image")
parser.add_argument('--path', type=str,
                    help="path of image")
args = vars(parser.parse_args())

# instanciar detector
Detector = fed.predict_emotions()
type_input = args['input']

if type_input == 'image':
    # ----------------------------- image -----------------------------
    #ingestar data
    im = cv2.imread(args['path'])
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

if type_input == 'webcam':
    # ----------------------------- video -----------------------------
    #ingestar data
    vs = VideoStream(src=0).start()
    while True:
        im = vs.read()
        im = imutils.resize(im, width=720)
        # detectar_rostro
        emotions,boxes_face = Detector.get_emotion(im)
        # visualizacion
        if len(emotions)!=0:
            img_post = fed.bounding_box(im,boxes_face,emotions)
        else:
            img_post = im
        cv2.imshow('emotion_detection',img_post)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break