import cv2
from imutils.video import VideoStream
import os
import face_recognition
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import imutils



def prepare_image(face_image):
    face_image = cv2.resize(face_image, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = face_image.astype("float") / 255.0
    face_image= img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def prepare_image_tf1(face_image):
    face_image = cv2.resize(face_image, (48,48))
    face_image = face_image.astype("float") / 255.0
    face_image= img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def prepare_image_tf2(face_image):
    face_image = cv2.resize(face_image, (128,128))
    face_image = face_image.astype("float") / 255.0
    face_image= img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image

def preprocess_img(face_image,rgb=True,w=48,h=48):
    face_image = cv2.resize(face_image, (w,h))
    if rgb == False:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = face_image.astype("float") / 255.0
    face_image= img_to_array(face_image)
    face_image = np.expand_dims(face_image, axis=0)
    return face_image


def bounding_box(img,box,match_name):
    for i in np.arange(len(box)):
        y0,x1,y1,x0 = box[i]
        img = cv2.rectangle(img,
                      (x0,y0),
                      (x1,y1),
                      (0,255,0),3);
        cv2.putText(img, match_name, (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img

def sel_load_model(model_select):
    if model_select == 1:
        name_model = 'model_baseline_v1.hdf5'
        w,h = (48,48)
        rgb = False
    elif model_select == 2:
        name_model = 'model_v1.hdf5 '
        w,h = (48,48)
        rgb = False
    elif model_select == 3:
        name_model = 'model_baseline_dropout.h5'
        w,h = (48,48)
        rgb = False
    elif model_select == 4:
        name_model = 'model_dropout.hdf5'
        w,h = (48,48)
        rgb = False
    elif model_select == 5:
        name_model = 'model_baseline_tf_learning.h5'
        w,h = (48,48)
        rgb = True
    elif model_select == 6:
        name_model = 'model_tf_learning_mobilnet.hdf5'
        w,h = (128,128)
        rgb = True
    elif model_select == 7:
        name_model = 'model_tf_learning_ResNet50.hdf5'
        w,h = (48,48)
        rgb = True
    elif model_select == 8:
        name_model = 'model_tf_learning_InceptionV3.hdf5'
        w,h = (75,75)
        rgb = True
    elif model_select == 9:
        name_model = 'model_tf_learning_VGG16.hdf5'
        w,h = (48,48)
        rgb = True

    name_model = './Modelos/'+name_model  
    model = load_model(name_model)
    return model,rgb,w,h

# inicializacion
'''
input = 'video' o 'imagen'
'''
input = 'video'
'''
1. model_baseline_v1
2. model_v1
3. model_baseline_dropout
4. model_dropout
5. model_baseline_tf_learning
6. model_tf_learning
'''
sel_model = 4
model,rgb,w,h = sel_load_model(sel_model)
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

if input == 'imagen':
    #ingestar data
    im = cv2.imread('neutral.jpeg')
    # detectar_rostro
    box = face_recognition.face_locations(im)
    x0,y1,x1,y0 = box[0]
    face_image = im[x0:x1,y0:y1]
    # preprocesar data
    '''
    if sel_model in [1,2,3,4]:
        face_image = prepare_image(face_image)
    elif sel_model == 5:
        face_image = prepare_image_tf1(face_image)
    elif sel_model == 6:
        face_image = prepare_image_tf2(face_image)
    '''
    face_image = preprocess_img(face_image,rgb,w,h)
    # predecir imagen
    prediction = model.predict(face_image)[0]
    emotion = labels[prediction.argmax()]
    # visualizacion
    img_post = bounding_box(im,box,emotion)
    cv2.imshow('emotion_detection',img_post)

    cv2.imwrite('P_neutral.jpg', img_post) 

    cv2.waitKey(0)

if input == 'video':
    # Inicializar
    vs = VideoStream(src=0).start()
    while True:
        im = vs.read()
        im = imutils.resize(im, width=720)

        try:
            # detectar_rostro
            box = face_recognition.face_locations(im)
            x0,y1,x1,y0 = box[0]
            face_image = im[x0:x1,y0:y1]
            # preprocesar data
            face_image = preprocess_img(face_image,rgb,w,h)
            # predecir imagen
            prediction = model.predict(face_image)[0]
            emotion = labels[prediction.argmax()]
            # visualizacion
            img_post = bounding_box(im,box,emotion)
        except:
            img_post = im
        cv2.imshow('emotion_detection',img_post)
        if cv2.waitKey(1) &0xFF == ord('q'):
            break


