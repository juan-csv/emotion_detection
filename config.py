# haar cascade para la detecction de rostro
detect_frontal_face = 'haarcascades/haarcascade_frontalface_alt.xml'
# modelo de deteccion de emociones
path_model = './Modelos/model_dropout.hdf5'
# Parametros del modelo, la imagen se debe convertir a una de tamaño 48x48 en escala de grises
w,h = 48,48
rgb = False
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
