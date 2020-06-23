# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Sequential, load_model, model_from_json
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# %% [markdown]
# # Inicializacion

# %%
Modelos = {}
# %%
'''
1. model_baseline_v1
2. model_v1
3. model_baseline_dropout
4. model_dropout
5. model_baseline_tf_learning
6. model_tf_learning
7. model_tf_learning2
8. model_tf_learning3
9. model_tf_learning4
'''

for model_select in [1,2,3,4,5,6,7,8,9]:
    if model_select == 1:
        name_model = 'model_baseline_v1.hdf5'
    elif model_select == 2:
        name_model = 'model_v1.hdf5'
    elif model_select == 3:
        name_model = 'model_baseline_dropout.h5'
    elif model_select == 4:
        name_model = 'model_dropout.hdf5'
    elif model_select == 5:
        name_model = 'model_baseline_tf_learning.h5'
    elif model_select == 6:
        name_model = 'model_tf_learning_mobilnet.hdf5'
    elif model_select == 7:
        name_model = 'model_tf_learning_ResNet50.hdf5'
    elif model_select == 8:
        name_model = 'model_tf_learning_InceptionV3.hdf5'
    elif model_select == 9:
        name_model = 'model_tf_learning_VGG16.hdf5'

    name_model = './Modelos/'+name_model


    # %%
    def f_cargar_modelo2():
        from keras.models import model_from_json
        # load json and create model
        json_file = open('model_dropout.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model_weights_dropout.h5")
        return loaded_model

    def f_cargar_modelo(name_model):
        model = load_model(name_model)
        return model


    # %%
    # Rutas de interes
    path_dataset = '/Users/macbook/GoogleDrive/Emotion_detection/Dataset/'
    path_train = path_dataset + 'train'
    path_validation = path_dataset + 'validation'

    # Info data
    num_classes = 7
    labels = ['angry','disgust','fear','happy','neutral','sad','surprise']
    w,h = 48,48
    batch_size = 512
    nb_train_samples = 28821
    nb_validation_samples = 7066


    # %%
    # Inicializndao el generador de imagenes
    if model_select in [1,2,3,4]: 
        val_datagen = ImageDataGenerator(rescale=1./255)
        val_generator = val_datagen.flow_from_directory(
            directory=path_validation,
            target_size=(w,h),
            color_mode='grayscale',
            class_mode='categorical')
    elif model_select in [5,7,9]:
        val_datagen = ImageDataGenerator(rescale=1./255)
        val_generator = val_datagen.flow_from_directory(
            directory=path_validation,
            target_size=(48,48),
            color_mode='rgb',
            class_mode='categorical')
    elif model_select == 6:
        w,h = 128,128
        val_datagen = ImageDataGenerator(rescale=1./255)
        val_generator = val_datagen.flow_from_directory(
            directory=path_validation,
            target_size=(w,h),
            color_mode='rgb',
            class_mode='categorical')
    elif model_select == 8:
        w,h = 75,75
        val_datagen = ImageDataGenerator(rescale=1./255)
        val_generator = val_datagen.flow_from_directory(
            directory=path_validation,
            target_size=(w,h),
            color_mode='rgb',
            class_mode='categorical')


    # %%
    # cargo el modelo
    model = f_cargar_modelo(name_model)

    class_labels = val_generator.class_indices
    class_labels = {v: k for k, v in class_labels.items()}
    classes = list(class_labels.values())

    #Confution Matrix and Classification Report
    Y_pred = model.predict_generator(val_generator)
    y_pred = np.argmax(Y_pred, axis=1)





    # %%
    model.evaluate(val_generator)


    # %%
    loss,val_acc = model.evaluate(val_generator)
    precisions, recall, f1_score, _ = sklearn.metrics.precision_recall_fscore_support(val_generator.classes, y_pred)





    # %%
    metrics ={
        'val_acc':val_acc,
        'loss':loss,
        'precision':precisions,
        'recall':recall,
        'f1':f1_score}
    print(metrics)

    # %%
    nm = name_model.split('/')[-1]
    Modelos[nm]=metrics



# guardar
import pickle
with open('metric_models.pickle', 'wb') as f:
    pickle.dump(Modelos, f)
print('modelo guardado en el archivo: metric_models.pickle')

'''
# cargar
with open('metric_models.pickle', 'rb') as f:
    var_you_want_to_load_into = pickle.load(f)
'''