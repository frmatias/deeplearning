from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.layers.normalization import BatchNormalization

import numpy as np
import pandas as np
import matplotlib.pyplot as plt

# Specifying the path of the data(train,test,validaton)
train = 'dataset/train/'
test = 'dataset/test/' 
val = 'dataset/val'

img_width,img_height= 800,800
input_shape = (img_width,img_height,3)

classificador = Sequential()

classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))
classificador.add(Conv2D(32, (3,3), input_shape = input_shape, activation = 'relu',padding="same"))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2),padding="same"))



classificador.add(Flatten())

classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 2000, activation = 'relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classificador.summary()


train_datagen = ImageDataGenerator(rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)


# Here we import images directly from Directory by using flow_from_directory method.
#flow_from_directory() automatically infers the labels from the directory structure of the folders containing images
train_generator = train_datagen.flow_from_directory(
    train,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    val,
    target_size=(img_width, img_height),
    batch_size=16,
    class_mode='binary')

#We Fit the model here using fit_generator as we are dealing with large datasets.
#We Fit the model here using fit_generator as we are dealing with large datasets.
classificador.fit_generator(
    train_generator,
    steps_per_epoch=5217//16,
    epochs=10000,
    validation_data=validation_generator,
    validation_steps=17 // 16)

#Accuracy of test data.
scores = classificador.evaluate_generator(test_generator,624/16)
print("\nAccuracy:"+" %.2f%%" % ( scores[1]*100))

# saving model in H5 format.
model.save('vison_v2.0.h5')

# saving model in Json format.
model_json = classificador.to_json()
with open("model2.json","w") as json_file:
    json_file.write(model_json)
    




image_teste = image.load_img('../input/chest_xray/chest_xray/train/NORMAL/NORMAL2-IM-0927-0001.jpeg', target_size = (img_width,img_height))

image_teste = image.img_to_array(image_teste)

image_teste /= 255

image_teste = np.expand_dims(image_teste, axis = 0)

previsao = classificador.predict(image_teste)

    
if(previsao >= 0.5):
    aux = previsao
    aux = aux * 100
    print('Normal:', round(aux[0][0],2) ,'%')

if(previsao < 0.5):
    aux2 = previsao
    aux2 = 1-aux2
    aux2 = aux2 * 100
    print('Pneu:', round(aux2[0][0],2) ,'%')
    


#ImageDataGenerator-Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches).