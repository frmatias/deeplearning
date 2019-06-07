# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, image
import numpy as np


classificador = Sequential()

classificador.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255, rotation_range = 7, horizontal_flip = True, shear_range = 0.2, height_shift_range = 0.07, zoom_range = 0.2)

gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set', target_size = (64,64), batch_size = 32, class_mode = 'binary')

base_teste = gerador_teste.flow_from_directory('dataset/test_set', target_size = (64,64), batch_size = 32, class_mode = 'binary')

classificador.fit_generator(base_treinamento, steps_per_epoch = 4000 / 32, epochs = 5, validation_data = base_teste, validation_steps = 1000/32)

base_treinamento.class_indices

image_teste = image.load_img('dataset/test_set/cachorro/dog.3500.jpg', target_size = (64,64))

image_teste = image.img_to_array(image_teste)

image_teste /= 255

image_teste = np.expand_dims(image_teste, axis = 0)

previsao = classificador.predict(image_teste)

    
if(previsao >= 0.5):
    aux = previsao
    aux = aux * 100
    print('Gato:', round(aux[0][0],2) ,'%')

if(previsao < 0.5):
    aux2 = previsao
    aux2 = 1-aux2
    aux2 = aux2 * 100
    print('Cachorro:', round(aux2[0][0],2) ,'%')
    
