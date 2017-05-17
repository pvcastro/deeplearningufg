from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from generate_images import GerarImagens

geracao_imagens = GerarImagens()
train_generator, test_generator, target_width, target_height = geracao_imagens.gerar()

if K.image_data_format() == 'channels_first':
    input_shape = (3, target_width, target_height)
else:
    input_shape = (target_width, target_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch=2000, epochs=50, validation_data=test_generator, validation_steps=800)
model.save_weights('first_try.h5')  # always save your weights after training or during training