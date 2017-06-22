#extraído do site: https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint




def train(network, model_name, img_width, img_height):

    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    nb_train_samples = 2000
    nb_validation_samples = 800
    epochs = 50
    batch_size = 16

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in network.layers:
        layer.trainable = False

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    print(network.output_shape, network.output_shape[1:])
    top_model.add(Flatten(input_shape=network.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(2, activation='softmax'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    # top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model = Model(input=network.input, output=top_model(network.output))

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=[ModelCheckpoint(model_name + '.h5', save_best_only=True)],
        verbose=2
    )

#def evaluate(model):


vgg16 = applications.VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
print('VGG16 Model loaded.')

vgg19 = applications.VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
print('VGG19 Model loaded.')

#inception = applications.InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299,299,3)))
#print('Inception Model loaded.')

xception = applications.Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(299,299,3)))
print('Xception Model loaded.')

resnet = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))
print('Resnet Model loaded.')

train(vgg16, 'vgg16', img_width=224, img_height=224)
train(vgg19, 'vgg19', img_width=224, img_height=224)
# train(inception, 'inception', img_width=299, img_height=299)
train(xception, 'xception', img_width=299, img_height=299)
train(resnet, 'resnet', img_width=224, img_height=224)

