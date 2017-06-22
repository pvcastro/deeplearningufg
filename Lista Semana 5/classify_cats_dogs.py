"""
Author: Abner Ayala-Acevedo

Dataset: Kaggle Dataset Dogs vs Cats
https://www.kaggle.com/c/dogs-vs-cats/data
- test folder unlabelled data

Example: Dogs vs Cats (Directory Structure)
test_dir/
    test/
        001.jpg
        002.jpg
        ...
        cat001.jpg
        cat002.jpg
        ...

If you need using a multi-class classification model change binary_cross_entropy to categorical_cross_entropy
"""

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as k
from keras.models import load_model

# parameters dependent on your dataset: modified to your example
batch_size = 32  # try 4, 8, 16, 32, 64, 128, 256 dependent on CPU/GPU memory capacity (powers of 2 values).

def classify(img_width, img_height, model_name):
    model = load_model(model_name)

    # Read Data
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory('data/test',
                                                      target_size=(img_width, img_height),
                                                      batch_size=batch_size,
                                                      shuffle=False)

    scoreSeg = model.evaluate_generator(test_generator, 400)
    print("Accuracy = ", scoreSeg[1])


classify(224, 224, 'vgg16.h5')
classify(224, 224, 'vgg19.h5')
classify(299, 299, 'xception.h5')
classify(224, 224, 'resnet.h5')

# release memory
k.clear_session()
