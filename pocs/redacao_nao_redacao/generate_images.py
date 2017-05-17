from setup_imagens import SetupImagens
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class GerarImagens(object):

    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        # this is the augmentation configuration we will use for training
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

    def gerar(self):

        setup = SetupImagens(url_base='http://projetoredacao.s3.amazonaws.com/files/')
        all_images_filename = setup.maybe_download('imagens.zip', 106087279)
        setup.maybe_extract(filename=all_images_filename, expected_folders=['redacao', 'nao_e_redacao'])
        ## pastas estão invertidas, então usa-se um dictionary para regularizar
        class_by_folder = {'redacao': 'nao_e_redacao', 'nao_e_redacao': 'redacao'}
        setup.split_training_test(class_by_folder=class_by_folder)
        average_width, average_height = setup.get_average_images_shape(['data/train/redacao', 'data/test/redacao', 'data/train/nao_e_redacao', 'data/test/nao_e_redacao'])

        # this is a generator that will read pictures found in subfolders of 'data/train', and indefinitely generate batches of augmented image data
        train_generator = self.get_train_generator('data/train', target_width=average_width, target_height=average_height)
        test_generator = self.get_test_generator('data/test', target_width=average_width, target_height=average_height)

        return train_generator, test_generator, average_width, average_height

    def get_train_generator(self, directory, target_width, target_height):
        return self.train_datagen.flow_from_directory(
            directory,  # this is the target directory
            target_size=(target_width, target_height),  # all images will be resized to average
            batch_size=self.batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    def get_test_generator(self, directory, target_width, target_height):
        return self.test_datagen.flow_from_directory(
            directory,
            target_size=(target_width, target_height),
            batch_size=self.batch_size,
            class_mode='binary')