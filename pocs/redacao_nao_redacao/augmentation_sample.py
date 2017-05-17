from setup_imagens import SetupImagens
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

setup = SetupImagens(url_base='http://projetoredacao.s3.amazonaws.com/files/')

all_images_filename = setup.maybe_download('imagens.zip', 106087279)

all_folders = setup.maybe_extract(filename=all_images_filename, expected_folders=['redacao', 'nao_e_redacao'])

setup.split_training_test(folder_names=all_folders)

average_width, average_height = setup.get_average_images_shape(['redacao/train', 'redacao/test', 'nao_e_redacao/train', 'nao_e_redacao/test'])

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

image_name = setup.load_image(folder='redacao/train')

preview_dir = setup.get_preview_dir('preview')

img = load_img(image_name)  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir=preview_dir, save_prefix='augment', save_format='jpg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely