from setup_imagens import SetupImagens

setup = SetupImagens(url_base='http://projetoredacao.s3.amazonaws.com/files/')
all_images_filename = setup.maybe_download('imagens.zip', 106087279)
all_folders = setup.maybe_extract(filename=all_images_filename, expected_folders=['redacao', 'nao_e_redacao'])
class_by_folder = {'redacao': 'nao_e_redacao', 'nao_e_redacao': 'redacao'}
setup.split_training_test(class_by_folder=class_by_folder)