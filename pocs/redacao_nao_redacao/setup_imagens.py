import numpy, os, sys, zipfile, shutil

from six.moves.urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from PIL import Image

class SetupImagens(object):

    def __init__(self, url_base, data_root='.'):
        self.url_base = url_base
        self.data_root = data_root
        self.last_percent_reported = None

    def download_progress_hook(self, count, blockSize, totalSize):
        """A hook to report the progress of a download. This is mostly intended for users with
        slow internet connections. Reports every 5% change in download progress.
        """
        percent = int(count * blockSize * 100 / totalSize)

        if self.last_percent_reported != percent:
            if percent % 5 == 0:
                sys.stdout.write("%s%%" % percent)
                sys.stdout.flush()
            else:
                sys.stdout.write(".")
                sys.stdout.flush()

            self.last_percent_reported = percent

    def maybe_download(self, filename, expected_bytes, force=False):
        """Download a file if not present, and make sure it's the right size."""
        dest_filename = os.path.join(self.data_root, filename)
        if force or not os.path.exists(dest_filename):
            print('Attempting to download:', filename)
            filename, _ = urlretrieve(self.url_base + filename, dest_filename, reporthook=self.download_progress_hook)
            print('\nDownload Complete!')
        statinfo = os.stat(dest_filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', dest_filename)
        else:
            raise Exception('Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
        return dest_filename

    def maybe_extract(self, filename, expected_folders):
        print('Extracting data for %s. This may take a while. Please wait.' % self.data_root)
        file = zipfile.ZipFile(filename, 'r')
        sys.stdout.flush()
        file.extractall(self.data_root)
        file.close()
        data_folders = [
            os.path.join(self.data_root, d) for d in sorted(os.listdir(self.data_root))
            if os.path.isdir(os.path.join(self.data_root, d))]
        found_folders = []
        for folder in expected_folders:
            if os.path.join(self.data_root, folder) not in data_folders:
                print('Expected %s folder. Found %s instead.' % (folder, data_folders))
            else:
                found_folders.append(folder)
        print(found_folders)
        return found_folders

    def split_training_test(self, folder_names):
        print('Splitting each folder between training and test folders.')
        for folder in folder_names:
            folder_path = os.path.join(self.data_root, folder)
            ## Recria pasta de treino
            train_path = os.path.join(folder_path, 'train')
            if os.path.exists(train_path):
                shutil.rmtree(train_path)
            os.makedirs(train_path)
            ## Recria pasta de teste
            test_path = os.path.join(folder_path, 'test')
            if os.path.exists(test_path):
                shutil.rmtree(test_path)
            os.makedirs(test_path)
            ## Obtém somente nomes dos arquivos de imagem
            image_files = self.get_image_files(folder_path=folder_path)
            ## Faz o split dos dados para 70% de treino e 30% de teste
            training_data, test_data = train_test_split(image_files, test_size=0.3, random_state=42)
            ## Move da pasta de origem para a de teste ou treino
            for file in training_data:
                file_path = os.path.join(folder_path, file)
                shutil.move(file_path, train_path)
            for file in test_data:
                file_path = os.path.join(folder_path, file)
                shutil.move(file_path, test_path)

    def get_image_files(self, folder_path):
        return [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.jpg')]

    def load_image(self, folder, index=0):
        folder_path = os.path.join(self.data_root, folder)
        file_name = self.get_image_files(folder_path=folder_path)[index]
        return os.path.join(folder_path, file_name)

    def get_preview_dir(self, preview_dir_name):
        preview_path = os.path.join(self.data_root, preview_dir_name)
        if not os.path.exists(preview_path):
            os.makedirs(preview_path)
        return preview_path

    def get_average_images_shape(self, folders):
        total_width = []
        total_height = []
        total_images = 0
        for folder in folders:

            folder_path = os.path.join(self.data_root, folder)

            # Access all jpg files in directory
            allfiles = os.listdir(folder_path)
            imlist = [filename for filename in allfiles if filename[-4:] in [".jpg", ".JPG"]]
            total_images += len(imlist)

            for file_name in imlist:
                image_path = os.path.join(folder_path, file_name)
                image = Image.open(image_path)
                width, height = image.size
                total_width.append(width)
                total_height.append(height)

        average_width = numpy.mean(total_width)
        average_height = numpy.mean(total_height)

        return average_width, average_height