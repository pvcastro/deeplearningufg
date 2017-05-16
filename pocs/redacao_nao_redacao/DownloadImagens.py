import os
import sys
import zipfile

from six.moves.urllib.request import urlretrieve

class DownloadImagens(object):

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