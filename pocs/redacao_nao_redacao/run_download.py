from DownloadImagens import DownloadImagens

download = DownloadImagens(url_base='http://projetoredacao.s3.amazonaws.com/files/')

all_images_filename = download.maybe_download('imagens.zip', 106087279)

all_folders = download.maybe_extract(filename=all_images_filename, expected_folders=['redacao','nao_e_redacao'])