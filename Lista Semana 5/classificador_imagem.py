# www.deeplearninbrasil.com.br
# uso via linha de comando
# python classificador_imagem.py --imagem images/soccer_ball.jpg --model vgg16

from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import Xception # TensorFlow apenas
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

# parse de argumentos da linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagem", required=True,
	help="caminho da imagem de entrada")
ap.add_argument("-model", "--model", type=str, default="vgg16",
	help="nome da rede neural prétreinada a ser usada (vgg16, vgg19, inception, xception, resnet")
args = vars(ap.parse_args())

MODELS = {
	"vgg16": VGG16,
	"vgg19": VGG19,
	"inception": InceptionV3,
	"xception": Xception, # funciona apenas se o keras está usando o tensorflow
	"resnet": ResNet50
}

# verifica se o usuario informou um modelo de rede neural valido via linha de comando
if args["model"] not in MODELS.keys():
	raise AssertionError("Informe uma rede neural disponível: vgg16, vgg19, inception, xception ou resnet")

# inicializa a imagem de entrada na dimensão (224x224) 
# Esse valor muda dependendo da rede neural utilizada
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# Para redes inceptionV3 ou Xception, a dimensão é (299x299) 
if args["model"] in ("inception", "xception"):
	inputShape = (299, 299)
	preprocess = preprocess_input

# Carrega a rede neural a partir do disco, 
# na primeira execução faz o download da internet
print("Carregando {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

# carrega a imagem usando e a redimensiona para a entrada de rede neural
print("Carregando e pre-processando a imagem...")
imagem = load_img(args["imagem"], target_size=inputShape)
imagem = img_to_array(imagem)

# a imagem é representada por um vetor NumPy 
# da forma (inputShape[0], inputShape[1], 3) entretanto, precisamos expandir 
# a dimensão fazendo a forma (1, inputShape[0], inputShape[1], 3)
imagem = np.expand_dims(imagem, axis=0)

# pre-processamento da imagem usando a funcao apropriada 
# baseada na rede neural carregada (i.e., subtração de média, scaling, etc.)
imagem = preprocess(imagem)

# classifica a imagem
print("Classificando a imagem com a rede neural '{}'...".format(args["model"]))
classe_predita = model.predict(imagem)
P = imagenet_utils.decode_predictions(classe_predita)

# exibe as 5 maiores probabilidades de classes
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# Abre a imagem via OpenCV
orig = cv2.imread(args["imagem"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Classe: {}, {:.2f}%".format(label, prob * 100),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classificação", orig)
cv2.waitKey(0)