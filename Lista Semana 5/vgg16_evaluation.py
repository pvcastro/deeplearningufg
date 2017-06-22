from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras import optimizers
import numpy as np

model = load_model('vgg16.h5')
#model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

# inicializa a imagem de entrada na dimensão (224x224)
# Esse valor muda dependendo da rede neural utilizada
inputShape = (224, 224)
preprocess = imagenet_utils.preprocess_input

# carrega a imagem usando e a redimensiona para a entrada de rede neural
print("Carregando e pre-processando a imagem...")
imagem = load_img('data/test/cats/cat.1400.jpg', target_size=inputShape)
imagem = img_to_array(imagem)

# a imagem é representada por um vetor NumPy 
# da forma (inputShape[0], inputShape[1], 3) entretanto, precisamos expandir 
# a dimensão fazendo a forma (1, inputShape[0], inputShape[1], 3)
imagem = np.expand_dims(imagem, axis=0)

# pre-processamento da imagem usando a funcao apropriada 
# baseada na rede neural carregada (i.e., subtração de média, scaling, etc.)
imagem = preprocess(imagem)

# classifica a imagem
classe_predita = model.predict(imagem, verbose=1)
classe_predita = np.resize(classe_predita, (1, 1000))
P = imagenet_utils.decode_predictions(classe_predita)

# exibe as 5 maiores probabilidades de classes
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))