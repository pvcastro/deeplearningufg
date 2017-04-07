import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from mlp import MultiLayerPerceptron
from util import normalize_features

## Cria dataframe com os dados de fertilidade
fertility_df = pandas.read_csv('fertility_Diagnosis.txt',header=-1)
fertility_df.columns = ['Season','Age','Childish diseases','Accident or serious trauma', 'Surgical intervention',
                        'High fevers in the last year', 'Frequency of alcohol consumption','Smoking habit',
                        'Number of hours spent sitting per day ene-16','Output']
## Mapeia a coluna de saída dos dados para valores numéricos
fertility_df['Output'] = fertility_df['Output'].map({'N': 0, 'O': 1}).astype(int)

## Faz o split dos dados para 70% de treino e 30% de teste
training_data, test_data, training_output, test_output = train_test_split(fertility_df, fertility_df['Output'], test_size=0.3)
## Remove os dados de saída dos dados de treino e teste
del training_data['Output']
del test_data['Output']

## Normalização dos dados de entrada
normalized_training_data, normas = normalize_features(training_data)

## 

mlp = MultiLayerPerceptron(
    neuronios_por_camada=[2,2],
    entradas=[0.05, 0.1], desejados=np.array([0.01, 0.99]),
    taxa_aprendizagem=0.5, epocas=1, precisao=1e-7, plot=False
)

mlp.treinar()