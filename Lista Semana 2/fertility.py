import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas
from mlp import MultiLayerPerceptron

## Cria dataframe com os dados de fertilidade
fertility_df = pandas.read_csv('fertility_Diagnosis.txt', header=-1)
normalized_features = ['Age','Number of hours spent sitting per day ene-16']
categorical_features = ['Season','Childish diseases','Accident or serious trauma','Surgical intervention',
          'High fevers in the last year','Frequency of alcohol consumption','Smoking habit']
fertility_df.columns = ['Season','Age','Childish diseases','Accident or serious trauma','Surgical intervention',
          'High fevers in the last year','Frequency of alcohol consumption','Smoking habit',
          'Number of hours spent sitting per day ene-16','Output']
## Mapeia a coluna de saída dos dados para valores numéricos
fertility_df['Output'] = fertility_df['Output'].map({'N': 0, 'O': 1}).astype(int)
## Mapeia as colunas de dados categóricos para valores descritivos, para serem usados no one hot encoding
fertility_df['Season'] = fertility_df['Season'].map({-1: 'winter', -0.33: 'spring', 0.33: 'summer', 1: 'fall'}).astype(str)
fertility_df['Childish diseases'] = fertility_df['Childish diseases'].map({0: 'Yes', 1: 'No'}).astype(str)
fertility_df['Accident or serious trauma'] = fertility_df['Accident or serious trauma'].map({0: 'Yes', 1: 'No'}).astype(str)
fertility_df['Surgical intervention'] = fertility_df['Surgical intervention'].map({0: 'Yes', 1: 'No'}).astype(str)
fertility_df['High fevers in the last year'] = fertility_df['High fevers in the last year'].map({-1: 'Less than three months ago', 0: 'More than three months ago', 1: 'No'}).astype(str)
fertility_df['Frequency of alcohol consumption'] = fertility_df['Frequency of alcohol consumption'].map({0.2: 'several times a day', 0.4: 'every day', 0.6: 'several times a week', 0.8: 'once a week', 1.0: 'hardly ever or never'}).astype(str)
fertility_df['Smoking habit'] = fertility_df['Smoking habit'].map({-1: 'Never', 0: 'Occasional', 1: 'Daily'}).astype(str)

## Faz o one hot encoding das colunas categórias
fertility_df = pandas.get_dummies(fertility_df, columns=categorical_features)

## Faz o split dos dados para 70% de treino e 30% de teste
training_data, test_data, training_output, test_output = train_test_split(fertility_df, fertility_df['Output'], test_size=0.3)
## Remove os dados de saída dos dados de treino e teste
del training_data['Output']
del test_data['Output']

## Normalização dos dados de entrada
normalized_training_data = training_data
normalized_training_data[normalized_features] = preprocessing.normalize(training_data[normalized_features])

## Realiza o treinamento do MLP
mlp = MultiLayerPerceptron(
    neuronios_por_camada=[len(normalized_training_data.columns),1],
    entradas=normalized_training_data, desejados=training_output.tolist(),
    taxa_aprendizagem=0.1, epocas=1000, precisao=1e-7, plot=False
)
mlp.treinar()