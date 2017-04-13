import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas
from mlp import MultiLayerPerceptron
from imblearn.under_sampling import NearMiss

## Cria dataframe com os dados de fertilidade
fertility_df = pandas.read_csv('fertility_Diagnosis.txt', header=-1)
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

## Retira dados de saída das amostras
fertility_df_output = fertility_df['Output']
del fertility_df['Output']

## Faz o one hot encoding das colunas categórias
fertility_df = pandas.get_dummies(fertility_df, columns=categorical_features)

## Faz o balanceamento dos dados, baseado nas saídas desbalanceadas do conjunto de dados
nm = NearMiss(random_state=42)
fertility_df_balanced, fertility_output_balanced = nm.fit_sample(fertility_df, fertility_df_output)
# fertility_df_balanced = fertility_df.as_matrix()
# fertility_output_balanced = fertility_df_output.tolist()

## Faz o split dos dados para 70% de treino e 30% de teste
training_data, test_data, training_output, test_output = train_test_split(fertility_df_balanced, fertility_output_balanced, test_size=0.3, random_state=42)

## Realiza o treinamento do MLP
quantidade_features = training_data.shape[1]

mlp = MultiLayerPerceptron(
    numero_de_entradas=quantidade_features, neuronios_por_camada=[quantidade_features, 1],
    taxa_aprendizagem=0.5, epocas=5000, precisao=0, debug_training=False, plot=False
)

mlp.treinar(matriz_entradas=training_data, valores_desejados=np.array(training_output))

mlp.avaliar(saidas_esperadas=test_output, amostras_de_teste=test_data)

mlp.avaliar(saidas_esperadas=fertility_df_output.tolist(), amostras_de_teste=fertility_df.as_matrix())