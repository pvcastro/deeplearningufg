import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
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
training_data, test_data, training_output, test_output = train_test_split(fertility_df, fertility_df['Output'], test_size=0.3, random_state=0)
## Remove os dados de saída dos dados de treino e teste
del training_data['Output']
del test_data['Output']

## Normalização dos dados de entrada
normalized_training_data = training_data
normalized_training_data.loc[:,normalized_features] = preprocessing.normalize(training_data[normalized_features])

## Realiza o treinamento do MLP
training_data_matrix = normalized_training_data.as_matrix()
training_output_list = training_output.tolist()
quantidade_features = training_data_matrix.shape[1]

mlp = MultiLayerPerceptron(
    numero_de_entradas=quantidade_features, neuronios_por_camada=[quantidade_features, 1],
    taxa_aprendizagem=0.5, epocas=3000, precisao=0, debug_training=False, plot=True
)

mlp.treinar(matriz_entradas=training_data_matrix, valores_desejados=np.array(training_output_list))

normalized_test_data = test_data
normalized_test_data.loc[:,normalized_features] = preprocessing.normalize(test_data[normalized_features])
test_data_matrix = normalized_test_data.as_matrix()

previsoes = []
previsoes_ajustadas = []
for amostra_teste in test_data_matrix:
    previsao = mlp.prever(amostra_teste)
    previsoes.append(previsao)
    if np.greater_equal(previsao, [0.5]):
        previsao_arredondada = 1
    else:
        previsao_arredondada = 0
    previsoes_ajustadas.append(previsao_arredondada)

test_output_list = test_output.tolist()

print("------------------------------------------Resultados------------------------------------------")
print("Previsões arredondadas", previsoes_ajustadas)
print("Previsões", previsoes)
print("Saídas desejadas", test_output_list)

erros = 0
tamanho = len(test_output_list)
for i in range(tamanho):
    if test_output_list[i] != previsoes_ajustadas[i]:
        erros += 1
acertos = tamanho - erros
accuracy = float(acertos)/tamanho
print("-----------------------------------------------------")
print('# Pacientes corretamente classificados =', acertos)
print('# Pacientes incorretamente classificados =', erros)
print('# Total de pacientes =', tamanho)
print("-----------------------------------------------------")
print('Precisão = %.2f' % accuracy)
print("")
print("Matriz de confusão:")
print(confusion_matrix(test_output_list, previsoes_ajustadas))
report = classification_report(y_true=test_output_list, y_pred=previsoes_ajustadas, target_names=['Normal', 'Altered'])
print("")
print("Relatório:")
print(report)