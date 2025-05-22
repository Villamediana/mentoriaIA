# Importa a biblioteca pandas, que serve para ler e trabalhar com tabelas de dados (Excel, CSV, etc)
import pandas as pd

# Importa o modelo de Regressão Logística, um modelo de classificação para prever 2 situações (sim ou não)
from sklearn.linear_model import LogisticRegression

# Lemos o csv chamado clima_sp.csv
tabela_clima = pd.read_csv("clima_sp.csv")

# Aqui pegamos apenas os cabeçalhos no csv: temperatura, umidade e vento
parametros = tabela_clima[["temperatura", "umidade", "vento"]]

# Pegamos a coluna que indica se choveu ou não (0 = não choveu, 1 = choveu)
resposta_correta = tabela_clima["choveu"]

# Criamos o modelo usando regressão logística
modelo_ia = LogisticRegression()

# Treinamos o modelo com os dados (entradas e respostas)
modelo_ia.fit(parametros, resposta_correta)

# Agora pedimos para o usuário digitar os dados do clima de hoje
temperatura_atual = float(input("Temperatura (°C): "))
umidade_atual = float(input("Umidade (%): "))
vento_atual = float(input("Vento (km/h): "))

# Organiza os dados digitados no formato que o modelo espera
clima_atual = [[temperatura_atual, umidade_atual, vento_atual]]


# Faz a previsão da probabilidade de chover com base nesses dados
#   [0] = probabilidade de não chover
#   [1] = probabilidade de chover
probabilidade_de_chuva = modelo_ia.predict_proba(clima_atual)[0][1]

# Mostra a porcentagem de chance de chover
print(f"\nProbabilidade de chuva: {probabilidade_de_chuva * 100:.2f}%")
