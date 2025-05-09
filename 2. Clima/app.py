# Importa a biblioteca pandas, que serve para ler e trabalhar com tabelas de dados (como se fosse Excel)
import pandas as pd

# Importa o modelo de Regressão Logística, que é uma forma simples de inteligência artificial para prever 2 situações (ex: sim ou não)
from sklearn.linear_model import LogisticRegression

# Lê o arquivo chamado clima_sp.csv, que tem os dados históricos do clima (temperatura, umidade, vento e se choveu)
tabela_clima = pd.read_csv("clima_sp.csv")

# Aqui pegamos apenas as colunas de entrada que a IA vai usar para aprender: temperatura, umidade e vento
parametros = tabela_clima[["temperatura", "umidade", "vento"]]

# Aqui pegamos a coluna que indica se choveu ou não naquele dia (0 = não choveu, 1 = choveu)
resposta_correta = tabela_clima["choveu"]

# Criamos o modelo de IA usando regressão logística
modelo_ia = LogisticRegression()

# Treinamos o modelo com os dados históricos (entradas e respostas)
modelo_ia.fit(parametros, resposta_correta)

# Agora pedimos para o usuário digitar os dados do clima de hoje
temperatura_atual = float(input("Temperatura (°C): "))
umidade_atual = float(input("Umidade (%): "))
vento_atual = float(input("Vento (km/h): "))

# Organiza os dados digitados no formato que o modelo espera
clima_atual = [[temperatura_atual, umidade_atual, vento_atual]]

# Faz a previsão da probabilidade de chover com base nesses dados
# A função predict_proba retorna dois números:
#   [0] = probabilidade de NÃO chover
#   [1] = probabilidade de chover
probabilidade_de_chuva = modelo_ia.predict_proba(clima_atual)[0][1]

# Mostra apenas a porcentagem de chance de chover (arredondada)
print(f"\nProbabilidade de chuva: {round(probabilidade_de_chuva * 100)}%")
