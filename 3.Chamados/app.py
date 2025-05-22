from sklearn.tree import DecisionTreeClassifier

# 1. Dados de treino: pares [impacto, consegue_trabalhar]
#    impacto: 1=baixo, 2=médio, 3=alto
#    consegue_trabalhar: 1=Sim, 2=Não
problemas = [
    [1, 1],  # baixo impacto + consegue trabalhar → 48h
    [1, 2],  # baixo impacto + não consegue trabalhar → 24h
    [2, 1],  # médio impacto + consegue trabalhar → 24h
    [2, 2],  # médio impacto + não consegue trabalhar → 2h
    [3, 1],  # alto impacto + consegue trabalhar → 2h
    [3, 2],  # alto impacto + não consegue trabalhar → 2h
]

# 2. Classes correspondentes ao tempo de resolução
tempos = ["48h", "24h", "24h", "2h", "2h", "2h"]

# 3. Criação e treinamento do modelo
modelo = DecisionTreeClassifier()
modelo.fit(problemas, tempos)

# 4. Entrada: descrição do problema (para registro, não afeta a predição)
print("Descreva seu problema:")
descricao = input().strip()

# 5. Entrada: nível de impacto
print("Qual o impacto?\n1. Só a mim\n2. Meu departamento\n3. O prédio")
impacto = int(input().strip())

# 6. Entrada: ainda consegue trabalhar?
print("Você ainda consegue trabalhar?\n1. Sim\n2. Não")
resposta = int(input().strip())

# 7. Predição do tempo estimado
tempo_estimado = modelo.predict([[impacto, resposta]])[0]

# 8. Saída: exibe o resultado
print("Tempo estimado para resolução:", tempo_estimado)
