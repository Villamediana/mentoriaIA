from sklearn.linear_model import LogisticRegression

# Dados de entrada: [forma, textura]
# Forma: 0 = redonda, 1 = achatada
# Textura: 0 = lisa, 1 = rugosa
dados = [
    [0.3, 0],  # maçã
    [0.4, 0],  # maçã
    [0.1, 1],  # laranja
    [0.2, 1]   # laranja
]

# Respostas: 0 = maçã, 1 = laranja
respostas = [0, 0, 1, 1]

# Criar e treinar o modelo
modelo = LogisticRegression()
modelo.fit(dados, respostas)

# Fruta nova para testar (levemente achatada e lisa)
nova_fruta = [[0.35, 0]]
resultado = modelo.predict(nova_fruta)

# Resultado
print("Resultado:", "Maçã" if resultado[0] == 0 else "Laranja")
