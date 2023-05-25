import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

# Carregando os dados
with open('breast.pkl', 'rb') as f:
    dataset = pickle.load(f)

# Separando os dados de treinamento e teste
X_train = dataset[0]  # Características de treino
X_test = dataset[1]  # Características de teste
y_train = dataset[2]  # Rótulos de treino
y_test = dataset[3]  # Rótulos de teste

# Definindo a estrutura da rede neural
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilando o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinando a rede neural
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Obtendo as previsões do modelo
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Transformando as probabilidades em rótulos binários

# Imprimindo o relatório de classificação
print(classification_report(y_test, y_pred, zero_division=1))


