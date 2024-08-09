# Importar las bibliotecas necesarias
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas de clase

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear un modelo de Bosque Aleatorio
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
rf_clf.fit(X_train, y_train)

# Realizar predicciones con los datos de prueba
y_pred = rf_clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del Bosque Aleatorio: {accuracy:.2f}")

# Visualización opcional de la importancia de las características
import matplotlib.pyplot as plt

# Obtener la importancia de las características
importances = rf_clf.feature_importances_
features = iris.feature_names

# Crear un gráfico de barras
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importancia de la característica')
plt.title('Importancia de las características en el Bosque Aleatorio')
plt.show()
