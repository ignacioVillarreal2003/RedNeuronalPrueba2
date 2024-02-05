import tensorflow as tf
import tensorflow_datasets as tfds

#Descargar set de datos de Fashion MNIST de Zalando
datos, metadatos = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

datos_entrenamiento, datos_pruebas = datos['train'], datos['test'] #Obtenemos los datos de entrenamiento y pruebas
nombres_clases = metadatos.features['label'].names #Etiquetas de las 10 categorias posibles


#Funcion de normalizacion para los datos (Pasar de 0-255 a 0-1)
def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)
  imagenes /= 255
  return imagenes, etiquetas

#Normalizar los datos y usamos cache
datos_entrenamiento = datos_entrenamiento.map(normalizar).cache()
datos_pruebas = datos_pruebas.map(normalizar).cache()


#Crear el modelo
modelo = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28,1)),
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax) #Para redes de clasificacion
])

#Compilar el modelo
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)


#Los numeros de datos en entrenamiento y pruebas (60k y 10k)
num_ej_entrenamiento = metadatos.splits["train"].num_examples
num_ej_pruebas = metadatos.splits["test"].num_examples

#El trabajo por lotes permite que entrenamientos con gran cantidad de datos se haga de manera mas eficiente
TAMANO_LOTE = 32

#Shuffle y repeat hacen que los datos esten mezclados de manera aleatoria para que la red no se vaya a aprender el orden de las cosas
datos_entrenamiento = datos_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(TAMANO_LOTE)
datos_pruebas = datos_pruebas.batch(TAMANO_LOTE)

import math

#Entrenar
historial = modelo.fit(datos_entrenamiento, epochs=5, steps_per_epoch= math.ceil(num_ej_entrenamiento/TAMANO_LOTE))



import numpy as np
import matplotlib.pyplot as plt

# Tomar un lote de imágenes de prueba
for imagenes_prueba, etiquetas_prueba in datos_pruebas.take(1):
    # Tomar una sola imagen del lote
    imagen = imagenes_prueba[22]
    etiqueta = etiquetas_prueba[22]

    # Preparar la imagen para la predicción
    imagen = tf.reshape(imagen, (1, 28, 28, 1))

    # Realizar la predicción
    prediccion = modelo.predict(imagen)

    # Mostrar la predicción y la etiqueta real
    print("Predicción: " + nombres_clases[np.argmax(prediccion[0])])
    print("Etiqueta real: " + nombres_clases[etiqueta.numpy()])

    # Mostrar la imagen
    imagen = imagen.numpy().reshape((28, 28))
    plt.figure()
    plt.imshow(imagen, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()
