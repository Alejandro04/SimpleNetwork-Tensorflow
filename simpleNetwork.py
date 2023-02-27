import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-30, -10, -5, 0, 15, 25, 35], dtype=float)
fahrenheit = np.array([-22, 14, 23, 32, 59, 77, 95], dtype=float)

layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comienza entrenamiento..")
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo entrenado!")

plt.xlabel("# Vueltas")
plt.ylabel("Magnitud de pérdida")
plt.plot(history.history["loss"])

print("Prección 1")
result = model.predict([80])
print("El resultado es" + str(result) + " fahrenheit.")