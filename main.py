import numpy as np
import pandas as pd
import tensorflow as tf


dataset = pd.read_excel('Combined_Cycle_Power_Plant.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=7, activation='relu'))
ann.add(tf.keras.layers.Dense(units=7, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))



ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
ann.fit(X_train, y_train, batch_size = 32, epochs = 150)
y_pred = ann.predict(X_test)
y_pred = np.reshape(y_pred , len(y_pred) )
print(y_pred)

print()
