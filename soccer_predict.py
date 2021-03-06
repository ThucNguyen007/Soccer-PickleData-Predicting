# Thuc Nguyen

'''

the data are now in the same form after flattening their exemplars into pairs of row vectors.  
The  inputs  are  22-element  binary  vectors.  
Each  element  represents one of the 22 players, 
with a 1 indicating that the player participated in a given match and a 0 meaning that he did not.

The  networks  use  sigmoid  activation  functions  in  the  output  layer. 
The  network  for additionally  contains a hidden layer of 100 ReLU neurons.
After training each network for 200 epochs, 
we get the following standard deviations for 4-element vectors indicating:

Goals scored: 4.59 
•Goals conceded: 3.93
•Drinks: 13.06 liters
•Financial gain: $111.520

 the  output  variables  are  not  simply  weighted  sums  of  individual  players’  characteristics.  
 If  they  were,  a  neuron  without  hidden  layers  could  do  a  perfect  job,  except  for  any  random noise in the data. 
 However, given that different team configurations exert additional, 
 non-linear effects on the output variables, we need at least one additional layer with non-linear activation function to closely learn the desired function.

'''

import pickle
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

# build the neural network model
num_inputs = 22
num_outputs = 4
batch_size = 100
num_epochs = 200
num_train_exemplars = 4000
num_test_exemplars = 1000
input_shape = (num_inputs,)

with (open("soccer_data.pickle", "rb")) as f:
    while True:
        try:
            (x_train, y_train), (x_test, y_test) = pickle.load(f)
        except EOFError:
            break

# Show the 10 first training example
print(f"Training sample:\n{x_train[:10]}\n")
print(f"Training label:\n{y_train[:10]}\n")

# Show the 10 first testing example
print(f"Testing sample:\n{x_test[:10]}\n")
print(f"Testing label:\n{y_test[:10]}\n")

# training and testing data between 0 & 1 by dividing by the maximum
train_data_Xnorm = x_train / 255.0
test_data_Xnorm = x_test / 255.0

train_data_Ynorm = y_train / 255.0
test_data_Ynorm = y_test / 255.0

# Scaling outputs to be larger than 0 and less than 1
# (Not exactly 0 or 1, because we want to use the sigmoid function that can 
# only approach but never reach these values)
y_train_norm = (y_train - [-1, -1, 0, -1000000]) / [12, 12, 200, 3000000]
y_test_norm = (y_test - [-1, -1, 0, -1000000]) / [12, 12, 200, 3000000]

# Create the neural network model
create_model = tf.keras.Sequential()

create_model.add(tf.keras.layers.Dense(100, input_shape=input_shape, activation='relu', name='hidden'))
create_model.add(tf.keras.layers.Dense(num_outputs, activation='sigmoid', name='output'))
create_model.add(tf.keras.layers.Lambda(lambda x: x * 400))

# 2. Compile the model, this time with a regression-specific loss function
create_model.compile(loss=tf.keras.losses.mse, optimizer=tf.keras.optimizers.Adam(), metrics=['mse'])

# 3. Fit the model
history = create_model.fit(x_train, y_train_norm, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=(x_test, y_test_norm))
create_model.summary()

# Check the classification performance of the trained network on the test data
final_train_loss, final_train_accuracy = create_model.evaluate(x_train, y_train, verbose=0)
final_test_loss, final_test_accuracy = create_model.evaluate(x_test, y_test, verbose=0)

print('Final training loss (mean square error):', final_train_loss)
print('Final test loss (mean square error):', final_test_loss)

y_train_norm = (y_train + [-1, -1, 0, 1000000]) / [12, 12, 300, 3000000]
y_test_norm = (y_test + [-1, -1, 0, 1000000]) / [12, 12, 300, 3000000]

# Get the rescaled predictions for the test set and compute their deviation from the desired ones
y_predict_norm = create_model.predict(x_test)

y_predict = y_predict_norm * [12, 12, 300, 3000000] - [1, 1, 0, 1000000]
y_diff = y_predict - y_test
y_stdev = np.std(y_diff, axis=0)

print(y_stdev)

def mse(testX, testY):
    # Calculates mean squared error between y_test and y_preds.
    return tf.metrics.mean_squared_error(testX, testY)

# Functions generating random inputs (2D coordinates) for each class
# We are simply assuming normally distributed samples for each class
def get_example():
    return [random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)]

def displayScores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())
