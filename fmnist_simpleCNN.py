import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

print(x_train.shape[0], "x_train")
print(x_test.shape[0], "x_test")

# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9
                        
# Data normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Further break training data into train / validation sets 
# Put 5000 into validation set and keep remaining 55 000 for train

(x_valid, x_train) = x_train[:5000], x_train[5000:]
(y_valid, y_train) = y_train[:5000], y_train[5000:]

# Reshape input to 28,28,1
w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential([
    Conv2D(num_filters, filter_size, input_shape = (28,28,1)),
    MaxPooling2D(pool_size),
    Flatten(),
    Dense(10, activation = 'softmax'),
])

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

from tensorflow.keras.utils import to_categorical

model.fit(x_train,
          to_categorical(y_train),
          epochs=3,
          validation_data=(x_valid, to_categorical(y_valid)))
          
# Predict model on first 5 images

predictions = model.predict (x_test[:5])

print(np.argmax(predictions, axis=1))

print(y_test[:5])

# Add one more conv layer

model = Sequential([
                    Conv2D(num_filters, filter_size, input_shape=(28,28,1)),
                    Conv2D(num_filters, filter_size),
                    MaxPooling2D(pool_size),
                    Flatten(),
                    Dense(10, activation='softmax'),
])

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(x_train,
          to_categorical(y_train),
          epochs=3,
          validation_data=(x_valid, to_categorical(y_valid))
          )
          
#dropout layers to protect from ovefitting
from tensorflow.keras.layers import Dropout

model = Sequential([
                    Conv2D(num_filters, filter_size, input_shape=(28,28,1)),
                    MaxPooling2D(pool_size),
                    Dropout(0.5),
                    Flatten(),
                    Dense(10, activation='softmax'),
])

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(x_train,
          to_categorical(y_train),
          epochs=3,
          validation_data=(x_valid, to_categorical(y_valid))
          )
          
# Fully-connected layers
model = Sequential([
                    Conv2D(num_filters, filter_size, input_shape=(28,28,1)),
                    MaxPooling2D(pool_size),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dense(10, activation='softmax'),
])

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(x_train,
          to_categorical(y_train),
          epochs=3,
          validation_data=(x_valid, to_categorical(y_valid))
          )
          
# Changing conv2d parameters
num_filters = 8
filter_size = 3

model = Sequential([
  Conv2D(
    num_filters,
    filter_size,
    input_shape=(28, 28, 1),
    strides=2,
    padding='same',
    activation='relu',
  ),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),
])

model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(x_train,
          to_categorical(y_train),
          epochs=3,
          validation_data=(x_valid, to_categorical(y_valid))
          )

# More complex CNN model 

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
             
# Saving the best weights
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)

model.fit(x_train,
         to_categorical(y_train),
         epochs=10,
         validation_data=(x_valid, to_categorical(y_valid)),
         callbacks=[checkpointer])
         
# Load the weights with the best validation accuracy
model.load_weights('model.weights.best.hdf5')

# Evaluate the model on test set
score = model.evaluate(x_test, to_categorical(y_test), verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])

#visualize prediction (first 15 examples)
y_hat = model.predict(x_test)

# Plot a random sample of 15 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index], 
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))