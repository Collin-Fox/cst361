import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the data
housing = pd.read_csv("housing.csv")

# Split the data into Labels and Features
features = pd.DataFrame(housing)
labels = features.pop(features.columns[9])

# Encode the labels
le = LabelEncoder()
labels = pd.DataFrame(le.fit_transform(labels))

# Split the data into training, testing, and validation sets
total_rows = features.shape[0]
train_size = int(.8 * total_rows)
test_size = int((total_rows - train_size) / 2)
train_features, test_features, validation_features = features[:train_size], features[
                                                                            train_size: test_size + train_size], features[
                                                                                                                 test_size + train_size:]

train_labels, test_labels, validation_labels = labels[:train_size], labels[train_size: test_size + train_size], labels[
                                                                                                                test_size + train_size:]

# Convert Features into float 32 type
train_features = train_features.astype(np.float32)
test_features = test_features.astype(np.float32)
validation_features = validation_features.astype(np.float32)

# Build the model
model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_features.keys())]),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(5, activation=tf.nn.softmax)
])

# Compile the model
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss= keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])



# Train the model
history = model.fit(train_features, train_labels, epochs=10)

print(model.evaluate(test_features, test_labels))


