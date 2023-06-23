import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

train.head()
X_train = train.drop(['label'], axis=1).values.reshape([60000, 28, 28, 1])
y_train = train['label'].values.reshape([60000, 1])

X_test = test.drop(['label'], axis=1).values.reshape([10000, 28, 28, 1])
y_test = test['label'].values.reshape([10000, 1])

y_train = to_categorical(y_train, num_classes=10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

X_train = X_train/255
X_val = X_val/255
X_test = X_test/255
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Conv2D(128, (3,3), padding = 'same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Conv2D(128, (3,3), padding = 'same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.6, 
                                            min_lr=0.00001)

batch_size = 64

train_datagen = ImageDataGenerator( 
        rotation_range=10,  
        zoom_range = 0.2, 
        width_shift_range=0.1,  
        height_shift_range=0.1,
        shear_range = 0.1,
        horizontal_flip=False,  
        vertical_flip=False
        )
train_datagen.fit(X_train)

history = model.fit(
            train_datagen.flow(X_train,y_train,batch_size = batch_size),
            epochs = 25,
            batch_size = batch_size,
            validation_data = (X_val,y_val),
            steps_per_epoch = X_train.shape[0]//batch_size,
            verbose = 1,
            callbacks=[learning_rate_reduction]
            )
preds = model.predict(X_test)
pred = []

for i in range(len(preds)):
    pred.append(list(preds[i]).index(max(preds[i])))
    
print(f"Accuracy: {accuracy_score(y_test, pred)*100}%")
