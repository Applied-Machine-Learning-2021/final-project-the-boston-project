#Mounting 
from google.colab import drive
import os
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

drive.mount('/content/drive')

cemetery_boxes_dir = "/content/drive/Shareddrives/UAVs n' stuff/Data Files"
cemetary_zipfiles = os.listdir(cemetery_boxes_dir)

zipfile_names = [cemetery_boxes_dir + x for x in cemetary_zipfiles]
zipfile_names

zipfile.ZipFile("/content/drive/Shareddrives/UAVs n' stuff/Data Files/UAV_lab.zip").extractall()

#Building the CNN
#Loading in the data for the CNN
train_dir = "/content/UAV_lab"
train_categories = set(os.listdir(train_dir))
test_dir = "/content/test"
test_categories = set(os.listdir(test_dir))

if train_categories.symmetric_difference(test_categories):
  print("Warning!: ", train_categories.symmetric_difference(test_categories))

print(sorted(train_categories))
print(len(train_categories))
print(sorted(test_categories))
print(len(test_categories))

train_dir = "/content/UAV_lab"

datagen = ImageDataGenerator()

train_image_iterator = tf.keras.preprocessing.image.DirectoryIterator(
    target_size=(100, 100),
    directory=train_dir,
    batch_size=128,
    image_data_generator=None)


#Fitting the model with specified layers and padding 
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
                           input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_image_iterator,
    epochs=100,
    shuffle=True
)

#Visualizing the accuracy and loss of the Model
plt.plot(list(range(len(history.history['accuracy']))),
         history.history['accuracy'])
plt.show()

plt.plot(list(range(len(history.history['loss']))), history.history['loss'])
plt.show()

#Evaluating the model
test_dir = "/content/test"

test_image_iterator = tf.keras.preprocessing.image.DirectoryIterator(
    target_size=(100, 100),
    directory=test_dir,
    batch_size=128,
    shuffle=False,
    image_data_generator=None)

model.evaluate(test_image_iterator)

#Making predictions on the model
#Initializing the actual classes based on the test image iterator
predictions = model.predict(test_image_iterator)
actual_classes = test_image_iterator.classes

#Setting appending the predictions into a list
predicted_class = [np.argmax(p) for p in predictions]
predicted_class

#calcualting the F1_score for the model
f1_score(actual_classes, predicted_class, average='micro')
