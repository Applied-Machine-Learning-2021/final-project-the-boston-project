<!--
Name of your teams' final project
-->
# final-project
## [National Action Council for Minorities in Engineering(NACME)](https://www.nacme.org) Google Applied Machine Learning Intensive (AMLI) at the `University of Kentucky`

<!--
List all of the members who developed the project and
link to each members respective GitHub profile
-->
Developed by: 
- [Drew Butler](https://github.com/drewbutler) - `University of Central Florida`
- [Victoria Gomez-Small](https://github.com/Via104) - `Northeastern University` 
- [Jordan Ogbu Felix](https://github.com/JordanOgbuFelix) - `University of California Davis`

![image](https://user-images.githubusercontent.com/85504234/127721641-08e9fdec-52be-4f4e-9c95-e491652bda30.png)



TA:
- [Christian Powell](http://github.com/cdpowell) - `University of Kentucky`

## Description

The goal of our project was to build a CNN that determines whether or not an area of land contained a cemetery or not. We used TensorFlow and to train our model with a sigmoid activation on the last layer of the model, and a 3 x 3 kenel_size. 


## Usage instructions

## CNN features

```
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
])```

1. Fork this repo
2. Change directories into your project
3. On the command line, type `pip3 install requirements.txt`
4. ....
