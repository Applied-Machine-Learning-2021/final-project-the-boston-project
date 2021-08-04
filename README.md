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
Upload CNN_(drew).ipynb file into Google colab

For this line to work properly you need to have aerial images uploded into your Google drive 
```
drive.mount('/content/drive')
```


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
```
## Pictures 
![image](https://user-images.githubusercontent.com/85504234/128245403-c96a3a25-3e8e-44c3-bd93-d5cb93bfc191.png)             ![image](https://user-images.githubusercontent.com/85504234/128245615-cd516118-32a6-4cbd-a03d-1e51faa1a029.png)      ![image](https://user-images.githubusercontent.com/85504234/128246305-da7b3e9b-d655-4ba4-b52d-677110a02ad1.png)    ![image](https://user-images.githubusercontent.com/85504234/128261489-badd28f0-dca3-46c6-a73b-a86bd1027fa5.png)




