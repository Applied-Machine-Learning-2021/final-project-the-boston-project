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

Machine Learning Advisor:
- [Christian Powell](http://github.com/cdpowell) - `University of Kentucky`

Other Advisors:
- [Dr.Elena Sesma](https://anthropology.as.uky.edu/users/ese230) - `University of Kentucky`
- [Dr. Suzanne Smith](https://www.engr.uky.edu/directory/smith-suzanne) - `University of Kentucky`

## Our Project

Our project was originally started by an Dr.Sesma, doctor of Anthropology  and  Assistant Professor at the University of Kentucky. which specializations in African Diaspora and historical archaeology. For those who are not aware what archelogy is, it the study of human history and prehistory through the excavation of sites and the analysis of artifacts which include but are not limited to  maps and diaries , photographs or pictures and , and oral histories along  with physical remains from the deceased. One of the most common difficulties about archeology is that is  a destructive science. Once an  area is excavated  (where earth is moved  carefully and systematically from (an area) in order to find buried remains.), we can’t put things back the way we found them. Many of the  sites have been connected to marginalized communities, especially the African-American communities, which have been historically been destroyed in the process of countless development, expansion projects, or even for quests of truth. This eventually leads to question  "How  we can best mitigate the damage archeologists can do especially when trying to dig into the past.

In efforts to solve this problem our time took up the task of  building  a Convolutional Neural Network (CNN) that determines whether or not an area of land contained a cemetery or not so archeologists can plan properly and mitigate damage through escavation. We used TensorFlow and to train our model with a sigmoid activation on the last layer of the model, and a 3 x 3 kenel_size. 


## Usage instructions
Upload CNN_.ipynb file into Google colab

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




