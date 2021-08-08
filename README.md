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

In efforts to solve this problem our team took up the task of  building  a Convolutional Neural Network (CNN) that determines whether or not an area of land contained a cemetery or not so archeologists can plan properly and mitigate damage through escavation. We used TensorFlow and to train our model with a sigmoid activation on the last layer of the model, and a 3 x 3 kenel_size. 


## Usage instructions
Upload CNN_.ipynb file into Google colab

For this line to work properly you need to have aerial images uploded into your Google drive 
```
drive.mount('/content/drive')
```
## Uses Keras DirectoryIterator
This function uses the keras open library to form a variable to train our CNN model. This chooses the specified directory that will eventually train the model to make predictions. 

```
train_image_iterator = tf.keras.preprocessing.image.DirectoryIterator(
    target_size=(100, 100),
    directory=train_dir,
    batch_size=128,
    image_data_generator=None)
```

## CNN features
These are the features that used to construct the CNN. We chose to use a 3x3 kernal size and sigmoid activation on the layer because we are dealing with binary classification.
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
])
```

## Model features
Compiling and training the model. 100 epochs are used to train based on the file size.
```
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_image_iterator,
    epochs=100,
    shuffle=True
)
```
## Exploratory Data Analysis \ Main_(LiDAR_and_point_cloud_testing)
Before we decided to use the images to train our model, we planned on using the lidar data we gathered from the kyfromabove website. downloaded one of the zipfile from the the site and uncompressed the .laz file from it into a .txt file using laszip.exe. As we were looking at the list of coordinates we took note that they were not ordered by x or y and were randomly placed. To fix this issue so that we could feed the data to our model we tried to create a structured array of x’s and y’s that were evenly spaced between the x and y minimum and maximum. then we would take all of the data points and place them into their corresponding bins and find the average z value of all the points that belonged to each bin.

First we uploaded one of the .txt files into Google Colab and created a function that would open the file and placed the x y and z coordinates it contained into a pandas dataframe. 
```
def open_lidar(filename, verbose=False):
    """Method for opening LiDAR text files and handling possible line errors
    """
    # open file and read in each line
    with open(filename, "r") as f:
        lines = f.readlines()

    # iterate through lines to cast to lists of floats
    new_lines = list()
    for line in lines:

        # in case file is corrupt 
        try:
            new_lines.append([float(val) for val in line.split()])

        except Exception as e:
            if verbose:  # printing only if verbose, ignore otherwise
                print(line)

    # convert nested list to pandas dataframe
    new_lines = pd.DataFrame(new_lines)
    new_lines = new_lines.rename(columns={0: "x", 1: "y", 2: "z"})

    return new_lines
```
Then we calculated the maximum, minimum and scale for the x y and z values and placed these statistics into a dataframe. The scales were calculated by subtracting the minimum from the maximum of the x, y and z values.

```
#calculated the max values, the min, the scale(max-min) and the median value 
max = lidar_df.max() 
min = lidar_df.min()

# organized the data into a list to be converted into a dataframe  
lidar_stats = pd.DataFrame(
    [min, max, max - min],
    index=["min", "max", "scale"]
)
```
Next were created a dataframe with columns that were numbers evenly space between the minimum nad maximum x and indecies that were numbers evenly space between the minimum and maximum y.
```
side = 2048
increment = side / lidar_stats.loc["scale"]
point_grid = pd.DataFrame(
    index=[y for y in np.arange(lidar_stats["y"]["min"], lidar_stats["y"]["max"], increment["y"])],
    columns=[x for x in np.arange(lidar_stats["x"]["min"], lidar_stats["x"]["max"], increment["x"])],
)
```
lastly we created a function that would take in an x, y and list of nearby coordinated and calculated the average z values with the x and y of each 3d coordinated in the list acting as a weight. Then we used a for-loop to travese the dataframe and gather all of the lidar points that would belong in that cell.
```
#This function will help form a LiDAR grid with a proper set of consistent points
def calc_elevation(x, y, nearby_data_points):
  """Helper method for calculating the elevation based off nearby points.
  """

  # if not nearby_data_points:
  #   return None

  distances = [((val[0] - x)**2 + (val[1] - y)**2)**0.5 for val in nearby_data_points]
  total_distance = len(distances)
  return sum([val[2] * (distances[index]/total_distance) for index, val in enumerate(nearby_data_points)])


gridded_df = pd.DataFrame(columns=np.arange(lidar_stats["x"]["min"], lidar_stats["x"]["max"], increment["x"]))

for x in np.arange(lidar_stats["x"]["min"], lidar_stats["x"]["max"], increment["x"]):
  for y in np.arange(lidar_stats["y"]["min"], lidar_stats["y"]["max"], increment["y"]):
    values = lidar_df[
                  
                      (lidar_df["x"] >= x) & \
                      (lidar_df["x"] <= (x + increment["x"])) & \
                      (lidar_df["y"] >= y) & \
                      (lidar_df["y"] <= (y + increment["y"]))
                      ]
    delete_indexes = values.index
    lidar_df.drop(labels=delete_indexes, axis=0, inplace=True)

    
    gridded_df[x][y] = calc_elevation(x, y, values.values)

    
```

This code was not very efficient and took too long to run thus we came up with another way of grouping and analyzing the dat

## Pictures 
![image](https://user-images.githubusercontent.com/85504234/128245403-c96a3a25-3e8e-44c3-bd93-d5cb93bfc191.png)             ![image](https://user-images.githubusercontent.com/85504234/128245615-cd516118-32a6-4cbd-a03d-1e51faa1a029.png)      ![image](https://user-images.githubusercontent.com/85504234/128246305-da7b3e9b-d655-4ba4-b52d-677110a02ad1.png)    ![image](https://user-images.githubusercontent.com/85504234/128261489-badd28f0-dca3-46c6-a73b-a86bd1027fa5.png)




