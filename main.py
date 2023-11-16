import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.legacy import Adam
from sklearn.metrics import accuracy_score
import random

np.random.seed(22)

ddir = './input_ed'
tr_p = './input_ed/Train'
te_p = './input_ed'

im_h = 30
im_w = 30
chans = 3

n_categories = len(os.listdir(tr_p))

SIGNS = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

flds = os.listdir(tr_p)

n_training = []
n_classes = []

for folder in flds:
    f_train = os.listdir(tr_p + '/' + folder)
    n_training.append(len(train_files))
    n_classes.append(SIGNS[int(folder)])

lists_all = zip(n_training, n_classes)
pairs_s = sorted(lists_all)
tuples = zip(*pairs_s)
n_training, n_classes = [ list(tuple) for tuple in  tuples]

test = pd.read_csv(ddir + '/Test.csv')
imgs = test["Path"].values

D_IMAGES = []
L_IMAGES = []

for i in range(n_categories):
    path = ddir + '/Train/' + str(i)
    images = os.listdir(path)

    for img in images:
        try:
            image = cv2.imread(path + '/' + img)
            image_fromarray = Image.fromarray(image, 'RGB')
            resize_image = image_fromarray.resize((im_h, im_w))
            D_IMAGES.append(np.array(resize_image))
            L_IMAGES.append(i)
        except:
            print("Error in " + img)

D_IMAGES = np.array(D_IMAGES)
L_IMAGES = np.array(L_IMAGES)

sh_ind = np.arange(D_IMAGES.shape[0])
np.random.shuffle(sh_ind)
D_IMAGES = D_IMAGES[sh_ind]
L_IMAGES = L_IMAGES[sh_ind]

TRAIN_X, VAL_X, TRAIN_Y, VAL_Y = train_test_split(D_IMAGES, L_IMAGES, test_size=0.3, random_state=42, shuffle=True)

TRAIN_X = TRAIN_X/255 
VAL_X = VAL_X/255

TRAIN_Y = keras.utils.to_categorical(TRAIN_Y, n_categories)
VAL_Y = keras.utils.to_categorical(VAL_Y, n_categories)

model = keras.models.Sequential([    
    keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', input_shape=(im_h,im_w,chans)),
    keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(axis=-1),
    
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(rate=0.5),
    
    keras.layers.Dense(43, activation='softmax')
])

lr = 0.001
epochs = 30

opt = Adam(lr=lr, decay=lr / (epochs * 0.5))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

model.fit(aug.flow(TRAIN_X, TRAIN_Y, batch_size=32), epochs=epochs, validation_data=(VAL_X, VAL_Y))

test = pd.read_csv(ddir + '/Test.csv')

labels = test["ClassId"].values
imgs = test["Path"].values

data =[]

for img in imgs:
    try:
        image = cv2.imread(ddir + '/' +img)
        image_fromarray = Image.fromarray(image, 'RGB')
        resize_image = image_fromarray.resize((im_h, im_w))
        data.append(np.array(resize_image))
    except:
        print("Error in " + img)

X_test = np.array(data)
X_test = X_test/255

model.save('model.h5')
predictions = model.predict(X_test)
pred = np.argmax(predictions, axis=1)

print('Test Data accuracy: ',accuracy_score(labels, pred)*100)
