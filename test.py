import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model

model=load_model('./saved_models/model.h5')

test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

test_generator=test_datagen.flow_from_directory('./test/', # this is where you specify the path to the main data folder
                                                 target_size=(160,160),
                                                 color_mode='rgb',
                                                 batch_size=100,
                                                 class_mode='categorical',
                                                 shuffle=True)

# Score trained model.
scoreSeg = model.evaluate_generator(test_generator, 400)
print("Accuracy = ",scoreSeg[1])
