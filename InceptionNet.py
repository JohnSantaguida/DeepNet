import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import inception
from PIL import Image

#print(tf.__version__)
inception.data_dir= 'data/inception/'
inception.maybe_download()

model=inception.Inception()

def classify(image_path):
    # Display the image.
    img = Image.open(image_path)
    img.show()
    # Use the Inception model to classify the image.
    pred = model.classify(image_path=image_path)

    # Print the scores and names for the top-10 predictions.
    model.print_scores(pred=pred, k=10, only_first_name=True)

image_path = ( 'Shark/shark_1.jpg')
print ("-----------------------------------------------------")
print(classify(image_path))
