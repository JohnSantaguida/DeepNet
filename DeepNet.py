#RUNNER CLASS FOR OUR NETOWORK
#see https://github.com/llSourcell/tensorflow_image_classifier/blob/master/src/label_image.py for help

# CAN ALL BE DELETED  - RESTARTING

from PIL import Image, ImageOps
import numpy as np
import glob

FISH_NAMES=['Bass','Catfish','Eel','Flounder','Salmon','Shark','Trout','Tuna']
IMG_EXTNS = ['jpg','jpeg','gif','png']
image_list = []
image_labels = []
for ex in IMG_EXTNS:
    for fish in FISH_NAMES:
        for filename in glob.glob('./' + fish + '/*.' + ex): #pull images into list
            im=Image.open(filename).convert('LA')
            im.thumbnail((28,28), Image.ANTIALIAS)
            image_labels.append(fish)
            image_list.append(im)
print(image_list)
print(len(image_labels))