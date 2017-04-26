from PIL import Image
import glob


FISH_NAMES=['Bass','Catfish','Eel','Flounder','Salmon','Shark','Trout','Tuna']
IMG_EXTNS = ['jpg','jpeg','gif','png']
image_list = []
for ex in IMG_EXTNS:
    for filename in glob.glob('./*/*.' + ex): #pull images into list
        im=Image.open(filename)
        image_list.append(im)
#print(image_list) -Debug

