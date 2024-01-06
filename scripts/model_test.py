import numpy as np
from skimage.transform import resize
from skimage.io import imread
import matplotlib.pyplot as plt
import pickle

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))


def classification(img,model):
    Categories = ['Cup','Book','Box']
    img = imread(img)
    plt.imshow(img)
    plt.show()
    img_resize = resize(img, (150,150,3))
    l = np.array([img_resize.flatten()]).reshape(1,-1)
    probability = model.predict_proba(l)
    for ind, val in enumerate(Categories):
        print(f'{val} = {probability[0][ind]*100}%')
    print("The predicted image is : "+Categories[model.predict(l)[0]])
    return Categories[model.predict(l)[0]]


filename = 'classification_model.sav'
loaded_model = pickle.load(open('model/' + filename, 'rb'))
path_test = "data/cup_test1.jpeg"
classification(path_test, loaded_model)
