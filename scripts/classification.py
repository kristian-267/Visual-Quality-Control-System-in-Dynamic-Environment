import numpy as np
from sklearn import svm
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pickle

import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))


# Load images
Categories=['Cup','Book','Box']
train_path = "data/train"
test_path = "data/test"
flat_data_arr=[] 
target_arr=[]
for i in Categories:   
    print(f'loading... category : {i}')
    path=os.path.join(train_path,i)
    for img in os.listdir(path):
        if img != '.DS_Store':           
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(150,150,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
    print(f'loaded category:{i} successfully')
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data) #dataframe
df['Target']=target
x=df.iloc[:,:-1] #input data 
y=df.iloc[:,-1] #output data

data = x #principalComponents

# Train SVM model
param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
svc=svm.SVC(probability=True)
model=GridSearchCV(svc,param_grid)

x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.20,random_state=77,stratify=y)
print('Splitted Successfully')
model.fit(x_train,y_train)
print('The Model is trained well with the given images')

y_pred=model.predict(x_test)
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

# Save model
filename = 'classification_model.sav'
pickle.dump(model, open('model/' + filename, 'wb'))
