
# coding: utf-8

# In[3]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Dense
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import cv2
import scipy
import os
#%matplotlib inline
import matplotlib.pyplot as pet


# In[3]:


epochs = 10
BASE_DIR = 'C:/Users/ahowe/Desktop/VAC/' #'../'
batch_size = 100


# In[4]:


from keras.preprocessing import image
def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []

    for seismic_type in os.listdir(folder):
        if not seismic_type.startswith('.'):
            if seismic_type in ['Class1']:
                label = '0'
            else:
                label = '1'
            for image_filename in os.listdir(folder + seismic_type):
                img_file = cv2.imread(folder + seismic_type + '/' + image_filename)
                if img_file is not None:
                    # Downsample the image to 120, 160, 3
                    #img_file = scipy.misc.imresize(arr=img_file, size=(120, 160, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y


# In[1]:


def get_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(150, 300, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


# In[6]:


from IPython.display import display, Image
display(Image(filename='C:/Users/ahowe/Desktop/VAC/images/Test/Class1/10.jpg'))
from PIL import Image

im = Image.open('C:/Users/ahowe/Desktop/VAC/images/Test/Class1/10.jpg')
width, height = im.size


# In[7]:


X_train, y_train = get_data(BASE_DIR + 'images/Train/')
X_test, y_test = get_data(BASE_DIR + 'images/Test/')
X_train = X_train*1./255.
X_test = X_test*1./255.
encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)


# In[ ]:


X_train.shape
y_train.shape


# In[ ]:


model = get_model()
history = model.fit(X_train,y_train,validation_split=0.2,epochs=epochs,shuffle=True,batch_size=batch_size)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score

print('Predicting on test data')
y_pred = np.rint(model.predict(X_test))
print(accuracy_score(y_test,y_pred))


# In[ ]:


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))


# In[ ]:


model.save(BASE_DIR+'images/OneConv2V2.h5')


# In[4]:


#model = get_model()
#model.load_weights('C:/Users/ahowe/Desktop/VAC/images/OneConv2V1.h5')


# In[7]:


from ann_visualizer.visualize import ann_viz;

ann_viz(model, title="One Convolution")

