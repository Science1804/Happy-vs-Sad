import os,zipfile
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG
from google.colab import files
from keras.preprocessing import image

!git clone https://github.com/Science1804/Happy-vs-Sad.git
os.listdir('Happy-vs-Sad')
myzip = 'Happy-vs-Sad/happy-or-sad.zip'

zip_read = zipfile.ZipFile(myzip,'r')
zip_read.extractall('happy-or-sad')#Give a Folder name where to extract
zip_read.close()

desired_accuracy = 0.85

class myCallbacks(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    if (logs.get('acc')>desired_accuracy):
      print('\n Reached 85% accuracy so Cancelled the Training')
      self.model.stop_training = True
      
callbacks = myCallbacks()
model = tf.keras.Sequential([
                             tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                             tf.keras.layers.MaxPooling2D(2,2),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(256,activation='relu'),
                             tf.keras.layers.Dense(1,activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
train_datagen = IDG(rescale=1/255)
train_generator = train_datagen.flow_from_directory('happy-or-sad',target_size=(150,150),batch_size=5,class_mode='binary')
history1 = model.fit_generator(train_generator,steps_per_epoch=8,epochs=15,verbose=1,callbacks=[callbacks]) 
# For plotting Accuracy and loss
def plot_graphs(history,string):
  import matplotlib.pyplot as plt
  %matplotlib inline
  plt.plot(history.history[string])
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.show()

plot_graphs(history1,'acc')#plots accuracy
plot_graphs(history1,'loss')#plots loss

#Try uploading an emoticon yourself 
#procedure made suitable to follow in google Colab
uploaded = files.upload()
for fn in uploaded.keys():
  path = '/content/' + fn
  img = image.load_img(path,target_size=(150,150))
  x = image.img_to_array(img)
  x = np.expand_dims(x,axis=0)
  myimages = np.vstack([x])
  classes = model.predict(myimages,batch_size=10)
  if classes[0] > 0.7 :
    print(fn + " is a Happy emoticon")
  else :
    print(fn + ' is a Sad emoticon')  
