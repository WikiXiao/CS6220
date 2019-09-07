from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

np.random.seed(5)
batch_size = 32
num_classes = 10
epochs = 100
num_predictions = 20
train_size = 40000
val_size = 10000
train_true_size = 7000
val_true_size = 3000
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'colorRich_10x_model.h5'
file_name ='colorRich_10x.txt'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_val = x_train[train_size:]
x_train = x_train[:train_size]
y_val = y_train[train_size:]
y_train = y_train[:train_size]

train_rand_idx = np.random.choice(train_size,train_true_size)
val_rand_idx = np.random.choice(val_size,val_true_size)

x_train = x_train[train_rand_idx]
x_val = x_val[val_rand_idx]
y_train = y_train[train_rand_idx]
y_val = y_val[val_rand_idx]

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:]==i)[0]
    x_idx = x_train[idx,::]
    img_num = np.random.randint(x_idx.shape[0])
    im = np.transpose(x_idx[img_num,::], (0, 1, 2))
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

x_train = x_train.astype('float32')/255
x_val = x_val.astype('float32')/255
x_test = x_test.astype('float32')/255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val,num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model_info = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_val, y_val),
          shuffle=True)


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

fig, axs = plt.subplots(1,2,figsize=(15,5))
axs[0].plot(range(1,len(model_info.history['acc'])+1),model_info.history['acc'])
axs[0].plot(range(1,len(model_info.history['val_acc'])+1),model_info.history['val_acc'])
axs[0].set_title('Model Accuracy')
axs[0].set_ylabel('Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_xticks(np.arange(1,len(model_info.history['acc'])+1),len(model_info.history['acc'])/10)
axs[0].legend(['train', 'val'], loc='best')

axs[1].plot(range(1,len(model_info.history['loss'])+1),model_info.history['loss'])
axs[1].plot(range(1,len(model_info.history['val_loss'])+1),model_info.history['val_loss'])
axs[1].set_title('Model Loss')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_xticks(np.arange(1,len(model_info.history['loss'])+1),len(model_info.history['loss'])/10)
axs[1].legend(['train', 'val'], loc='best')
plt.show()

with open(file_name, 'wb') as file_pi:
  pickle.dump(model_info.history, file_pi)