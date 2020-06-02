import random
import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout, UpSampling2D, MaxPooling2D

def Show1(img1, img2):
    # вывод на экран картинки
    plt.subplot(1, 2, 1), plt.imshow(img1, 'gray')
    plt.title("img1")
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(img2, 'gray')
    plt.title("img2")
    plt.xticks([]), plt.yticks([])
    plt.show()
def main():
   print("hello")

   seed = 42
   np.random.seed=seed
   IMG_HEIGHT=128
   IMG_WIDTH=128
   #IMG_HEIGHT, IMG_WIDTH = 256, 256
   IMG_CHANNELS=3

   x_train=np.zeros((345,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),  dtype=np.uint8)
   y_train = np.zeros((345, IMG_HEIGHT, IMG_WIDTH), dtype=np.bool)
   #y_train = np.zeros((345, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
   #y_train = np.zeros((345, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
   #x_train = np.zeros((345, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

   print("Resize training images")
   for i in range(1, 346):
       img = cv2.imread(r"C:\Users\RTF\Desktop\TwoWork\29.02.20\start\3 ({id}).tif".format(id=i))
       #img = cv2.imread(r"C:\Users\RTF\Desktop\TwoWork\29.02.20\start\3 ({id}).tif".format(id=i), 0)
       #img= cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
       img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
       x_train[i-1]=img
       #mask = cv2.imread(r"C:\Users\RTF\Desktop\TwoWork\29.02.20\mask\3 ({id}).tif".format(id=i), 0)
       #mask=cv2.imread(r"C:\Users\RTF\Desktop\TwoWork\29.02.20\mask\3 ({id}).tif".format(id=i), 0)
       #mask=cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
       mask = cv2.imread(r"C:\Users\RTF\Desktop\TwoWork\29.02.20\mask\3 ({id}).tif".format(id=i), 0)
       mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
       y_train[i-1]=mask

   x_test =np.zeros((95,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),  dtype=np.uint8)
   #x_test = np.zeros((95, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
   sizes_test = []
   print("Resize test images")
   for i in range(1, 96):
       #img = cv2.imread(r"C:\Users\RTF\Desktop\TwoWork\29.02.20\start\3 ({id}).tif".format(id=i), 0)
       img = cv2.imread(r"C:\Users\RTF\Desktop\TwoWork\13.02.2020\start\2 ({id}).tif".format(id=i))
       sizes_test.append([img.shape[0], img.shape[1]])
       #img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
       img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
       x_test[i - 1] = img
   print("Done!")
   # Show1(x_train[10], y_train[10])
   # Show1(x_train[33], y_train[33])
   # Show1(x_train[1], y_train[1])

   #Build the model
   inputs = tf.keras.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

   s=tf.keras.layers.Lambda(lambda x: x/255)(inputs)
   c1=tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
   c1=tf.keras.layers.Dropout(0.1)(c1)
   c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
   p1=tf.keras.layers.MaxPooling2D((2,2))(c1)

   c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
   c2 = tf.keras.layers.Dropout(0.1)(c2)
   c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
   p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

   c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
   c3 = tf.keras.layers.Dropout(0.2)(c3)
   c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
   p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

   c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
   c4 = tf.keras.layers.Dropout(0.2)(c4)
   c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
   p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

   c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
   c5 = tf.keras.layers.Dropout(0.3)(c5)
   c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

   #expansive path
   u6=tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
   u6=tf.keras.layers.concatenate([u6,c4])
   c6=tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
   c6=tf.keras.layers.Dropout(0.2)(c6)
   c6=tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

   u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
   u7 = tf.keras.layers.concatenate([u7, c3])
   c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
   c7 = tf.keras.layers.Dropout(0.2)(c7)
   c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

   u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
   u8 = tf.keras.layers.concatenate([u8, c2])
   c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
   c8 = tf.keras.layers.Dropout(0.1)(c8)
   c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

   u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
   u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
   c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
   c9 = tf.keras.layers.Dropout(0.1)(c9)
   c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

   outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

   model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   model.summary()

   ##############################################
   #Modelcheckpoint
   checkpointer= tf.keras.callbacks.ModelCheckpoint('my_model.h5', verbose=1, save_best_only=True)

   callbacks=[
       tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
       tf.keras.callbacks.TensorBoard(log_dir='logs'),
       checkpointer
   ]

   results=model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)
   #results = model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=1, callbacks=callbacks)

   ###############################

   idx=random.randint(0, len(x_train))

   preds_train = model.predict(x_train[:int(x_train.shape[0]*0.9)], verbose=1)
   preds_val = model.predict(x_train[int(x_train.shape[0]*0.9):], verbose=1)
   preds_test=model.predict(x_test, verbose=1)

   preds_train_t=(preds_train>0.5).astype(np.uint8)
   preds_val_t=(preds_val>0.5).astype(np.uint8)
   preds_test_t=(preds_test>0.5).astype(np.uint8)

   #Perform a sanity check on same random training samples
   ix=random.randint(0, len(preds_train_t))

   plt.imshow(x_train[ix])
   plt.show()
   plt.imshow(y_train[ix])
   plt.show()
   #plt.imshow(preds_train_t[ix])
   plt.imshow(preds_train_t[ix][:, :, 0])
   plt.show()

   # Perform a sanity check on same random validation samples
   ix=random.randint(0, len(preds_val_t))

   plt.imshow(x_train[int(x_train.shape[0]*0.9):][ix])
   plt.show()
   plt.imshow(y_train[int(y_train.shape[0]*0.9):][ix])
   plt.show()
   plt.imshow(preds_val_t[ix][:, :, 0])
   plt.show()

   model.save('my_result2.h5')

if __name__ == "__main__":
    main()