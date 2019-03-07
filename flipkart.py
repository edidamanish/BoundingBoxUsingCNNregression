import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tqdm
import random
import pandas as pd
from PIL import Image
import pickle
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ELU
from keras.layers import AveragePooling2D, MaxPooling2D, Input, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau


### Change the image directory here ###

image_dir = 'images/'


### Change the location of the training.csv here ###
train_csv = pd.read_csv('training.csv')
### Change test.csv directory ###
test_csv = pd.read_csv('test.csv')

train_csv.sample(frac=1, random_state = 1).reset_index(drop=True) #To shuffle the images



import random
def image_generator(batch_size = 128):
    indexes = random.sample(range(23000), 23000)
  
    i=0
    while True:
        batch_image = []
        batch_box = []
        
        for b in range(batch_size):
            if(i == len(indexes)):
                i = 0
                
                indexes = random.sample(range(23000), 23000)
        
            path = image_dir + train_csv['image_name'][indexes[i]]
            image=Image.open(path).convert('RGB')
            x1 = train_csv['x1'][indexes[i]]/640 * 224
            x2 = train_csv['x2'][indexes[i]]/640 * 224
            y1 = train_csv['y1'][indexes[i]]/480 * 224
            y2 = train_csv['y2'][indexes[i]]/480 * 224
            rnd = random.randint(0,4)
            
            if(rnd%5 == 0):
              
                image=image.resize((224,224))
                image=np.array(image,dtype=np.float32)
                image=image/255
                batch_image.append(image)
                _box = np.array([x1, y1, x2 ,y2])
                batch_box.append(_box)

            ### Various Data Augmentation configurations. ###
            elif (rnd%5 ==1):
                image=image.resize((224,224))
                image = image.rotate(90, expand = True)
                image=np.array(image,dtype=np.float32)
                image=image/255
                batch_image.append(image)
                _box = np.array([y1, 224-x2,y2, 224-x1])
                batch_box.append(_box)
            elif (rnd%5 == 2):
                image=image.resize((224,224))
                image = image.rotate(-90, expand = True)
                image=np.array(image,dtype=np.float32)
                image=image/255
                batch_image.append(image)
                _box = np.array([224-y2, x1, 224 -y1, x2])
                batch_box.append(_box)
            elif (rnd%5 == 3):
                image=image.resize((224,224))
                image = image.rotate(180, expand = True)
                image=np.array(image,dtype=np.float32)
                image=image/255
                batch_image.append(image)
                _box = np.array([224-x2, 224 - y2, 224 -x1, 224-y1])
                batch_box.append(_box)
            elif (rnd%5 == 4):
                image=image.resize((224,224))
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                image=np.array(image,dtype=np.float32)
                image=image/255
                
                batch_image.append(image)
                _box = np.array([224-x1, y1, 224 -x2, y2])
                batch_box.append(_box)
                
                
            i+=1
        yield(np.array(batch_image), np.array(batch_box))






def getvaldata():
    data_val = np.empty((1000, 224, 224, 3))

    for i in tqdm.tqdm(range( 23000, 24000)):
        id = i;
        path = image_dir + train_csv['image_name'][i]
        image=Image.open(path).convert('RGB')

        image=image.resize((224,224))
        image=np.array(image,dtype=np.float32)
        image=image/255

        data_val[i-23000] = image
    
    box_val = np.empty((1000, 4,))


    for i in tqdm.tqdm(range(23000, 24000)) :

        id =i
        x1 = train_csv['x1'][i]/640 * 224
        x2 = train_csv['x2'][i]/640 * 224
        y1 = train_csv['y1'][i]/480 * 224
        y2 = train_csv['y2'][i]/480 * 224
        _box = np.array([x1, y1, x2 ,y2])
        box_val[i-23000] = _box
    
    return data_val,box_val 
data_test,box_test = getvaldata()

def my_metric(labels,predictions):
    threshhold=0.9 
    x=predictions[:,0]*224
    x=tf.maximum(tf.minimum(x,224.0),0.0)
    y=predictions[:,1]*224
    y=tf.maximum(tf.minimum(y,224.0),0.0)
    width=predictions[:,2]*224
    width=tf.maximum(tf.minimum(width,224.0),0.0)
    height=predictions[:,3]*224
    height=tf.maximum(tf.minimum(height,224.0),0.0)
    label_x=labels[:,0]
    label_y=labels[:,1]
    label_width=labels[:,2]
    label_height=labels[:,3]
    a1=tf.multiply(width,height)
    a2=tf.multiply(label_width,label_height)
    x1=tf.maximum(x,label_x)
    y1=tf.maximum(y,label_y)
    x2=tf.minimum(x+width,label_x+label_width)
    y2=tf.minimum(y+height,label_y+label_height)
    IoU=tf.abs(tf.multiply((x1-x2),(y1-y2)))/(a1+a2-tf.abs(tf.multiply((x1-x2),(y1-y2))))
    condition=tf.less(threshhold,IoU)
    sum=tf.where(condition,tf.ones(tf.shape(condition)),tf.zeros(tf.shape(condition)))
    return tf.reduce_mean(sum)

def smooth_l1_loss(true_box,pred_box):
    loss=0.0
    for i in range(4):
        residual=tf.abs(true_box[:,i]-pred_box[:,i]*224)
        condition=tf.less(residual,1.0)
        small_res=0.5*tf.square(residual)
        large_res=residual-0.5
        loss=loss+tf.where(condition,small_res,large_res)
    return tf.reduce_mean(loss)

def resnet_block(inputs,num_filters,kernel_size,strides,activation='relu'):
    x=Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(inputs)
    x=BatchNormalization()(x)
    if(activation):
        x=Activation('relu')(x)
    return x

def resnet18():
    inputs=Input((224,224,3))
    
    # conv1
    x=resnet_block(inputs,64,[7,7],2)

    # conv2
    x=MaxPooling2D([3,3],2,'same')(x)
    for i in range(2):
        a=resnet_block(x,64,[3,3],1)
        b=resnet_block(a,64,[3,3],1,activation=None)
        x=keras.layers.add([x,b])
        x=Activation('relu')(x)
    
    # conv3
    a=resnet_block(x,128,[1,1],2)
    b=resnet_block(a,128,[3,3],1,activation=None)
    x=Conv2D(128,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    a=resnet_block(x,128,[3,3],1)
    b=resnet_block(a,128,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)
    x = Dropout(0.2)(x)

    # conv4
    a=resnet_block(x,256,[1,1],2)
    b=resnet_block(a,256,[3,3],1,activation=None)
    x=Conv2D(256,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)

    a=resnet_block(x,256,[3,3],1)
    b=resnet_block(a,256,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)
    x = Dropout(0.2)(x)

    # conv5
    a=resnet_block(x,512,[1,1],2)
    b=resnet_block(a,512,[3,3],1,activation=None)
    x=Conv2D(512,kernel_size=[1,1],strides=2,padding='same',kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(x)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)
    

    a=resnet_block(x,512,[3,3],1)
    b=resnet_block(a,512,[3,3],1,activation=None)
    x=keras.layers.add([x,b])
    x=Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x=AveragePooling2D(pool_size=7,data_format="channels_last")(x)
    # out:1*1*512

    y=Flatten()(x)
    # out:512
    y=Dense(1000,kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(y)
    outputs=Dense(4,kernel_initializer='he_normal',kernel_regularizer=l2(1e-3))(y)
    
    model=Model(inputs=inputs,outputs=outputs)
    return model

model = resnet18()

import keras.losses
model.compile(optimizer=Adam(), loss= smooth_l1_loss, metrics = [my_metric])

model.summary()

def lr_sch(epoch):
    #400 total
    if epoch <20:
        return 1e-3 # 1e-3
    if 20<=epoch<40:
        return 1e-4
    if 40<=epoch<250:
        return 1e-5
    if 250<=epoch<300:
        return 1e-6
    if 300<=epoch<350:
        return 1e-5
    if epoch>=350:
        return 1e-6

lr_scheduler=LearningRateScheduler(lr_sch)
lr_reducer=ReduceLROnPlateau(monitor='val_my_metric',factor=0.2,patience=5,mode='max',min_lr=1e-3)

### We load the last best checkpoint to continue our training from the last learned stage.   ###
### It is advisable to change the learning states in lr_sch based on the current train_loss ###
### and val_loss.

try:
    model = keras.models.load_model('model_best.h5', custom_objects = {'smooth_l1_loss':smooth_l1_loss, #23 is the best
                                                                                        'my_metric': my_metric})
except:
    print("Previously checkpointed weights not avaialble in current file.")

checkpoint=ModelCheckpoint('model_best.h5',monitor='val_my_metric',verbose=0,save_best_only=True,mode='auto')


### While training the model is should be noted that if we see that the model is overfitting... ###
### ... that is, if the training loss is decreasing but the val_loss is not, then we have to... ###
### ... stop the current training and change the learning rates in the lr_sch() as we want. ... ###

model_details=model.fit_generator(image_generator(batch_size = 128), steps_per_epoch = 100,  validation_data = (data_test, box_test),
                        epochs=400, callbacks=[lr_scheduler,lr_reducer,checkpoint],verbose=1 )

model.save('model.h5')

scores=model.evaluate(data_test,box_test,verbose=1)
print('Test loss : ',scores[0])
print('Test accuracy : ',scores[1])



model = keras.models.load_model('model_best.h5', custom_objects = {'smooth_l1_loss':smooth_l1_loss, 
                                                                                     'my_metric': my_metric})


for i in tqdm.tqdm(range(0, len(test_csv))):

    path = image_dir + test_csv['image_name'][i]
    image = Image.open(path).convert('RGB')
    image = image.resize((224, 224))
    image=np.array(image,dtype=np.float32)
    image=image/255

    data_test = np.array([image]) #_data[:]
    result = model.predict(data_test)

    test_csv['x1'][i] = result[0][0]*640
    test_csv['y1'][i] = result[0][1]*480
    test_csv['x2'][i] = result[0][2]*640
    test_csv['y2'][i] = result[0][3]*480

### Change the prediction.csv directory ###
test_csv.to_csv('pred.csv',index= False)
