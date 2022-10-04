import tensorflow as tf
import numpy as np
import open3d as o3d
import os
import copy
from PIL import Image
import matplotlib.pyplot as plt
from myutils.utils_load_data import *
from nets_deeplabv3.deeplab import *


#设置为GPU方式
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


"""
数据处理
"""
####参数
sceneID_train=np.arange(100)
sceneID_test_seen=np.arange(100,130)
sceneID_test_similar=np.arange(130,160)
sceneID_test_novel=np.arange(160,190)
sceneID_all=np.arange(190)
annID=np.arange(256)

camera='kinect'
img_size=224
# batch_size=8
batch_size=6
epoch=20

model_save_path='h5/grasp_simple_deeplabv3_xception.h5'
num_classes=89


####加载路径
##路径
path_x,path_y=load_path(sceneID_train,annID)

##乱序
np.random.seed(2022)
np.random.shuffle(path_x)
np.random.seed(2022)
np.random.shuffle(path_y)

##分配
n=len(path_x)   #48640
n_train=int(0.8*n)   #38912
train_path_x,train_path_y=path_x[:n_train],path_y[:n_train]
test_path_x,test_path_y=path_x[n_train:],path_y[n_train:]


####数据生成器
train_data_loader=DataSequence(train_path_x,train_path_y,batch_size,img_size,shuffle=True)
test_data_loader=DataSequence(test_path_x,test_path_y,batch_size,img_size,shuffle=False)



"""
模型处理
"""
####创建模型
input_shape = [img_size,img_size,3]
model=Deeplabv3(input_shape, num_classes, alpha=1., backbone="xception", downsample_factor=16)


####编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  #将学习率由0.001调整到0.0005
    loss=['sparse_categorical_crossentropy'],
    metrics=['accuracy']
)


####断点续训
checkpoint_save_path='./checkpoint/grasp_simple_xception.ckpt'
if os.path.exists(checkpoint_save_path+'.index'):
    model.load_weights(checkpoint_save_path)

cp_callbacl=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                               save_weights_only=True,
                                               monitor='val_loss',
                                               save_best_only=True)

####调节学习率
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,mode='auto')


####训练模型
history=model.fit(
    train_data_loader,
    epochs=epoch,
    validation_data=test_data_loader,
    callbacks=[cp_callbacl,reduce_lr]
)


####保存模型
model.save(model_save_path)


####打印模型
model.summary()


####可视化模型
plt.subplot(1,2,1)
plt.plot(history.epoch,history.history['loss'],label='train_loss')
plt.plot(history.epoch,history.history['val_loss'],label='test_loss')
plt.title('The loss curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.epoch,history.history['accuracy'],label='train_acc')
plt.plot(history.epoch,history.history['val_accuracy'],label='test_acc')
plt.title('The accuracy curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()