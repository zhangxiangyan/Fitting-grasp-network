import tensorflow as tf
import numpy as np
import open3d as o3d
import os
import copy
from PIL import Image
import matplotlib.pyplot as plt
from myutils.utils_load_data import *
import cv2
from myutils.utils_color_imshow import *
from myutils.utils_draw_mask import *
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
sceneID_test=np.arange(100,190)
annID=np.arange(256)

camera='kinect'
img_size=224
batch_size=8
num_classes=89


model_save_path='h5/grasp_simple_deeplabv3_xception_all.h5'

##随机场景和视角
num_random_choice=4
sceneID_random=np.random.choice(sceneID_test, num_random_choice, replace=False)

####加载路径
##路径
test_path_x,test_path_y=load_path(sceneID_random,annID[0:1])



##图像处理函数
def load_x(path):
    img_x = []
    for i in path:
        img = Image.open(i)
        img = img.resize((img_size,img_size), Image.NEAREST)  # 邻近采样
        img = np.array(img) / 255.
        img_x.append(img)
    return np.array(img_x,dtype=np.float32)

def load_y(path):
    img_y = []
    for i in path:
        img = Image.open(i)
        img = img.resize((img_size, img_size), Image.NEAREST)
        img = np.array(img)
        img_y.append(img)
    return np.array(img_y,dtype=np.int16)


##得到数据
test_x=load_x(test_path_x)
test_y=load_y(test_path_y)


"""
模型处理
"""
####加载模型
input_shape = [img_size,img_size,3]
model=Deeplabv3(input_shape, num_classes, alpha=1., backbone="xception", downsample_factor=16)

model.load_weights(model_save_path)

pred=model.predict(test_x)
pred=np.argmax(pred,axis=-1)





"""
可视化
"""
num=num_random_choice
for i in range(num):
    plt.subplot(num,4,1+4*i)
    plt.imshow(test_x[i])
    plt.axis('off')
    if i==0:
        plt.title('Image')

    plt.subplot(num,4,2+4*i)
    plt.imshow(color_imshow(test_y[i]))
    plt.axis('off')
    if i == 0:
        plt.title('Label')

    plt.subplot(num,4,3+4*i)
    plt.imshow(color_imshow(pred[i]))
    plt.axis('off')
    if i == 0:
        plt.title('Prediction')

    plt.subplot(num,4,4+4*i)
    plt.imshow(draw_mask_pred_show(color_imshow(pred[i]), pred[i]))
    plt.axis('off')
    if i == 0:
        plt.title('Fitting')

plt.show()