import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

####用于评估
def generate_matrix(gt_image, pre_image, num_class):
    mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask

    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)
    return confusion_matrix

def Pixel_Accuracy_Class(confusion_matrix):  #加smooth防止分母为0
    Acc = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    Acc = np.nanmean(Acc)
    return Acc



####用于训练
# def MPA(num_class):
#     def _MPA(y_true, y_pred):
#         #利用mask实现降维
#         mask = (y_true >= 0) & (y_true < num_class)
#         # 计数+混淆矩阵
#         # label = num_class * y_true[mask].astype('int') + y_pred[mask]
#         # h = tf.constant(y_true[mask], dtype=tf.int32)
#         label = num_class * y_true[mask]+ y_pred[mask]
#         # h = tf.constant(6, dtype=tf.int32)
#
#         # count = np.bincount(label, minlength=num_class ** 2)
#         count = tf.convert_to_tensor(np.bincount(label, minlength=num_class ** 2).reshape(-1))
#         confusion_matrix = count.reshape(num_class, num_class)
#         #计算mpa
#         Acc = tf.convert_to_tensor(np.diag(confusion_matrix))/ confusion_matrix.sum(axis=1)
#         Acc = tf.convert_to_tensor(np.nanmean(Acc))
#         return Acc
#     return _MPA


# def f_score(threshold=0.5,beta=0.5,smooth=1e-5):
#     def _F_Score(y_true,y_pred):
#         y_pred=K.greater(y_pred,threshold) #将被转变为布尔值
#         y_pred=K.cast(y_pred,K.floatx()) #把布尔值转换为数值
#
#         tp=K.sum(y_true*y_pred,axis=[0,1,2])
#         # fn=K.sum(y_true,axis=[0,1,2])-tp
#         fp=K.sum(y_pred,axis=[0,1,2])-tp
#
#         score=(tp+smooth)/(tp+fp+smooth)  #加smooth是为了防止分母为0
#         return score
#     return _F_Score




