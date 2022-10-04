import numpy as np
import os
root = 'E:/graspnet/DataSet'


"""
加载碰撞标签
"""
def loadCollisionLabel(sceneId):  #这是给一个抓取场景
    labels = np.load(os.path.join(root, 'collision_label', 'scene_' + str(sceneId).zfill(4), 'collision_labels.npz'))
    collisionLabel = []
    for j in range(len(labels)):
        collisionLabel.append(labels['arr_{}'.format(j)])
    return collisionLabel

"""
加载抓取标签
"""
def loadGraspLabel(objId):   #加载单个物体
    file = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(objId).zfill(3))))  #打开文件
    graspLabel = (file['points'].astype(np.float32), file['offsets'].astype(np.float32), file['scores'].astype(np.float32))
    return graspLabel

"""
抓取点在物体位姿下的变换
"""
def transform_points(points, trans):  #points, trans:(N,3),(4,4)
    ones = np.ones([points.shape[0], 1], dtype=points.dtype)  # 补充一列只是为了进行矩阵运算,变为（N,4）
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T  # 多维的矩阵运算不能用dot,而要用matmul
    return points_[:, :3]  # 去除最后一维补充的列

"""
接近向量和旋转角度生成旋转矩阵
"""
def batch_viewpoint_params_to_matrix(batch_towards, batch_angle):#接近向量和平面内旋转角度生成旋转矩阵(n, 3)，(n, )--》(n, 3, 3)
    axis_x = batch_towards
    ones = np.ones(axis_x.shape[0], dtype=axis_x.dtype)
    zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
    axis_y = np.stack([-axis_x[:,1], axis_x[:,0], zeros], axis=-1)
    mask_y = (np.linalg.norm(axis_y, axis=-1) == 0)
    axis_y[mask_y] = np.array([0, 1, 0])
    axis_x = axis_x / np.linalg.norm(axis_x, axis=-1, keepdims=True)
    axis_y = axis_y / np.linalg.norm(axis_y, axis=-1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)
    R1 = np.stack([axis_x, axis_y, axis_z], axis=-1)

    sin = np.sin(batch_angle)
    cos = np.cos(batch_angle)
    R2 = np.stack([ones, zeros, zeros, zeros, cos, -sin, zeros, sin, cos], axis=-1)
    R2 = R2.reshape([-1,3,3])

    matrix = np.matmul(R1, R2)
    return matrix.astype(np.float32)