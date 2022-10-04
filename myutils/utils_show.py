import numpy as np
import open3d as o3d
import os
import copy
from PIL import Image


"""
生成夹持器模型
"""
def mesh_gripper(center,R,width,depth,score):
    #parameters
    short=0.004  #截面边长
    long=0.02   #基础指长
    end=0.04   #尾长

    #box
    left = mesh_box(depth + long + short, short, short, -(long +short), -(width / 2+short), -short / 2)
    # left = mesh_box(depth + long + short, short, short, -(long + short), -width / 2, -short / 2)
    right = mesh_box(depth + long + short, short, short, -(long+short), width / 2, -short / 2)
    bottom = mesh_box(short, width, short, -(long+short), -width / 2, -short / 2)
    tail = mesh_box(end, short, short, -(long+short+end), -short / 2, -short / 2)

    #gripper
        # vertices
    left_points = np.array(left.vertices)
    right_points = np.array(right.vertices)
    bottom_points = np.array(bottom.vertices)
    tail_points = np.array(tail.vertices)
    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center  # 旋转和平移变换


        # triangels
    left_triangles = np.array(left.triangles)
    right_triangles = np.array(right.triangles) + 8
    bottom_triangles = np.array(bottom.triangles) + 16
    tail_triangles = np.array(tail.triangles) + 24
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)

        # vertex_colors
    color_r = score
    color_g = 0
    color_b = 1 - score

    # color_r = 0
    # color_g = 0
    # color_b = 1

    color = np.array([[color_r, color_g, color_b] for _ in range(len(vertices))])

        #gripper_mesh
    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(color)

    return gripper

"""
生成抓取器的一个杆
"""
def mesh_box(w,h,d,dx=0,dy=0,dz=0):
    ####parameters
    #vertices顶点
    vertices=np.array([[0,0,0],
              [w,0,0],
              [0,0,d],
              [w,0,d],
              [0,h,0],
              [w,h,0],
              [0,h,d],
              [w,h,d]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    #triangels三角面片
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                          [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                          [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    #vertex_colors顶点颜色
    ###box
    box=o3d.geometry.TriangleMesh()
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box



"""
生成场景点云
"""
def creat_scene_cloudpoint(sceneID,annID,camera='kinect'):
    # 文件路径
    path = 'E:/graspnet/DataSet/scenes/scene_{}/{}'.format(str(sceneID).zfill(4),camera)
    path_rgb = os.path.join(path, 'rgb', '{}.png'.format(str(annID).zfill(4)))
    path_depth = os.path.join(path, 'depth', '{}.png'.format(str(annID).zfill(4)))

    # 相机内参
    # camera = np.load(os.path.join(path, 'camK.npy'))
    # fx, fy = camera[0, 0], camera[1, 1]
    # cx, cy = camera[0, 2], camera[1, 2]
    if camera == 'kinect':
        fx,fy,cx,cy=631.5,631.2,639.5,359.5
    elif camera == 'realsense':
        fx,fy,cx,cy=927.17,927.37,639.5,359.5
    width, height = 1280, 720
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # 加载深度图和彩图
    color_raw = o3d.io.read_image(path_rgb)
    depth_raw = o3d.io.read_image(path_depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw,
                                                                    convert_rgb_to_intensity=False)  # convert_rgb_to_intensity=False加这句显示的就是彩图

    # 生成点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    return pcd



"""
生成点云的第二种方式：只用深度图，用于训练数据的数据预处理
"""
def create_pcd_from_depth(sceneID,annID,camera='kinect'):
    #路径
    # 文件路径
    path = 'E:/graspnet/DataSet/scenes/scene_{}/{}'.format(str(sceneID).zfill(4),camera)
    path_depth = os.path.join(path, 'depth', '{}.png'.format(str(annID).zfill(4)))
    depth = np.array(Image.open(path_depth))

    #参数
    if camera == 'kinect':
        fx,fy,cx,cy=631.5,631.2,639.5,359.5
    elif camera == 'realsense':
        fx,fy,cx,cy=927.17,927.37,639.5,359.5
    width, height = 1280, 720
    scale=1000

    #生成点云
    xmap = np.arange(width)
    ymap = np.arange(height)
    xmap, ymap = np.meshgrid(xmap, ymap)   #生成网格点坐标矩阵
    points_z = depth / scale
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)

    return cloud






"""
随机选取给定数量的夹持器个数
"""
def random_sample(grasp,numGrasp=20):
    shuffled_grasp_group_array = copy.deepcopy(grasp)
    np.random.shuffle(shuffled_grasp_group_array)
    return shuffled_grasp_group_array[:numGrasp]




"""
点云坐标转变为图像坐标
"""
import os
import cv2
import matplotlib.pyplot as plt

root='E:/graspnet/DataSet'
w,h=1280,720


def PointCloud_to_img(points,sceneId,annId,camera='kinect'):
    # colors=cv2.imread(os.path.join(root, 'scenes', 'scene_%04d' % sceneId, camera, 'rgb', '%04d.png' % annId))/ 255.0
    # depths=cv2.imread(os.path.join(root, 'scenes', 'scene_%04d' % sceneId, camera, 'depth', '%04d.png' % annId), cv2.IMREAD_UNCHANGED)

    intrinsics = np.load(os.path.join(root, 'scenes', 'scene_%04d' % sceneId, camera, 'camK.npy'))  # 加载相机内参文件
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    s = 1000.0
    # if camera == 'kinect':
    #     fx,fy,cx,cy=631.5,631.2,639.5,359.5
    # elif camera == 'realsense':
    #     fx,fy,cx,cy=927.17,927.37,639.5,359.5

    depths_new=points[:,2]*s
    x=np.rint(points[:,0]/points[:,2]*fx+cx)
    y=np.rint(points[:,1]/points[:,2]*fy+cy)
    img_xy=np.stack([y,x],axis=-1).astype(np.int32)

    return img_xy,depths_new