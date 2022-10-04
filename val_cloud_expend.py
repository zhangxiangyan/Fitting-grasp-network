from myutils.utils_load_data import *
from myutils.utils_fit_rot_rect_grasp_expend import *
from myutils.utils_show import *
from myutils.utils_grasp import *
from scipy.spatial.transform import Rotation as R


#设置为GPU方式
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


"""
数据处理
"""
####参数
sceneID, annID=3,0
camera='kinect'
img_size=224
batch_size=8



####加载路径
##路径
path_rgb,path_seg,path_depth=load_path_all_single(sceneID, annID)

##图像处理函数
def load_rgb(path):
    img = Image.open(path)
    img = img.resize((img_size, img_size), Image.NEAREST)  # 邻近采样
    img = np.array(img) / 255
    return np.array(img,dtype=np.float32).reshape((1,img_size,img_size,3))


def load_seg(path):
    img = Image.open(path)
    img = img.resize((img_size,img_size), Image.NEAREST)
    img = np.array(img)
    return np.array(img,dtype=np.int16)

test_x=load_rgb(path_rgb)
test_y=load_seg(path_seg)


"""
模型处理
"""
####加载模型
model=keras.models.load_model('h5/grasp_simple_deeplabv3_xception.h5')
pred=model.predict(test_x)
pred=np.argmax(pred,axis=-1)


"""
图像还原
"""
def arr2img_enlarge(arr):
    arr=arr.astype(np.uint8)   #必须要加这一行，否则会报错
    img=Image.fromarray(arr)
    img=img.resize((1280,720),Image.NEAREST)
    img.save('img/pred_seg.png')   #这里一定要先存成一张图，否则后期加载会有问题

arr2img_enlarge(pred[0])


"""
旋转矩形拟合
"""
grasp_point,grasp_angle,w=rot_rect_grasp_px_ang_w('img/pred_seg.png',10,15)
grasp_point=np.int32(np.round(grasp_point))  #不是整数，需要四舍五入


"""
深度图处理
"""
#通过抓取点的x,y坐标，找到深度图中对应的z
depth = np.array(Image.open(path_depth))
z=[]
for i in range(len(grasp_point)):
    z.append(depth[grasp_point[i, 1], grasp_point[i, 0]])   #这里一定要写成这样，因为之前像素就调换过
z=np.array(z)



"""
相机坐标
"""
##转换为点云
#参数
fx, fy, cx, cy = 631.5, 631.2, 639.5, 359.5
width, height = 1280, 720
scale = 1000

#生成点云
xmap=grasp_point[:,0]
ymap=grasp_point[:,1]

points_z = z / scale
points_x = (xmap - cx) * points_z / fx
points_y = (ymap - cy) * points_z / fy
points = np.stack([points_x, points_y, points_z], axis=-1)


"""
姿态
"""
batch_towards=np.array([0,0,1]).reshape(-1,3)   #竖直向下
batch_towards=np.repeat(batch_towards,len(grasp_angle), axis=0)


batch_angle=grasp_angle
Rs=batch_viewpoint_params_to_matrix(batch_towards, batch_angle)

random_view=np.zeros((len(Rs),3,3))
sz=60    #正负sz/2度的范围内
random_a=sz*np.random.rand(len(grasp_angle))-sz/2
random_b=sz*np.random.rand(len(grasp_angle))-sz/2

for i in range(len(Rs)):
    r = R.from_euler('xyz', [random_a[i],random_b[i],0], degrees=True)   #绕x,y轴旋转一个角度
    # r = R.from_euler('xyz', [random_a[i], 0, 0], degrees=True)  # 绕x轴旋转一个角度
    # r = R.from_euler('xyz', [0,random_b[i], 0], degrees=True)  # 绕y轴旋转一个角度
    random_view[i,:,:]= r.as_matrix()

Rs=np.matmul(random_view,Rs)  #左乘随机旋转矩阵




"""
可视化
"""
####可视化
gripper=[]
scores = 1

# 生成夹爪
for i in range(len(points)):
    gripper.append(mesh_gripper(points[i],Rs[i],w[i],0.01, scores))


pcd = creat_scene_cloudpoint(sceneID, annID, camera=camera)
o3d.visualization.draw_geometries([pcd,*gripper])
