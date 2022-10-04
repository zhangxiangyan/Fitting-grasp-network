import cv2
import numpy as np
import imutils


####水平扩展
def rot_rect_grasp_px(img_path,k):   #k是扩展的个数
	##读入图片
	img=cv2.imread(img_path,0)  #读入灰度值


	##读入所有物体编号
	obj=np.unique(img)
	obj=obj[obj!=0]  #不要背景的分类


	##大循环
	grasp_point = []
	grasp_angle = []
	thresh1 = 20
	thresh2 = 800
	for i in obj:
		# 通过值获取索引
		point = np.argwhere(img == i)
		point = point[:, [1, 0]]  # 这里需要交换两列

		# 拟合旋转矩形
		rect = cv2.minAreaRect(point)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		if ((w > thresh1) & (h > thresh1) & (w < thresh2) & (h < thresh2)):  # 将矩形限定在合理范围内
			box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
			# grasp_point.append(x)
			# grasp_point.append(y)

			l1=(box[1,1]-box[0,1])**2+(box[1,0]-box[0,0])**2   #三个点就能算出长边和短边的长度了
			l2=(box[2,1]-box[1,1])**2+(box[2,0]-box[1,0])**2
			if l1>l2:  #总是用长边进行计算
				#计算长边中线上的点
				dx=(box[1,0]-box[0,0])/k
				dy=(box[1,1]-box[0,1])/k
				for j in range(1,k):
					grasp_point.append((box[1,0]+box[2,0])/2-j*dx)
					grasp_point.append((box[1,1]+box[2,1])/2-j*dy)
					#计算角度
					if (box[1,0]-box[0,0])==0:
						theta =np.pi/2
					else:
						theta=np.arctan((box[1,1]-box[0,1])/(box[1,0]-box[0,0]))  #用长边算角度
					grasp_angle.append(theta)
			else:
				#计算长边中线上的点
				dx=(box[2,0]-box[1,0])/k
				dy=(box[2,1]-box[1,1])/k
				for j in range(1,k):
					grasp_point.append((box[1,0]+box[0,0])/2+j*dx)
					grasp_point.append((box[1,1]+box[0,1])/2+j*dy)
					#计算角度
					if (box[2, 0] - box[1, 0])==0:
						theta =np.pi/2
					else:
						theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))
					grasp_angle.append(theta)

	grasp_point=np.array(grasp_point).reshape(-1,2)
	grasp_angle=np.array(grasp_angle)

	return grasp_point,grasp_angle



####水平扩展+宽度
def rot_rect_grasp_px_w(img_path,k):   #k是扩展的个数
	##读入图片
	img=cv2.imread(img_path,0)  #读入灰度值


	##读入所有物体编号
	obj=np.unique(img)
	obj=obj[obj!=0]  #不要背景的分类


	##大循环
	grasp_point = []
	grasp_angle = []
	grasp_w = []  # 存储短边方向
	thresh1 = 20
	thresh2 = 800
	for i in obj:
		# 通过值获取索引
		point = np.argwhere(img == i)
		point = point[:, [1, 0]]  # 这里需要交换两列

		# 拟合旋转矩形
		rect = cv2.minAreaRect(point)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		if ((w > thresh1) & (h > thresh1) & (w < thresh2) & (h < thresh2)):  # 将矩形限定在合理范围内
			box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
			# grasp_point.append(x)
			# grasp_point.append(y)

			l1=(box[1,1]-box[0,1])**2+(box[1,0]-box[0,0])**2   #三个点就能算出长边和短边的长度了
			l2=(box[2,1]-box[1,1])**2+(box[2,0]-box[1,0])**2
			if l1>l2:  #总是用长边进行计算
				#计算长边中线上的点
				dx=(box[1,0]-box[0,0])/k
				dy=(box[1,1]-box[0,1])/k
				for j in range(1,k):
					grasp_point.append((box[1,0]+box[2,0])/2-j*dx)
					grasp_point.append((box[1,1]+box[2,1])/2-j*dy)
					#计算角度
					if (box[1,0]-box[0,0])==0:
						theta =np.pi/2
					else:
						theta=np.arctan((box[1,1]-box[0,1])/(box[1,0]-box[0,0]))  #用长边算角度
					grasp_angle.append(theta)

					grasp_w.append(l2)
			else:
				#计算长边中线上的点
				dx=(box[2,0]-box[1,0])/k
				dy=(box[2,1]-box[1,1])/k
				for j in range(1,k):
					grasp_point.append((box[1,0]+box[0,0])/2+j*dx)
					grasp_point.append((box[1,1]+box[0,1])/2+j*dy)
					#计算角度
					if (box[2, 0] - box[1, 0])==0:
						theta =np.pi/2
					else:
						theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))
					grasp_angle.append(theta)

					grasp_w.append(l1)

	grasp_point=np.array(grasp_point).reshape(-1,2)
	grasp_angle=np.array(grasp_angle)
	grasp_w=np.array(grasp_w)

	grasp_w[grasp_w < 1000] = 0.03
	grasp_w[(grasp_w > 1000) & (grasp_w < 4000)] = 0.05
	grasp_w[(grasp_w > 4000)&(grasp_w < 10000)] = 0.07
	grasp_w[(grasp_w > 10000) & (grasp_w < 25000)] = 0.10
	grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.13
	grasp_w[(grasp_w > 35000)] = 0.15

	return grasp_point,grasp_angle,grasp_w




####水平扩展+角度随机偏转
def rot_rect_grasp_px_ang(img_path,k,rd_ang=5):   #k是扩展的个数
	##读入图片
	img=cv2.imread(img_path,0)  #读入灰度值


	##读入所有物体编号
	obj=np.unique(img)
	obj=obj[obj!=0]  #不要背景的分类


	##大循环
	grasp_point = []
	grasp_angle = []
	thresh1 = 20
	thresh2 = 800
	for i in obj:
		# 通过值获取索引
		point = np.argwhere(img == i)
		point = point[:, [1, 0]]  # 这里需要交换两列

		# 拟合旋转矩形
		rect = cv2.minAreaRect(point)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		if ((w > thresh1) & (h > thresh1) & (w < thresh2) & (h < thresh2)):  # 将矩形限定在合理范围内
			box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
			# grasp_point.append(x)
			# grasp_point.append(y)

			l1=(box[1,1]-box[0,1])**2+(box[1,0]-box[0,0])**2   #三个点就能算出长边和短边的长度了
			l2=(box[2,1]-box[1,1])**2+(box[2,0]-box[1,0])**2
			if l1>l2:  #总是用长边进行计算
				#计算长边中线上的点
				dx=(box[1,0]-box[0,0])/k
				dy=(box[1,1]-box[0,1])/k
				for j in range(1,k):
					grasp_point.append((box[1,0]+box[2,0])/2-j*dx)
					grasp_point.append((box[1,1]+box[2,1])/2-j*dy)
					#计算角度
					if (box[1,0]-box[0,0])==0:
						theta =np.pi/2
					else:
						theta=np.arctan((box[1,1]-box[0,1])/(box[1,0]-box[0,0]))  #用长边算角度
					grasp_angle.append(theta)
			else:
				#计算长边中线上的点
				dx=(box[2,0]-box[1,0])/k
				dy=(box[2,1]-box[1,1])/k
				for j in range(1,k):
					grasp_point.append((box[1,0]+box[0,0])/2+j*dx)
					grasp_point.append((box[1,1]+box[0,1])/2+j*dy)
					#计算角度
					if (box[2, 0] - box[1, 0])==0:
						theta =np.pi/2
					else:
						theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))
					grasp_angle.append(theta)

	grasp_point=np.array(grasp_point).reshape(-1,2)
	grasp_angle=np.array(grasp_angle)

	angle_random=(np.pi/180)*rd_ang*(2*np.random.rand((k-1)*len(obj))-1)  #随机增加+-5度范围内随机数
	grasp_angle=grasp_angle+angle_random

	return grasp_point,grasp_angle



####水平扩展+角度随机偏转
def rot_rect_grasp_px_ang_w(img_path,k,rd_ang=5):   #k是扩展的个数
	##读入图片
	img=cv2.imread(img_path,0)  #读入灰度值


	##读入所有物体编号
	obj=np.unique(img)
	obj=obj[obj!=0]  #不要背景的分类


	##大循环
	grasp_point = []
	grasp_angle = []
	grasp_w = []  # 存储短边方向
	thresh1 = 20
	thresh2 = 800
	for i in obj:
		# 通过值获取索引
		point = np.argwhere(img == i)
		point = point[:, [1, 0]]  # 这里需要交换两列

		# 拟合旋转矩形
		rect = cv2.minAreaRect(point)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		if ((w > thresh1) & (h > thresh1) & (w < thresh2) & (h < thresh2)):  # 将矩形限定在合理范围内
			box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
			# grasp_point.append(x)
			# grasp_point.append(y)

			l1=(box[1,1]-box[0,1])**2+(box[1,0]-box[0,0])**2   #三个点就能算出长边和短边的长度了
			l2=(box[2,1]-box[1,1])**2+(box[2,0]-box[1,0])**2
			if l1>l2:  #总是用长边进行计算
				#计算长边中线上的点
				dx=(box[1,0]-box[0,0])/k
				dy=(box[1,1]-box[0,1])/k
				for j in range(1,k):
					grasp_point.append((box[1,0]+box[2,0])/2-j*dx)
					grasp_point.append((box[1,1]+box[2,1])/2-j*dy)
					#计算角度
					if (box[1,0]-box[0,0])==0:
						theta =np.pi/2
					else:
						theta=np.arctan((box[1,1]-box[0,1])/(box[1,0]-box[0,0]))  #用长边算角度
					grasp_angle.append(theta)

					grasp_w.append(l2)
			else:
				#计算长边中线上的点
				dx=(box[2,0]-box[1,0])/k
				dy=(box[2,1]-box[1,1])/k
				for j in range(1,k):
					grasp_point.append((box[1,0]+box[0,0])/2+j*dx)
					grasp_point.append((box[1,1]+box[0,1])/2+j*dy)
					#计算角度
					if (box[2, 0] - box[1, 0])==0:
						theta =np.pi/2
					else:
						theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))
					grasp_angle.append(theta)

					grasp_w.append(l1)

	grasp_point=np.array(grasp_point).reshape(-1,2)
	grasp_angle=np.array(grasp_angle)

	angle_random=(np.pi/180)*rd_ang*(2*np.random.rand(len(grasp_angle))-1)  #随机增加+-5度范围内随机数
	grasp_angle=grasp_angle+angle_random
	grasp_w=np.array(grasp_w)

	grasp_w[grasp_w < 1000] = 0.03
	grasp_w[(grasp_w > 1000) & (grasp_w < 4000)] = 0.05
	grasp_w[(grasp_w > 4000)&(grasp_w < 10000)] = 0.07
	grasp_w[(grasp_w > 10000) & (grasp_w < 25000)] = 0.10
	grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.13
	grasp_w[(grasp_w > 35000)] = 0.15

	return grasp_point,grasp_angle,grasp_w





####两端（失败）
def rot_rect_grasp_ld(img_path):
	##读入图片
	img=cv2.imread(img_path,0)  #读入灰度值


	##读入所有物体编号
	obj=np.unique(img)
	obj=obj[obj!=0]  #不要背景的分类


	##大循环
	grasp_point = []   #中心点，用获取高
	grasp_ld=[]  #两端点
	grasp_angle = []
	thresh1 = 20
	thresh2 = 800
	k=5
	for i in obj:
		# 通过值获取索引
		point = np.argwhere(img == i)
		point = point[:, [1, 0]]  # 这里需要交换两列

		# 拟合旋转矩形
		rect = cv2.minAreaRect(point)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		if ((w > thresh1) & (h > thresh1) & (w < thresh2) & (h < thresh2)):  # 将矩形限定在合理范围内
			box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
			grasp_point.append(x)
			grasp_point.append(y)

			grasp_point.append(x)  #以便给两个端点各配一个高
			grasp_point.append(y)

			l1=(box[1,1]-box[0,1])**2+(box[1,0]-box[0,0])**2   #三个点就能算出长边和短边的长度了
			l2=(box[2,1]-box[1,1])**2+(box[2,0]-box[1,0])**2
			if l1>l2:  #总是用长边进行计算
				#计算长边中线上的点
				dx=(box[1,0]-box[0,0])/k
				dy=(box[1,1]-box[0,1])/k
				#一个端点
				grasp_ld.append((box[1,0]+box[2,0])/2+dx)
				grasp_ld.append((box[1,1]+box[2,1])/2+dy)
				# 另一个端点
				grasp_ld.append((box[1,0]+box[2,0])/2-(k+1)*dx)
				grasp_ld.append((box[1,1]+box[2,1])/2-(k+1)*dy)
				#计算角度
				if (box[1,0]-box[0,0])==0:
					theta =np.pi/2
				else:
					theta=np.arctan((box[1,1]-box[0,1])/(box[1,0]-box[0,0]))  #用长边算角度
				grasp_angle.append(theta)
				grasp_angle.append(theta)   #给两个端点各配一个角度
			else:
				#计算长边中线上的点
				dx=(box[2,0]-box[1,0])/k
				dy=(box[2,1]-box[1,1])/k
				# 一个端点
				grasp_ld.append((box[1,0]+box[0,0])/2-dx)
				grasp_ld.append((box[1,1]+box[0,1])/2-dy)
				# 另一个端点
				grasp_ld.append((box[1,0]+box[0,0])/2+(k+1)*dx)
				grasp_ld.append((box[1,1]+box[0,1])/2+(k+1)*dy)
				#计算角度
				if (box[2, 0] - box[1, 0])==0:
					theta =np.pi/2
				else:
					theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))
				grasp_angle.append(theta)
				grasp_angle.append(theta)  # 给两个端点各配一个角度

	grasp_point=np.array(grasp_point).reshape(-1,2)
	grasp_ld=np.array(grasp_ld).reshape(-1, 2)
	grasp_angle=np.array(grasp_angle)


	return grasp_point,grasp_ld,grasp_angle


