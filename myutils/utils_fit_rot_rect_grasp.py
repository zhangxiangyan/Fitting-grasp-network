import cv2
import numpy as np
import imutils


def rot_rect_grasp(img_path):
	##读入图片
	img=cv2.imread(img_path,0)  #读入灰度值


	##读入所有物体编号
	obj=np.unique(img)
	obj=obj[obj!=0]  #不要背景的分类


	##大循环
	grasp_point = []
	grasp_angle = []
	for i in obj:
		# 通过值获取索引
		point = np.argwhere(img == i)
		point = point[:, [1, 0]]  # 这里需要交换两列

		# 拟合旋转矩形
		rect = cv2.minAreaRect(point)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		grasp_point.append(x)
		grasp_point.append(y)
		grasp_angle.append(a)

	grasp_point=np.array(grasp_point).reshape(-1,2)
	grasp_angle=np.array(grasp_angle)

	return grasp_point,grasp_angle


####根据物体分类进行检测
def rot_rect_grasp_ang(img_path):
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
			grasp_point.append(x)
			grasp_point.append(y)

			l1=(box[1,1]-box[0,1])**2+(box[1,0]-box[0,0])**2
			l2=(box[2,1]-box[1,1])**2+(box[2,0]-box[1,0])**2
			if l1>l2:
				if (box[1,0]-box[0,0])==0:
					theta =np.pi/2
				else:
					theta=np.arctan((box[1,1]-box[0,1])/(box[1,0]-box[0,0]))
			else:
				if (box[2, 0] - box[1, 0])==0:
					theta =np.pi/2
				else:
					theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))

			grasp_angle.append(theta)

	grasp_point=np.array(grasp_point).reshape(-1,2)
	grasp_angle=np.array(grasp_angle)

	return grasp_point,grasp_angle



####二值检测
def rot_rect_grasp_ang_bi(img):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cnts = cv2.findContours(img_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# 遍历轮廓集
	grasp_point = []
	grasp_angle = []
	thresh1 = 50
	thresh2 = 1000
	for c in cnts:
		rect = cv2.minAreaRect(c)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		if ((w > thresh1) & (h > thresh1)&(w < thresh2) & (h < thresh2)):  #筛除掉那些误判的点
			box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
			grasp_point.append(x)
			grasp_point.append(y)

			l1 = (box[1, 1] - box[0, 1]) ** 2 + (box[1, 0] - box[0, 0]) ** 2
			l2 = (box[2, 1] - box[1, 1]) ** 2 + (box[2, 0] - box[1, 0]) ** 2
			if l1 > l2:
				if (box[1, 0] - box[0, 0]) == 0:
					theta = np.pi / 2
				else:
					theta = np.arctan((box[1, 1] - box[0, 1]) / (box[1, 0] - box[0, 0]))
			else:
				if (box[2, 0] - box[1, 0]) == 0:
					theta = np.pi / 2
				else:
					theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))

			grasp_angle.append(theta)

	grasp_point = np.array(grasp_point).reshape(-1, 2)
	grasp_angle = np.array(grasp_angle)

	return grasp_point, grasp_angle



####根据物体分类进行检测,拟合宽度
def rot_rect_grasp_ang_w(img_path):
	##读入图片
	img=cv2.imread(img_path,0)  #读入灰度值


	##读入所有物体编号
	obj=np.unique(img)
	obj=obj[obj!=0]  #不要背景的分类


	##大循环
	grasp_point = []
	grasp_angle = []
	grasp_w=[]   #存储短边方向
	thresh1 = 20
	thresh2 = 600
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

			l1=(box[1,1]-box[0,1])**2+(box[1,0]-box[0,0])**2
			l2=(box[2,1]-box[1,1])**2+(box[2,0]-box[1,0])**2
			if l1>l2:
				grasp_w.append(l2)
				if (box[1,0]-box[0,0])==0:
					theta =np.pi/2
				else:
					theta=np.arctan((box[1,1]-box[0,1])/(box[1,0]-box[0,0]))
			else:
				grasp_w.append(l1)
				if (box[2, 0] - box[1, 0])==0:
					theta =np.pi/2
				else:
					theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))

			grasp_angle.append(theta)

	grasp_point=np.array(grasp_point).reshape(-1,2)
	grasp_angle=np.array(grasp_angle)
	grasp_w=np.array(grasp_w)

	# grasp_w[grasp_w<4000]=0.05
	# grasp_w[(grasp_w > 4000)&(grasp_w < 10000)] = 0.07
	# grasp_w[(grasp_w > 10000) & (grasp_w < 25000)] = 0.10
	# grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.13
	# grasp_w[(grasp_w > 35000)] = 0.15

	grasp_w[grasp_w < 1000] = 0.03
	grasp_w[(grasp_w > 1000) & (grasp_w < 4000)] = 0.05
	grasp_w[(grasp_w > 4000)&(grasp_w < 10000)] = 0.07
	grasp_w[(grasp_w > 10000) & (grasp_w < 25000)] = 0.10
	grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.13
	grasp_w[(grasp_w > 35000)] = 0.15

	return grasp_point,grasp_angle,grasp_w



####二值检测，拟合宽度
def rot_rect_grasp_ang_bi_w(img):
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cnts = cv2.findContours(img_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# 遍历轮廓集
	grasp_point = []
	grasp_angle = []
	grasp_w = []  # 存储短边方向
	thresh1 = 30
	thresh2 = 1000
	for c in cnts:
		rect = cv2.minAreaRect(c)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		if ((w > thresh1) & (h > thresh1)&(w < thresh2) & (h < thresh2)):  #筛除掉那些误判的点
			box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
			grasp_point.append(x)
			grasp_point.append(y)

			l1 = (box[1, 1] - box[0, 1]) ** 2 + (box[1, 0] - box[0, 0]) ** 2
			l2 = (box[2, 1] - box[1, 1]) ** 2 + (box[2, 0] - box[1, 0]) ** 2
			if l1 > l2:
				grasp_w.append(l2)
				if (box[1, 0] - box[0, 0]) == 0:
					theta = np.pi / 2
				else:
					theta = np.arctan((box[1, 1] - box[0, 1]) / (box[1, 0] - box[0, 0]))
			else:
				grasp_w.append(l1)
				if (box[2, 0] - box[1, 0]) == 0:
					theta = np.pi / 2
				else:
					theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))

			grasp_angle.append(theta)

	grasp_point = np.array(grasp_point).reshape(-1, 2)
	grasp_angle = np.array(grasp_angle)
	grasp_w=np.array(grasp_w)

	# grasp_w[grasp_w<3000]=0.02
	# grasp_w[(grasp_w > 3000)&(grasp_w < 10000)] = 0.05
	# grasp_w[(grasp_w > 10000) & (grasp_w < 25000)] = 0.08
	# grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.12
	# grasp_w[(grasp_w > 35000)] = 0.15

	grasp_w[grasp_w < 1000] = 0.03
	grasp_w[(grasp_w > 1000) & (grasp_w < 4000)] = 0.05
	grasp_w[(grasp_w > 4000)&(grasp_w < 10000)] = 0.07
	grasp_w[(grasp_w > 10000) & (grasp_w < 25000)] = 0.10
	grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.13
	grasp_w[(grasp_w > 35000)] = 0.15

	return grasp_point, grasp_angle,grasp_w


####二值检测，拟合宽度
def rot_rect_grasp_ang_bi_w_mask(img):
	# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# 遍历轮廓集
	grasp_point = []
	grasp_angle = []
	grasp_w = []  # 存储短边方向
	thresh1 = 30
	thresh2 = 1000
	for c in cnts:
		rect = cv2.minAreaRect(c)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		if ((w > thresh1) & (h > thresh1)&(w < thresh2) & (h < thresh2)):  #筛除掉那些误判的点
			box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
			grasp_point.append(x)
			grasp_point.append(y)

			l1 = (box[1, 1] - box[0, 1]) ** 2 + (box[1, 0] - box[0, 0]) ** 2
			l2 = (box[2, 1] - box[1, 1]) ** 2 + (box[2, 0] - box[1, 0]) ** 2
			if l1 > l2:
				grasp_w.append(l2)
				if (box[1, 0] - box[0, 0]) == 0:
					theta = np.pi / 2
				else:
					theta = np.arctan((box[1, 1] - box[0, 1]) / (box[1, 0] - box[0, 0]))
			else:
				grasp_w.append(l1)
				if (box[2, 0] - box[1, 0]) == 0:
					theta = np.pi / 2
				else:
					theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))

			grasp_angle.append(theta)

	grasp_point = np.array(grasp_point).reshape(-1, 2)
	grasp_angle = np.array(grasp_angle)
	grasp_w=np.array(grasp_w)

	grasp_w[grasp_w<1000]=0.03
	grasp_w[(grasp_w > 1000)&(grasp_w < 3000)] = 0.05
	grasp_w[(grasp_w > 3000)&(grasp_w < 10000)] = 0.07
	grasp_w[(grasp_w > 10000) & (grasp_w < 25000)] = 0.10
	grasp_w[(grasp_w > 25000) & (grasp_w < 30000)] = 0.13
	grasp_w[(grasp_w > 30000) & (grasp_w < 35000)] = 0.15
	grasp_w[(grasp_w > 35000)] = 0.17

	return grasp_point, grasp_angle,grasp_w


####水平扩展+宽度+短边+长边+角度随机旋转,用于vrep仿真
def rot_rect_grasp_px_w_db_cb_ang_vrep(img,k1,k2,rd_ang):   #k1是长边扩展的个数，k2是短边扩展的个数,flag控制有无短边扩展
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cnts = cv2.findContours(img_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	##大循环
	grasp_point = []
	grasp_angle = []
	grasp_w = []  # 存储短边方向
	# thresh1 = 20
	# thresh2 = 800
	thresh1 = 30
	thresh2 = 1000
	for i in cnts:
		# 拟合旋转矩形
		rect = cv2.minAreaRect(i)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		if ((w > thresh1) & (h > thresh1) & (w < thresh2) & (h < thresh2)):  # 将矩形限定在合理范围内
			box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
			# grasp_point.append(x)
			# grasp_point.append(y)

			l1=(box[1,1]-box[0,1])**2+(box[1,0]-box[0,0])**2   #三个点就能算出长边和短边的长度了
			l2=(box[2,1]-box[1,1])**2+(box[2,0]-box[1,0])**2

			#####长边的参数
			if l1>l2:  #总是用长边进行计算
				#计算长边中线上的点
				dx=(box[1,0]-box[0,0])/k1
				dy=(box[1,1]-box[0,1])/k1
				for j in range(1,k1):
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
				dx=(box[2,0]-box[1,0])/k1
				dy=(box[2,1]-box[1,1])/k1
				for j in range(1,k1):
					grasp_point.append((box[1,0]+box[0,0])/2+j*dx)
					grasp_point.append((box[1,1]+box[0,1])/2+j*dy)
					#计算角度
					if (box[2, 0] - box[1, 0])==0:
						theta =np.pi/2
					else:
						theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))
					grasp_angle.append(theta)

					grasp_w.append(l1)


			#####短边的参数
			if l1 < l2:  # 现在用短边进行计算
				# 计算长边中线上的点
				dx = (box[1, 0] - box[0, 0]) / k2
				dy = (box[1, 1] - box[0, 1]) / k2
				for j in range(1, k2):
					grasp_point.append((box[1, 0] + box[2, 0]) / 2 - j * dx)
					grasp_point.append((box[1, 1] + box[2, 1]) / 2 - j * dy)
					# 计算角度
					if (box[1, 0] - box[0, 0]) == 0:
						theta = np.pi / 2
					else:
						theta = np.arctan((box[1, 1] - box[0, 1]) / (box[1, 0] - box[0, 0]))  # 用长边算角度
					grasp_angle.append(theta)

					grasp_w.append(l2)
			else:
				# 计算长边中线上的点
				dx = (box[2, 0] - box[1, 0]) / k2
				dy = (box[2, 1] - box[1, 1]) / k2
				for j in range(1, k2):
					grasp_point.append((box[1, 0] + box[0, 0]) / 2 + j * dx)
					grasp_point.append((box[1, 1] + box[0, 1]) / 2 + j * dy)
					# 计算角度
					if (box[2, 0] - box[1, 0]) == 0:
						theta = np.pi / 2
					else:
						theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))
					grasp_angle.append(theta)

					grasp_w.append(l1)


	grasp_point=np.array(grasp_point).reshape(-1,2)
	grasp_angle=np.array(grasp_angle)
	grasp_w=np.array(grasp_w)

	angle_random=(np.pi/180)*rd_ang*(2*np.random.rand(len(grasp_angle))-1)  #随机增加+-5度范围内随机数
	grasp_angle=grasp_angle+angle_random

	# grasp_w[grasp_w < 1000] = 0.03
	# grasp_w[(grasp_w > 1000) & (grasp_w < 4000)] = 0.045
	# grasp_w[(grasp_w > 4000) & (grasp_w < 8000)] = 0.06
	# grasp_w[(grasp_w > 8000)&(grasp_w < 13000)] = 0.08
	# grasp_w[(grasp_w > 13000) & (grasp_w < 25000)] = 0.10
	# grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.13
	# grasp_w[(grasp_w > 35000) & (grasp_w < 50000)] = 0.15
	# grasp_w[(grasp_w > 50000) & (grasp_w < 70000)] = 0.19
	# grasp_w[(grasp_w > 70000)] = 0.23

	a=0.03
	grasp_w[grasp_w < 1000] = 0.03+a
	grasp_w[(grasp_w > 1000) & (grasp_w < 4000)] = 0.045+a
	grasp_w[(grasp_w > 4000) & (grasp_w < 8000)] = 0.06+a
	grasp_w[(grasp_w > 8000)&(grasp_w < 13000)] = 0.08+a
	grasp_w[(grasp_w > 13000) & (grasp_w < 25000)] = 0.10+a
	grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.13+a
	grasp_w[(grasp_w > 35000) & (grasp_w < 50000)] = 0.15+a
	grasp_w[(grasp_w > 50000) & (grasp_w < 70000)] = 0.19+a
	grasp_w[(grasp_w > 70000)] = 0.23+a


	return grasp_point,grasp_angle,grasp_w



####水平扩展+宽度+短边+长边+角度随机旋转,用于vrep仿真,查看长宽的差距
def rot_rect_grasp_px_w_db_cb_ang_vrep_dist(img,k1,k2,rd_ang):   #k1是长边扩展的个数，k2是短边扩展的个数,flag控制有无短边扩展
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cnts = cv2.findContours(img_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	##大循环
	grasp_point = []
	grasp_angle = []
	grasp_w = []  # 存储短边方向
	# thresh1 = 20
	# thresh2 = 800
	thresh1 = 30
	thresh2 = 1000
	for i in cnts:
		# 拟合旋转矩形
		rect = cv2.minAreaRect(i)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		if ((w > thresh1) & (h > thresh1) & (w < thresh2) & (h < thresh2)):  # 将矩形限定在合理范围内
			box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
			# grasp_point.append(x)
			# grasp_point.append(y)

			l1=(box[1,1]-box[0,1])**2+(box[1,0]-box[0,0])**2   #三个点就能算出长边和短边的长度了
			l2=(box[2,1]-box[1,1])**2+(box[2,0]-box[1,0])**2

			#####长边的参数
			if l1>l2:  #总是用长边进行计算
				#计算长边中线上的点
				dx=(box[1,0]-box[0,0])/k1
				dy=(box[1,1]-box[0,1])/k1
				for j in range(1,k1):
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
				dx=(box[2,0]-box[1,0])/k1
				dy=(box[2,1]-box[1,1])/k1
				for j in range(1,k1):
					grasp_point.append((box[1,0]+box[0,0])/2+j*dx)
					grasp_point.append((box[1,1]+box[0,1])/2+j*dy)
					#计算角度
					if (box[2, 0] - box[1, 0])==0:
						theta =np.pi/2
					else:
						theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))
					grasp_angle.append(theta)

					grasp_w.append(l1)

			if np.absolute(w-h)<=40:
				#####短边的参数
				if l1 < l2:  # 现在用短边进行计算
					# 计算长边中线上的点
					dx = (box[1, 0] - box[0, 0]) / k2
					dy = (box[1, 1] - box[0, 1]) / k2
					for j in range(1, k2):
						grasp_point.append((box[1, 0] + box[2, 0]) / 2 - j * dx)
						grasp_point.append((box[1, 1] + box[2, 1]) / 2 - j * dy)
						# 计算角度
						if (box[1, 0] - box[0, 0]) == 0:
							theta = np.pi / 2
						else:
							theta = np.arctan((box[1, 1] - box[0, 1]) / (box[1, 0] - box[0, 0]))  # 用长边算角度
						grasp_angle.append(theta)

						grasp_w.append(l2)
				else:
					# 计算长边中线上的点
					dx = (box[2, 0] - box[1, 0]) / k2
					dy = (box[2, 1] - box[1, 1]) / k2
					for j in range(1, k2):
						grasp_point.append((box[1, 0] + box[0, 0]) / 2 + j * dx)
						grasp_point.append((box[1, 1] + box[0, 1]) / 2 + j * dy)
						# 计算角度
						if (box[2, 0] - box[1, 0]) == 0:
							theta = np.pi / 2
						else:
							theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))
						grasp_angle.append(theta)

						grasp_w.append(l1)
			else:
				pass


	grasp_point=np.array(grasp_point).reshape(-1,2)
	grasp_angle=np.array(grasp_angle)
	grasp_w=np.array(grasp_w)

	angle_random=(np.pi/180)*rd_ang*(2*np.random.rand(len(grasp_angle))-1)  #随机增加+-5度范围内随机数
	grasp_angle=grasp_angle+angle_random

	grasp_w[grasp_w < 1000] = 0.03
	grasp_w[(grasp_w > 1000) & (grasp_w < 4000)] = 0.045
	# grasp_w[(grasp_w > 4000) & (grasp_w < 8000)] = 0.06
	# grasp_w[(grasp_w > 8000)&(grasp_w < 13000)] = 0.08
	grasp_w[(grasp_w > 4000) & (grasp_w < 8000)] = 0.07
	grasp_w[(grasp_w > 8000)&(grasp_w < 13000)] = 0.085
	grasp_w[(grasp_w > 13000) & (grasp_w < 25000)] = 0.10
	grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.13
	grasp_w[(grasp_w > 35000) & (grasp_w < 50000)] = 0.15
	grasp_w[(grasp_w > 50000) & (grasp_w < 70000)] = 0.19
	grasp_w[(grasp_w > 70000)] = 0.23

	# a=0.03
	# grasp_w[grasp_w < 1000] = 0.03
	# grasp_w[(grasp_w > 1000) & (grasp_w < 4000)] = 0.045+a
	# grasp_w[(grasp_w > 4000) & (grasp_w < 8000)] = 0.06
	# grasp_w[(grasp_w > 8000)&(grasp_w < 13000)] = 0.08
	# grasp_w[(grasp_w > 13000) & (grasp_w < 25000)] = 0.10
	# grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.13
	# grasp_w[(grasp_w > 35000) & (grasp_w < 50000)] = 0.15
	# grasp_w[(grasp_w > 50000) & (grasp_w < 70000)] = 0.19
	# grasp_w[(grasp_w > 70000)] = 0.23


	return grasp_point,grasp_angle,grasp_w



####水平扩展+宽度+短边+长边+角度随机旋转,用于vrep仿真,查看长宽的差距
def rot_rect_grasp_px_w_db_cb_ang_vrep_dist_ram1(img,k1,k2,rd_ang):   #k1是长边扩展的个数，k2是短边扩展的个数,flag控制有无短边扩展
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cnts = cv2.findContours(img_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	##大循环
	grasp_point = []
	grasp_angle = []
	grasp_w = []  # 存储短边方向
	# thresh1 = 20
	# thresh2 = 800
	thresh1 = 30
	thresh2 = 1000
	for i in cnts:
		# 拟合旋转矩形
		rect = cv2.minAreaRect(i)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
		((x, y), (w, h), a) = rect
		if ((w > thresh1) & (h > thresh1) & (w < thresh2) & (h < thresh2)):  # 将矩形限定在合理范围内
			box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
			# grasp_point.append(x)
			# grasp_point.append(y)

			l1=(box[1,1]-box[0,1])**2+(box[1,0]-box[0,0])**2   #三个点就能算出长边和短边的长度了
			l2=(box[2,1]-box[1,1])**2+(box[2,0]-box[1,0])**2

			#####长边的参数
			if l1>l2:  #总是用长边进行计算
				#计算长边中线上的点
				dx=(box[1,0]-box[0,0])/k1
				dy=(box[1,1]-box[0,1])/k1
				grasp_point_x=[]
				grasp_point_y = []
				for j in range(1,k1):
					grasp_point_x.append((box[1,0]+box[2,0])/2-j*dx)
					grasp_point_y.append((box[1,1]+box[2,1])/2-j*dy)
				grasp_point_x=np.array(grasp_point_x)
				grasp_point_y = np.array(grasp_point_y)
				ram=np.random.randint(len(grasp_point_x))
				grasp_point.append(grasp_point_x[ram])
				grasp_point.append(grasp_point_y[ram])
				#计算角度
				if (box[1,0]-box[0,0])==0:
					theta =np.pi/2
				else:
					theta=np.arctan((box[1,1]-box[0,1])/(box[1,0]-box[0,0]))  #用长边算角度
				grasp_angle.append(theta)

				grasp_w.append(l2)
			else:
				#计算长边中线上的点
				dx=(box[2,0]-box[1,0])/k1
				dy=(box[2,1]-box[1,1])/k1
				grasp_point_x=[]
				grasp_point_y = []
				for j in range(1,k1):
					grasp_point_x.append((box[1,0]+box[0,0])/2+j*dx)
					grasp_point_y.append((box[1,1]+box[0,1])/2+j*dy)
				grasp_point_x=np.array(grasp_point_x)
				grasp_point_y = np.array(grasp_point_y)
				ram=np.random.randint(len(grasp_point_x))
				grasp_point.append(grasp_point_x[ram])
				grasp_point.append(grasp_point_y[ram])

				#计算角度
				if (box[2, 0] - box[1, 0])==0:
					theta =np.pi/2
				else:
					theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))
				grasp_angle.append(theta)

				grasp_w.append(l1)

			# if np.absolute(w-h)<=30:
			# 	#####短边的参数
			# 	if l1 < l2:  # 现在用短边进行计算
			# 		# 计算长边中线上的点
			# 		dx = (box[1, 0] - box[0, 0]) / k2
			# 		dy = (box[1, 1] - box[0, 1]) / k2
			# 		grasp_point_x = []
			# 		grasp_point_y = []
			# 		for j in range(1, k2):
			# 			grasp_point_x.append((box[1, 0] + box[2, 0]) / 2 - j * dx)
			# 			grasp_point_y.append((box[1, 1] + box[2, 1]) / 2 - j * dy)
			# 		grasp_point_x = np.array(grasp_point_x)
			# 		grasp_point_y = np.array(grasp_point_y)
			# 		ram = np.random.randint(len(grasp_point_x))
			# 		grasp_point.append(grasp_point_x[ram])
			# 		grasp_point.append(grasp_point_y[ram])
			#
			# 		# 计算角度
			# 		if (box[1, 0] - box[0, 0]) == 0:
			# 			theta = np.pi / 2
			# 		else:
			# 			theta = np.arctan((box[1, 1] - box[0, 1]) / (box[1, 0] - box[0, 0]))  # 用长边算角度
			# 		grasp_angle.append(theta)
			#
			# 		grasp_w.append(l2)
			# 	else:
			# 		# 计算长边中线上的点
			# 		dx = (box[2, 0] - box[1, 0]) / k2
			# 		dy = (box[2, 1] - box[1, 1]) / k2
			# 		grasp_point_x = []
			# 		grasp_point_y = []
			# 		for j in range(1, k2):
			# 			grasp_point_x.append((box[1, 0] + box[0, 0]) / 2 + j * dx)
			# 			grasp_point_y.append((box[1, 1] + box[0, 1]) / 2 + j * dy)
			# 		grasp_point_x = np.array(grasp_point_x)
			# 		grasp_point_y = np.array(grasp_point_y)
			# 		ram = np.random.randint(len(grasp_point_x))
			# 		grasp_point.append(grasp_point_x[ram])
			# 		grasp_point.append(grasp_point_y[ram])
			#
			# 		# 计算角度
			# 		if (box[2, 0] - box[1, 0]) == 0:
			# 			theta = np.pi / 2
			# 		else:
			# 			theta = np.arctan((box[2, 1] - box[1, 1]) / (box[2, 0] - box[1, 0]))
			# 		grasp_angle.append(theta)
			#
			# 		grasp_w.append(l1)
			# else:
			# 	pass


	grasp_point=np.array(grasp_point).reshape(-1,2)
	grasp_angle=np.array(grasp_angle)
	grasp_w=np.array(grasp_w)

	angle_random=(np.pi/180)*rd_ang*(2*np.random.rand(len(grasp_angle))-1)  #随机增加+-5度范围内随机数
	grasp_angle=grasp_angle+angle_random

	# b = np.random.randint(len(grasp_angle)) #设置一个随机数
	# grasp_point=grasp_point[b]
	# grasp_angle=grasp_angle[b]
	# grasp_w=grasp_w[b]


	grasp_w[grasp_w < 1000] = 0.03
	grasp_w[(grasp_w > 1000) & (grasp_w < 4000)] = 0.045
	grasp_w[(grasp_w > 4000) & (grasp_w < 8000)] = 0.07
	grasp_w[(grasp_w > 8000)&(grasp_w < 13000)] = 0.085
	grasp_w[(grasp_w > 13000) & (grasp_w < 25000)] = 0.10
	grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.13
	grasp_w[(grasp_w > 35000) & (grasp_w < 50000)] = 0.15
	grasp_w[(grasp_w > 50000) & (grasp_w < 70000)] = 0.19
	grasp_w[(grasp_w > 70000)] = 0.23

	# a=0.03
	# grasp_w[grasp_w < 1000] = 0.03
	# grasp_w[(grasp_w > 1000) & (grasp_w < 4000)] = 0.045+a
	# grasp_w[(grasp_w > 4000) & (grasp_w < 8000)] = 0.06
	# grasp_w[(grasp_w > 8000)&(grasp_w < 13000)] = 0.08
	# grasp_w[(grasp_w > 13000) & (grasp_w < 25000)] = 0.10
	# grasp_w[(grasp_w > 25000) & (grasp_w < 35000)] = 0.13
	# grasp_w[(grasp_w > 35000) & (grasp_w < 50000)] = 0.15
	# grasp_w[(grasp_w > 50000) & (grasp_w < 70000)] = 0.19
	# grasp_w[(grasp_w > 70000)] = 0.23



	return grasp_point,grasp_angle,grasp_w
