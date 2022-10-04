import cv2
import numpy as np


def draw_mask(img_path):
    ##读入图片
    img_rgb=cv2.imread(img_path,1)
    img_rgb[img_rgb!=0]=255

    img=cv2.imread(img_path,0)  #读入灰度值

    ##读入所有物体编号
    obj=np.unique(img)
    obj=obj[obj!=0]  #不要背景的分类

    ##大循环
    thresh1 = 40
    thresh2 = 600
    for i in obj:
        # 通过值获取索引
        point = np.argwhere(img == i)
        point = point[:, [1, 0]]  # 这里需要交换两列

        # 拟合旋转矩形
        rect = cv2.minAreaRect(point)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        ((x, y), (w, h), a) = rect
        #设置阈值
        if ((w > thresh1) & (h > thresh1) & (w < thresh2) & (h < thresh2)):
            box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
            box = np.int0(box)

            # 画出来
            cv2.drawContours(img_rgb, [box], 0, (255, 0, 255), 3)  # 画出旋转矩形
            box_center = (int(rect[0][0]), int(rect[0][1]))  # 矩形圆心坐标
            cv2.circle(img_rgb, box_center, 7, (255, 0, 0), -1)  # 图上画圆
        else:
            continue


    # 展示图像
    cv2.imshow("Image", img_rgb)
    cv2.waitKey(0)



#预测时用于显示拟合效果
def draw_mask_pred_show(pred_rgb,pred_mask):  #pred_rgb增加了不同颜色;pred_mask原来的预测结果
    ##读入所有物体编号
    obj=np.unique(pred_mask)
    obj=obj[obj!=0]  #不要背景的分类

    ##大循环
    for i in obj:
        # 通过值获取索引
        point = np.argwhere(pred_mask == i)
        point = point[:, [1, 0]]  # 这里需要交换两列

        # 拟合旋转矩形
        rect = cv2.minAreaRect(point)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
        ((x, y), (w, h), a) = rect

        box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
        box = np.int0(box)

        # 画出来
        cv2.drawContours(pred_rgb, [box], 0, (255, 255, 0), 1)  # 画出旋转矩形
        box_center = (int(rect[0][0]), int(rect[0][1]))  # 矩形圆心坐标
        cv2.circle(pred_rgb, box_center, 2, (255, 0, 0), -1)  # 图上画圆

    # #存储这张图
    # cv2.imwrite('img/pred_mask.png',pred_rgb)
    #
    # return cv2.imread('img/pred_mask.png')

    return pred_rgb

