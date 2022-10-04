import numpy as np
import cv2
import matplotlib.pyplot as plt


def replace_color(img, src_clr, dst_clr):
    ''' 通过矩阵操作颜色替换程序
    @param	img:	图像矩阵
    @param	src_clr:	需要替换的颜色(r,g,b)
    @param	dst_clr:	目标颜色		(r,g,b)
    @return				替换后的图像矩阵
    '''

    img_arr = np.asarray(img, dtype=np.double)

    r_img = img_arr[:, :, 0].copy()
    g_img = img_arr[:, :, 1].copy()
    b_img = img_arr[:, :, 2].copy()

    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2]  # 编码

    r_img[img == src_color] = dst_clr[0]
    g_img[img == src_color] = dst_clr[1]
    b_img[img == src_color] = dst_clr[2]

    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    dst_img = dst_img.transpose(1, 2, 0)

    return dst_img


#参数
def mask_vrep(path):
    img = cv2.imread(path)
    src_clr=img[10,10]   #[207,183,23]
    dst_clr=[0,0,0]

    #替换背景颜色
    img_remove_bg=replace_color(img, src_clr, dst_clr)  #把背景色替换成黑色

    #阈值分割
    gray = cv2.cvtColor(img_remove_bg, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)  # 简单滤波

    #腐蚀膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))
    dilated = cv2.dilate(th,kernel)  #先膨胀后腐蚀
    eroded = cv2.erode(dilated,kernel)

    return eroded



# #显示
# # cv2.imshow('img', img)
# # cv2.imshow('th', th)
# cv2.imshow('eroded', eroded)
#
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()