# -- coding: utf-8 --
import os

from LbpHog.GetLBPFeature import local_binary_pattern
from skimage import feature
import cv2
import matplotlib.pyplot as plt


def getFusion(path):
    # 计算 LBP 特征图
    image = cv2.imread(path, 0)
    radius = 1  # LBP 算法的半径值
    n_points = 8  # 领域点的数量
    lbp = local_binary_pattern(image, n_points, radius)
    _, hog = feature.hog(image, orientations=9, block_norm='L1', pixels_per_cell=[9, 9], cells_per_block=[3, 3],
                         visualize=True)
    fusion = lbp + hog
    # # 绘制LBP特征图
    # plt.figure(figsize=(8, 4))
    # plt.subplot(2, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original Image')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 2)
    # plt.imshow(lbp, cmap='gray')
    # plt.title('LBP Feature Map')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(hog, cmap='gray')
    # plt.title('HOG Feature Map')
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 4)
    # plt.imshow(fusion, cmap='gray')
    # plt.title('FUSION Feature Map')
    # plt.axis('off')
    #
    # plt.tight_layout()
    # plt.show()
    return fusion


def read_images_from_file(parent_path):
    image_list = os.listdir(parent_path)
    save_parent_path = '../dataset/FERPlus/train/feature'

    if not os.path.exists(save_parent_path):
        os.makedirs(save_parent_path)

    INDEX = 0
    for file in image_list:
        feature_image = getFusion(os.path.join(parent_path, file))
        save_file_name = file.split('.')[0] + '_feature.jpg'
        save_path = os.path.join(save_parent_path, save_file_name)
        cv2.imwrite(save_path, feature_image)
        INDEX += 1
        print("write image" + save_path)
        print(INDEX/28588)
    return 1


# read_images_from_file('./raf-db/train')
read_images_from_file('../dataset/FERPlus/train/image')
