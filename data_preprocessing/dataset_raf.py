import torch.utils.data as data
import cv2
import pandas as pd
import os
# import image_utils
import random
import cv2
import numpy as np


class RafDataSet(data.Dataset):
    def __init__(self, raf_path, raf_feature_path, train=True, transform=None, basic_aug=False):
        self.train = train
        self.transform = transform
        self.raf_path = raf_path
        self.raf_feature_path = raf_feature_path
        self.raf_path_train = os.path.join(raf_path, 'train')
        self.raf_path_val = os.path.join(raf_path, 'valid')
        self.raf_feature_path_train = os.path.join(raf_feature_path, 'train')
        self.raf_feature_path_val = os.path.join(raf_feature_path, 'valid')
        self.cls_label = os.listdir(self.raf_path_train)
        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        target = []
        file_names = []
        feature_file_names = []
        for cls in self.cls_label:
            if self.train:
                cls_path = os.path.join(self.raf_path_train, cls)
                cls_feature_path = os.path.join(self.raf_feature_path_train, cls)
            else:
                cls_path = os.path.join(self.raf_path_val, cls)
                cls_feature_path = os.path.join(self.raf_feature_path_val, cls)
            file_one_class = os.listdir(cls_path)
            feature_file_one_class = os.listdir(cls_feature_path)

            length_one_class = len(file_one_class)
            length_feature_one_class = len(feature_file_one_class)

            target.extend([int(cls) for i in range(length_one_class)])

            file_names.extend([os.path.join(cls_path, e) for e in file_one_class])
            feature_file_names.extend([os.path.join(cls_feature_path, e) for e in feature_file_one_class])

        self.target = np.array(target)
        self.file_paths = file_names
        self.feature_paths = feature_file_names

        self.basic_aug = basic_aug
        self.aug_func = [flip_image, add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def get_labels(self):
        return self.target

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        feature_path = self.feature_paths[idx]

        sample = cv2.imread(path)
        feature_sample = cv2.imread(feature_path)

        sample = sample[:, :, ::-1]  # BGR to RGB
        feature_sample = feature_sample[:, :, ::-1]
        target = self.target[idx]

        if self.train:
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                sample = self.aug_func[index](sample)
                feature_sample = self.aug_func[index](sample)

        if self.transform is not None:
            sample = self.transform(sample.copy())
            feature_sample = self.transform(feature_sample.copy())

        return sample, feature_sample, target  # , idx


def add_gaussian_noise(image_array, mean=0.0, var=30):
    std = var ** 0.5
    noisy_img = image_array + np.random.normal(mean, std, image_array.shape)
    noisy_img_clipped = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img_clipped


def flip_image(image_array):
    return cv2.flip(image_array, 1)


if __name__ == "__main__":
    test = RafDataSet(raf_path='../dataset/raf-db', raf_feature_path='../dataset/raf-db-lbphog', train=False)
