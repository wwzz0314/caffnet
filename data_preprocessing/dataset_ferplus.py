import torch.utils.data as data
import cv2
import pandas as pd
import os
# import image_utils
import random
import cv2
import numpy as np


class FerPlusSet(data.Dataset):
    def __init__(self, path, train=True, transform=None, basic_aug=False):
        self.train = train
        self.root = path
        self.transform = transform
        self.train_paths = os.path.join(self.root, 'train/label/label.csv')
        self.val_paths = os.path.join(self.root, 'test/label/label.csv')
        # 0:Neutral, 1:Happy, 2:Sad, 3:Surprise, 4:Fear, 5:Disgust, 6:Anger, 7:Contempt

        df_train = pd.read_csv(self.train_paths, sep=',', header=None)
        df_val = pd.read_csv(self.val_paths, sep=',', header=None)

        if self.train:
            dataset = df_train
        else:
            dataset = df_val

        file_names = dataset.iloc[:, 0].values
        self.target = np.argmax(dataset.iloc[:, 2:10].values, axis=1)

        self.file_paths = []
        self.feature_paths = []
        for f in file_names:
            if self.train:
                path = os.path.join(self.root, 'train/image', f)
                feature_path = os.path.join(self.root, 'train/feature', f.replace('.png', '_feature.jpg'))
            else:
                path = os.path.join(self.root, 'test/image', f)
                feature_path = os.path.join(self.root, 'test/feature', f.replace('.png', '_feature.jpg'))
            self.file_paths.append(path)
            self.feature_paths.append(feature_path)
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

        # sample = sample[:, :, ::-1]  # BGR to RGB
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
    test = FerPlusSet(path='../dataset/FERPlus', train=False)
    test.__getitem__(1)
