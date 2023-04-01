from tensorflow.keras import utils
import numpy as np
import imgaug.augmenters as iaa
import imgaug as ia
from glob import glob
import os
import cv2
import random as rd

class DataGenerator(utils.Sequence):
    def __init__(self, folder_path, batch_size, img_size, shuffle=True, aug=False):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.img_size = img_size
        
        ################### image augmentation ####################
        self.aug = aug
        seed_list = [rd.randrange(2**32 - 1) for i in range(10)]

        self.seq = iaa.Sequential([ 
            iaa.Fliplr(p=0.5, random_state=seed_list[0]), 
            iaa.Flipud(p=0.5, random_state=seed_list[1]), 
            iaa.Rot90((1, 3), random_state=seed_list[2]), 
            iaa.CropAndPad(percent=(-0.25, 0.25), random_state=seed_list[3]),
            iaa.Dropout(p=(0, 0.2), random_state=seed_list[4]),
            iaa.GammaContrast((0.5, 2.0), random_state=seed_list[5]),
            iaa.RegularGridVoronoi((10, 30), 20, p_drop_points=0.0, p_replace=0.1, max_size=None, random_state=seed_list[6]),
            iaa.ElasticTransformation(alpha=(0, 5.0), sigma=4, random_state=seed_list[7])
        # iaa.BlendAlpha((0.0, 0.1), iaa.contrast.AllChannelsHistogramEqualization(), random_state=seed_list[4]),
        # iaa.Affine(scale=(0.9, 1.1), rotate=(-15, 15), random_state=seed_list[3], cval=255),
        # iaa.PiecewiseAffine(scale=(0.01, 0.05), random_state=seed_list[4], cval=255), 
        ], random_state=seed_list[-1])
        ########################### INPUT FILE ###############################

        self.mask_paths = glob(os.path.join(folder_path, '*_label.tif'))##MASK
        self.img_paths = [p.replace('_label', '') for p in self.mask_paths]##IMG
        
        #######################################################################
        self.indexes = np.arange(len(self.mask_paths))
        self.on_epoch_end()

    def __len__(self):
        # batches per epoch
        return int(np.ceil(len(self.mask_paths) / self.batch_size)) 

    def __getitem__(self, batch_index):
        # Generate indexes of the batch
        idxs = self.indexes[batch_index * self.batch_size: (batch_index+1) * self.batch_size]
        # Find list of IDs
        batch_img_paths = [self.img_paths[i] for i in idxs]
        batch_mask_paths = [self.mask_paths[i] for i in idxs]

        # Generate data
        X, y = self.__data_generation(batch_img_paths, batch_mask_paths)
        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths, mask_paths):
        # Generates data containing batch_size samples
        x = np.empty((len(img_paths), self.img_size, self.img_size, 3), dtype=np.float32)
        y = np.empty((len(img_paths), self.img_size, self.img_size, 1), dtype=np.float32)

        for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            # img and mask preprocess
            img = self.preprocess(img)
            mask = self.preprocess(mask)
            x[i] = img
            y[i] = mask[:, :, :1]
            
        # imgaug: augmentation
        if self.aug:
            x, y = self.seq(images=x, heatmaps=y)
        # Binarize Mask
        y[y>0] = 1. # 0. or 1.
        return x, y

    def preprocess(self, img):
        data = cv2.resize(img, (self.img_size, self.img_size))
        data = data / 255. # normalize img: (0~1), mask: (0 or 1)
        return data
