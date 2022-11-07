# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
create train or eval dataset.
"""
import os
import cv2
from mindspore.dataset import vision

import transforms
import numpy as np
import mindspore.dataset as ds

# Per-channel mean and standard deviation values on ImageNet (in RGB order)
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/imagenet.lua
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Constants for lighting normalization on ImageNet (in RGB order)
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/imagenet.lua
_EIG_VALS = [[0.2175, 0.0188, 0.0045]]
_EIG_VECS = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]


class ImageNet:
    def __init__(self, data_path, do_train=True, image_size=224, scale_size=256, pca_std=0.1):
        self.data_path = data_path
        path_list = []
        data_num = 0
        classes = os.listdir(data_path)
        class2idx = {}
        idx = 0
        for i in classes:
            class_images = os.listdir(os.path.join(data_path, i))
            data_num += len(class_images)
            path_list += [(i, j) for j in class_images]
            class2idx[i] = idx
            idx += 1
        self.idx2class = classes
        self.class2idx = class2idx
        self.data_num = data_num
        self.image_paths = path_list
        self.do_train = do_train
        self.image_size = image_size
        self.scale_size = scale_size
        self.pca_std = pca_std

    # def _prepare_im(self, im):
    #     """Prepares the image for network input (HWC/BGR/int -> CHW/BGR/float)."""
    #     # Convert HWC/BGR/int to HWC/RGB/float format for applying transforms
    #     im = im[:, :, ::-1].astype(np.float32) / 255
    #     # Train and test setups differ
    #     train_size, test_size = self.image_size, self.scale_size
    #     if self.do_train:
    #         # For training use random_sized_crop, horizontal_flip, augment, lighting
    #         im = transforms.random_sized_crop(im, train_size)
    #         im = transforms.horizontal_flip(im, prob=0.5)
    #         im = transforms.lighting(im, self.pca_std, _EIG_VALS, _EIG_VECS)
    #     else:
    #         # For testing use scale and center crop
    #         im = transforms.scale_and_center_crop(im, test_size, train_size)
    #     # For training and testing use color normalization
    #     im = transforms.color_norm(im, _MEAN, _STD)
    #     # Convert HWC/RGB/float to CHW/BGR/float format
    #     im = np.ascontiguousarray(im[:, :, ::-1].transpose([2, 0, 1]))
    #     return im

    def _prepare_im(self, im):
        """Prepares the image for network input (HWC/BGR/int -> CHW/BGR/float)."""
        # Convert HWC/BGR/int to HWC/RGB/float format for applying transforms
        im = im[:, :, ::-1].astype(np.float32) / 255
        if self.do_train:
            # For training use random_sized_crop, horizontal_flip, augment, lighting
            im = transforms.lighting(im, self.pca_std, _EIG_VALS, _EIG_VECS)
        return im

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_paths[index][0], self.image_paths[index][1])
        im = cv2.imread(image_path)
        im = self._prepare_im(im)
        label = self.image_paths[index][0]
        return im, self.class2idx[label]

    def __len__(self):
        return self.data_num


def create_dataset(data_path, do_train=True, repeat_num=1, batch_size=128, rank_id=0, rank_size=1, image_size=224,
                   scale_size=256, pca_std=0.1, num_worker=8):
    """
    create a train or eval imagenet2012 dataset for Darknet53

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: GPU
        distribute(bool): data for distribute or not. Default: False

    Returns:
        dataset for ImageNet
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    pca_std = 0.1
    # image_net = ImageNet(data_path, do_train, image_size, scale_size, pca_std)
    data_set = ds.ImageFolderDataset(data_path, shuffle=True,
                                     num_shards=rank_size, shard_id=rank_id)
    # if rank_size == 1:
    #     data_set = ds.GeneratorDataset(image_net, ["image", "label"],
    #                                    shuffle=True,
    #                                    num_parallel_workers=num_worker)
    # else:
    #     data_set = ds.GeneratorDataset(image_net, ["image", "label"],
    #                                    shuffle=False,
    #                                    num_parallel_workers=num_worker,
    #                                    num_shards=rank_size,
    #                                    shard_id=rank_id)
    if do_train:
        trans_a = [
            # maybe different from random_sized_crop
            vision.RandomCropDecodeResize(image_size),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Rescale(1.0 / 255.0, 0.0),
        ]
        trans_b = [
            vision.Normalize(mean=mean, std=std),
            vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR),
            vision.HWC2CHW()
        ]
        data_set = data_set.map(operations=trans_a, input_columns="image", num_parallel_workers=num_worker)
        data_set = data_set.map(input_columns="image",
                                output_columns="image",
                                operations=lambda x: transforms.lighting(x, pca_std, _EIG_VALS, _EIG_VECS),
                                num_parallel_workers=num_worker)
        data_set = data_set.map(operations=trans_b, input_columns="image", num_parallel_workers=num_worker)
    else:
        trans = [
            vision.Decode(),
            vision.Resize(scale_size),
            vision.CenterCrop(image_size),
            vision.Rescale(1.0 / 255.0, 0.0),
            vision.Normalize(mean=mean, std=std),
            vision.ConvertColor(vision.ConvertMode.COLOR_RGB2BGR),
            vision.HWC2CHW()
        ]
        data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=num_worker)

    # type_cast_op = vision.ToType(mstype.int32)

    # data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def np_create_dataset(data_path, do_train=True, repeat_num=1, batch_size=128, rank_id=0, rank_size=1, image_size=224,
                      scale_size=256, pca_std=0.1, num_worker=8):
    """
    create a train or eval imagenet2012 dataset for Darknet53

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32
        target(str): the device target. Default: GPU
        distribute(bool): data for distribute or not. Default: False

    Returns:
        dataset for ImageNet
    """
    image_net = ImageNet(data_path, do_train, image_size, scale_size, pca_std)
    if rank_size == 1:
        data_set = ds.GeneratorDataset(image_net, ["image", "label"],
                                       shuffle=True,
                                       num_parallel_workers=num_worker)
    else:
        data_set = ds.GeneratorDataset(image_net, ["image", "label"],
                                       shuffle=False,
                                       num_parallel_workers=num_worker,
                                       num_shards=rank_size,
                                       shard_id=rank_id)

    # type_cast_op = vision.ToType(mstype.int32)

    # data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
