import os
import math
import random
from collections import defaultdict

import torch
import torchvision
import torchvision.transforms as transforms


imagenet_classes = ["T-shirt", "badminton-racket", "baozi", "guitar", "lychee", "others"]
imagenet_templates = ["{}"]


class CustomDataset():

    dataset_dir = 'custom'

    def __init__(self, root, num_shots, preprocess):

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'images')

        train_preprocess = transforms.Compose([
                                                transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                                            ])
        test_preprocess = preprocess

        self.train_dir = os.path.join(self.image_dir, "train")
        self.val_dir = os.path.join(self.image_dir, "val")
        self.test_dir = os.path.join(self.image_dir, "val")

        self.train = torchvision.datasets.ImageFolder(self.train_dir, transform=train_preprocess)
        self.val = torchvision.datasets.ImageFolder(self.val_dir, transform=test_preprocess)
        self.test = torchvision.datasets.ImageFolder(self.test_dir, transform=test_preprocess)
        
        self.template = imagenet_templates
        self.classnames = imagenet_classes

        split_by_label_dict = defaultdict(list)
        for i in range(len(self.train.imgs)):
            split_by_label_dict[self.train.targets[i]].append(self.train.imgs[i])
        imgs = []
        targets = []

        for label, items in split_by_label_dict.items():
            imgs = imgs + random.sample(items, num_shots)
            targets = targets + [label for i in range(num_shots)]
        self.train.imgs = imgs
        self.train.targets = targets
        self.train.samples = imgs