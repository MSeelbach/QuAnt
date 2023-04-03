from typing import List, Tuple

import PIL
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL.Image import Image
from torchvision import models
from torchvision import transforms
import numpy as np

from features_drawer import FeatureDrawer


class KeypointsViewer:

    def __init__(self, image_path: str, annotations_path: str):
        '''
        Builds an annotator for the given image
        :param image_path:
        '''

        self.keypoints = self.read_annotation(annotations_path)

        self.colors_by_id = {
            0: (255, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255)
        }

        self.image = cv2.imread(image_path)
        cv2.namedWindow('image')

        for idx, (x, y, object_id) in enumerate(self.keypoints):

            cv2.circle(self.image, (x, y), 5, self.colors_by_id[object_id], -1)
            cv2.putText(self.image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors_by_id[object_id], 2, cv2.LINE_AA)

    def show(self):
        '''

        :return: True if the annotation was successful
        '''

        cv2.imshow('image', self.image)
        cv2.waitKey(0)

    def read_annotation(self, filename: str):
        '''
        Reads keypoints from the given file
        :param filename:
        :return:
        '''

        current_keypoints = pd.read_csv(filename, header=None)

        all_keypoints = []
        for index, row in current_keypoints.iterrows():
            all_keypoints.append((row[0], row[1], row[2]))

        return all_keypoints





