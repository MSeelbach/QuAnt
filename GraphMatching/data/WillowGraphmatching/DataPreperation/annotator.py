from typing import List, Tuple

import PIL
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL.Image import Image
from torchvision import models
from torchvision import transforms
import numpy as np

from features_drawer import FeatureDrawer


class Annotator:

    def __init__(self, image_path: str):
        '''
        Builds an annotator for the given image
        :param image_path:
        '''

        self.annotation_by_id = {}
        self.current_object_id = -1

        self.colors_by_id = {
            0: (255, 0, 0),
            1: (0, 255, 0),
            2: (0, 0, 255)
        }

        def draw_circle(event, x, y, flags, param):
            nonlocal self

            if event == cv2.EVENT_LBUTTONDBLCLK:
                print(f"-- Recorded point ({x}, {y})")
                cv2.putText(self.image, str(len(self.annotation_by_id[self.current_object_id])), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors_by_id[self.current_object_id], 2, cv2.LINE_AA)
                cv2.circle(self.image, (x, y), 2, self.colors_by_id[self.current_object_id], -1)
                self.annotation_by_id[self.current_object_id].append((x, y))

        self.image = cv2.imread(image_path)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_circle)

    def interactive_annotations(self, current_object_id: int):
        '''

        :return: True if the annotation was successful
        '''

        # Registers the current object and creates a new list for its annotations
        self.current_object_id = current_object_id
        self.annotation_by_id[current_object_id] = []

        while True:
            print(f"- Press 'y' to end and confirm current annotations, 'n' to discard them")
            cv2.imshow('image', self.image)
            k = cv2.waitKey(100) & 0xFF
            if k == ord('n'):
                del self.annotation_by_id[current_object_id]
                return False
            elif k == ord('y'):
                return True

    def save(self, filename: str):
        '''
        Saves the annotations to the given file in csv format
        :param filename:
        :return:
        '''

        with open(filename, "w") as file:
            for current_object_id in sorted(self.annotation_by_id):
                current_annotation = self.annotation_by_id[current_object_id]
                for current_annotation_x, current_annotation_y in current_annotation:
                    file.write(f"{current_annotation_x}, {current_annotation_y}, {current_object_id}\n")





