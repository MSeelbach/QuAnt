from typing import List, Tuple

import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL.Image import Image
from torchvision import models
from torchvision import transforms
import numpy as np

from features_drawer import FeatureDrawer


class PatchedDescriptorExtractor:

    def __init__(self):

        # Pretrained AlexNet on ImageNet
        self.alexnet = models.alexnet(pretrained=True).cuda()

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Will contain extracted features
        self.conv4_features = None
        self.conv5_features = None

        pooling = nn.AdaptiveAvgPool2d((1,1))
        # Creates forward hooks
        def conv4_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            height = output.size(-2)
            width = height
            self.conv4_features = output.clone()[:, :, height // 2, width // 2]

        def conv5_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            height = output.size(-2)
            width = height
            self.conv5_features = output.clone()[:, :, height // 2, width // 2]

        # Registers forward hooks
        self.alexnet.features[8].register_forward_hook(conv4_hook)
        self.alexnet.features[10].register_forward_hook(conv5_hook)

        self.input_res = 224
        self.padding = self.input_res // 2

    def extract_descriptors(self, image: Image, keypoints: List[Tuple[float, float]], target_height: int) -> np.ndarray:
        '''

        :param image: image from which to extract the keypoints
        :param keypoints: List of (x, y) coordinates in image where to extract descriptors
        :param target_height: target image height in which to extract the features. Ignored if <= 0
        :return:
        '''

        with torch.no_grad():

            width, height = image.size

            rescale_factor = 1.0
            if target_height > 0.0 and target_height != height:
                rescale_factor = target_height / height
                target_width = rescale_factor * width
                target_width = round(target_width)
                image = image.resize((target_width, target_height), resample=PIL.Image.BILINEAR)

            # Extracts the new image widht and height
            width, height = image.size

            # Transforms the keypoints into tensors and normalizes them in [0, 1]
            keypoints = torch.tensor(keypoints, device="cuda:0", dtype=torch.float)
            keypoints *= rescale_factor  # Rescales the keypoints based on the target resolution
            # Transforms the keypoints to ints
            keypoints = keypoints.type(torch.long)

            # (1, 3, height, width)
            image_tensor = self.transforms(image).unsqueeze(0).cuda()
            padded_image_tensor = F.pad(image_tensor, (self.padding, self.padding, self.padding, self.padding), mode="reflect")

            all_patches = []
            for current_keypoint in keypoints:
                current_keypoint_col = current_keypoint[0].item() + self.padding
                current_keypoint_row = current_keypoint[1].item() + self.padding

                current_patch = padded_image_tensor[:, :, current_keypoint_row - self.padding: current_keypoint_row + self.padding, current_keypoint_col - self.padding: current_keypoint_col + self.padding]
                all_patches.append(current_patch)
            all_patches = torch.cat(all_patches, dim=0)

            # Forwards the image, hooks will save the features
            self.alexnet(all_patches)

            # Concatenates the features
            extracted_features = torch.cat([self.conv4_features, self.conv5_features], dim=-1)

            return extracted_features.cpu().numpy()

        return sampled_features.cpu().numpy()





