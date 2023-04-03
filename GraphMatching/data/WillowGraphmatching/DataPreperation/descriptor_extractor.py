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


class DescriptorExtractor:

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

        # Creates forward hooks
        def conv4_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            self.conv4_features = output.clone()

        def conv5_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            self.conv5_features = output.clone()

        # Registers forward hooks
        self.alexnet.features[8].register_forward_hook(conv4_hook)
        self.alexnet.features[10].register_forward_hook(conv5_hook)

    @staticmethod
    def sample_features_at(input_tensor: torch.Tensor, positions: torch.Tensor):
        '''
        Samples directions at the given sample positions

        :param ray_directions: (bs, features, height, width) tensor with ray directions
        :param positions: (bs, samples_per_image, 2) tensor with positions to sample (x, y) from the top left corner normalized in [0, 1]
        :return: (bs, samples_per_image, features) tensor with sampled ray directions
        '''

        # Puts in (batch_size, samples_per_image, 1, 2) format to simulate a 2D tensor of width 1
        positions = positions.unsqueeze(-2)
        # Puts in range [-1, +1] for grid_sample
        positions = (positions - 0.5) * 2

        sampled_input = F.grid_sample(input_tensor, positions, align_corners=True)  # Align corners = True so that directions are not considered as pixels
        # Puts in (batch_size, 3, samples_per_image, 1)
        sampled_input = sampled_input.squeeze(-1)
        # Puts in (batch_size, samples_per_image, features)
        sampled_input = sampled_input.permute([0, 2, 1])

        return sampled_input

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
            keypoints = torch.tensor(keypoints, device="cuda:0", dtype=torch.float).unsqueeze(0)
            keypoints *= rescale_factor  # Rescales the keypoints based on the target resolution
            keypoints[:, :, 0] /= (width - 1)
            keypoints[:, :, 1] /= (height - 1)

            # (1, 3, height, width)
            image_tensor = self.transforms(image).unsqueeze(0).cuda()

            # Forwards the image, hooks will save the features
            self.alexnet(image_tensor)

            # Concatenates the features
            extracted_features = torch.cat([self.conv4_features, self.conv5_features], dim=-3)

            FeatureDrawer.draw_features(extracted_features[0], "test_images/features")

            sampled_features = DescriptorExtractor.sample_features_at(extracted_features, keypoints).squeeze(0)

        return sampled_features.cpu().numpy()





