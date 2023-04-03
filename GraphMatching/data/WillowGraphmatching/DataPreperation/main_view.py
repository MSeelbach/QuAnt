import argparse
import os
from typing import Dict

import scipy
import scipy.io
import torch
import torchvision
import time
from torchvision import transforms
import glob
import pandas as pd
import numpy as np
from scipy import spatial

from PIL import Image

from annotator import Annotator
from descriptor_extractor import DescriptorExtractor
from keypoints_viewer import KeypointsViewer

from patched_descriptor_extractor import PatchedDescriptorExtractor


def main():

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    arguments = parser.parse_args()
    image = arguments.image

    annotations_filename = image[:-4] + ".csv"
    viewer = KeypointsViewer(image, annotations_filename)
    viewer.show()


if __name__ == "__main__":

    main()