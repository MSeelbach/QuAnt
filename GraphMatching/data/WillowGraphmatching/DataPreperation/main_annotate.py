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

from patched_descriptor_extractor import PatchedDescriptorExtractor


def main():

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=r'Cars_000a.png'  )# required=True)
    parser.add_argument("--objects_count", type=int, default= 2)#required=True)
    arguments = parser.parse_args()
    image = arguments.image
    objects_count = arguments.objects_count

    print(f"Annotating image for {objects_count} objects")

    annotator = Annotator(image)
    for current_object_idx in range(objects_count):
        annotation_result = annotator.interactive_annotations(current_object_idx)

        if not annotation_result:
            print("- Annotations discarded")
            exit(1)

    output_filename = image[:-4] + ".csv"
    annotator.save(output_filename)
    print(f"- Annotations saved to {output_filename}")


if __name__ == "__main__":

    main()