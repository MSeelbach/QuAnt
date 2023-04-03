import argparse
import os
import pickle
from multiprocessing.pool import Pool
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

from descriptor_extractor import DescriptorExtractor
from matcher import compute_similarity, compute_all_pair_matchings, compute_pair_similarities

from patched_descriptor_extractor import PatchedDescriptorExtractor
from splitter import Splitter

def do_split(args):

    input_directory, output_directory, repeats, split_arguments = args

    for idx in range(repeats):

        str_splits = [str(split) for split in split_arguments[1]]  # Number of points per motion in string format
        qbits = split_arguments[0] * sum(split_arguments[1]) * len(split_arguments[1])  # Number of qbits needed to encode each problem
        current_output_directory = os.path.join(output_directory, f"{split_arguments[0]}-{','.join(str_splits)}-{qbits}",  f"{idx:05d}")

        # Tries to split until it finds a good split
        success = False
        while not success:
            try:
                splitter = Splitter.load(input_directory)
                splitter.split(*split_arguments)
                splitter.save(current_output_directory, input_directory)
            except Exception as e:
                pass
            else:
                success = True

def main():

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, required=True)
    parser.add_argument("--output_directory", type=str, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()

    # Directory where to look for images and files
    input_directory = arguments.input_directory
    output_directory = arguments.output_directory
    repeats = arguments.repeats

    split_lists = (
        (3, [10, 10]),
        (3, [8, 8]),
        (3, [6, 6]),
        (3, [4, 4]),
        (4, [10, 10]),
        (4, [8, 8]),
        (4, [6, 6]),
        (4, [4, 4]),
        (3, [8, 8, 8]),
        (3, [6, 6, 6]),
        (3, [4, 4, 4]),

        (3, [10, 11]),
        (3, [8, 9]),
        (3, [6, 7]),
        (3, [4, 5]),
        (4, [10, 11]),
        (4, [8, 9]),
        (4, [6, 7]),
        (4, [4, 5]),
        (3, [8, 9, 10]),
        (3, [6, 7, 8]),
        (3, [4, 5, 6]),
    #)

    #split_lists = (
        (5, [7, 7]),
        (5, [7, 8]),
        (5, [8, 8]),
        (5, [8, 9]),
        (5, [9, 9]),
        (5, [9, 10]),
        (5, [10, 10]),
        (5, [10, 11]),
        (5, [11, 11]),

    )

    work_items = [(input_directory, output_directory, repeats, split_list) for split_list in split_lists]


    pool = Pool()
    pool.map(do_split, work_items)
    pool.close()




if __name__ == "__main__":

    main()