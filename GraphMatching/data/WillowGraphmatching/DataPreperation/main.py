import argparse
import os
import pickle
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


def main():

    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default= r"C:\local\quantummachinelearning\Untitled Folder\data\PascalVOC\TrainVal\VOCdevkit\VOC2011\JPEGImages") #required=True)
    parser.add_argument("--target_height", type=int, default=-1)
    arguments = parser.parse_args()

    # Directory where to look for images and files
    directory = arguments.directory
    target_height = arguments.target_height

    image_filenames = list(sorted(glob.glob(os.path.join(directory, "*.jpg"))))
    keypoint_filenames = list(sorted(glob.glob(os.path.join(directory, "*.csv"))))

    if len(image_filenames) != len(keypoint_filenames):
        raise Exception("The number of images and of keypoint annotation files differs")

    descriptors_by_filename = {}
    keypoints_by_filename = {}
    keypoints_ids_by_filename = {}

    descriptor_extractor = PatchedDescriptorExtractor()

    for current_image_filename, current_keypoint_filename in zip(image_filenames, keypoint_filenames):

        print(f"- Extracting descriptors for {os.path.basename(current_image_filename)}")

        basename = os.path.basename(current_image_filename)[:-4]
        output_filename = os.path.join(directory, basename + ".npy")

        # Reads image and keypoints
        image = Image.open(current_image_filename)
        current_keypoints = pd.read_csv(current_keypoint_filename, header=None)

        all_keypoints = []
        all_keypoints_ids = []
        for index, row in current_keypoints.iterrows():
            all_keypoints.append((row[0], row[1]))
            all_keypoints_ids.append(row[2])

        extracted_descriptors = descriptor_extractor.extract_descriptors(image, all_keypoints, target_height)

        self_similarity = compute_similarity(extracted_descriptors, extracted_descriptors)

        descriptors_by_filename[os.path.basename(current_image_filename)] = extracted_descriptors
        all_keypoints = np.asarray(all_keypoints)
        keypoints_by_filename[os.path.basename(current_image_filename)] = all_keypoints
        all_keypoints_ids = np.asarray(all_keypoints_ids)
        keypoints_ids_by_filename[os.path.basename(current_image_filename)] = all_keypoints_ids
        print(self_similarity)

        with open(output_filename, 'wb') as f:
            np.save(f, extracted_descriptors)

    keypoints_matches = compute_all_pair_matchings(descriptors_by_filename)

    scipy.io.savemat(os.path.join(directory, "descriptors.mat"), {os.path.basename(k).split(".")[0] + "_descriptors": v for k, v in descriptors_by_filename.items()})
    scipy.io.savemat(os.path.join(directory, "keypoints.mat"), {os.path.basename(k).split(".")[0] + "_keypoints": v for k, v in keypoints_by_filename.items()})
    scipy.io.savemat(os.path.join(directory, "keypoints_ids.mat"), {os.path.basename(k).split(".")[0] + "_keypoints_ids": v for k, v in keypoints_ids_by_filename.items()})
    scipy.io.savemat(os.path.join(directory, "keypoint_matches.mat"), {"keypoint_matches": keypoints_matches})

    with open(os.path.join(directory, "descriptors.pkl"), "wb") as file:
        pickle.dump(descriptors_by_filename, file)
    with open(os.path.join(directory, "keypoints.pkl"), "wb") as file:
        pickle.dump(keypoints_by_filename, file)
    with open(os.path.join(directory, "keypoints_ids.pkl"), "wb") as file:
        pickle.dump(keypoints_ids_by_filename, file)
    with open(os.path.join(directory, "keypoint_matches.pkl"), "wb") as file:
        pickle.dump(keypoints_matches, file)

    #compute_pair_similarities(descriptors_by_filename)


if __name__ == "__main__":

    main()