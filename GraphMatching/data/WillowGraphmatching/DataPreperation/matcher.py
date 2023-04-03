from typing import Dict

from scipy import spatial
import numpy as np

def compute_similarity(first_descriptors: np.ndarray, second_descriptors: np.ndarray) -> np.ndarray:
    '''
    Computes matrix of cosine similarities between all pairs of descriptors
    :param first_descriptors: (points_count_1, features_count) array with descriptors for the first image
    :param second_descriptors: (points_count_2, features_count) array with descriptors for the first image
    :return: (points_count_1, points_count_2) array with similarities
    '''

    points_count_1 = first_descriptors.shape[0]
    points_count_2 = second_descriptors.shape[0]
    similarity_matrix = np.zeros((points_count_1, points_count_2))
    for i in range(points_count_1):
        for j in range(points_count_2):
            similarity_matrix[i, j] = 1 - spatial.distance.cosine(first_descriptors[i], second_descriptors[j])

    return similarity_matrix

def compute_pair_matchings(first_descriptors: np.ndarray, second_descriptors: np.ndarray) -> np.ndarray:
    '''
    Computes the matches between two pair of points
    :param first_descriptors: (num_points, descriptor_size)
    :param second_descriptors: (num_points, descriptor_size)
    :return: (num_points, 1) array with id of the matching point in the other image for each point. -1 if match is not found
    '''

    excluded_rows = set()
    excluded_columns = set()

    first_points_count = first_descriptors.shape[0]
    second_points_count = second_descriptors.shape[0]

    similarities = compute_similarity(first_descriptors, second_descriptors)

    matches = np.zeros((first_points_count, 1), dtype=int) - 1

    found_match_idx = -1
    iterations_count = min(first_points_count, second_points_count)
    # Cannot match more than the number of points in an image
    for _ in range(iterations_count):
        found_match_idx = -1
        max_row = -1
        max_column = -1
        max_similarity = 0.0
        for row in range(first_points_count):
            if row in excluded_rows:
                continue
            for column in range(second_points_count):
                if column in excluded_columns:
                    continue
                current_similarity = float(similarities[row, column])
                if current_similarity > max_similarity:
                    max_similarity = current_similarity
                    max_row = row
                    max_column = column
        # If the match was found
        if max_column != -1:
            # Exclude the selected points from future matches
            excluded_rows.add(max_row)
            excluded_columns.add(max_column)
            matches[max_row, 0] = max_column

    return matches


def compute_all_pair_matchings(all_descriptors: Dict[str, np.ndarray]):
    '''
    Computes matches between all keypoint descriptors
    :param all_descriptors: Map from image names to descriptors
    :return: (sum_points, num_images) array of integers with id of matching point in each image for each point
             -1 if match has not been found
    '''

    # Total number of points
    num_points = sum(descriptor.shape[0] for descriptor in all_descriptors.values())
    # Total number of images
    num_images = len(all_descriptors)

    image_names = list(sorted(all_descriptors))
    all_image_matches = []
    for first_image_idx, first_image_name in enumerate(image_names):
        first_image_matches = []
        for second_image_idx, second_image_name in enumerate(image_names):
            first_image_descriptors = all_descriptors[first_image_name]
            second_image_descriptors = all_descriptors[second_image_name]

            current_matches = compute_pair_matchings(first_image_descriptors, second_image_descriptors)
            first_image_matches.append(current_matches)
        first_image_matches = np.concatenate(first_image_matches, axis=1)
        all_image_matches.append(first_image_matches)
    all_image_matches = np.concatenate(all_image_matches, axis=0)

    return all_image_matches

def compute_pair_similarities(all_descriptors: Dict[str, np.ndarray]):
    '''
    Computes similarities on pairs of successive images
    :param all_descriptors:
    :return:
    '''

    image_names = list(sorted(all_descriptors))
    for idx in range(len(image_names) - 1):
        current_descriptors = all_descriptors[image_names[idx]]
        next_descriptors = all_descriptors[image_names[idx + 1]]

        similarities = compute_similarity(current_descriptors, next_descriptors)
        diagonal_over_sum = float(similarities.diagonal().sum() / similarities.sum())

        print(f"- Similarity {image_names[idx]}|{image_names[idx + 1]}")
        print(f"Diagonal over sum: {diagonal_over_sum}")
        print(f"{similarities}")