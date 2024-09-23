# data_processing/data_augmentation.py

import numpy as np
from typing import List, Tuple

def segment_features(features: np.ndarray, segment_length: int) -> List[np.ndarray]:
    T = features.shape[0]
    segments = []
    for start in range(0, T, segment_length):
        end = start + segment_length
        if end <= T:
            segment = features[start:end]
            segments.append(segment)
    return segments

def recombine_segments(segments: List[np.ndarray], num_segments: int) -> np.ndarray:
    selected_indices = np.random.choice(len(segments), num_segments, replace=True)
    new_sequence = np.vstack([segments[i] for i in selected_indices])
    return new_sequence

def augment_features(features: np.ndarray, segment_length: int, num_augmented_samples: int) -> List[np.ndarray]:
    segments = segment_features(features, segment_length)
    num_segments = len(segments)
    augmented_features = []
    for _ in range(num_augmented_samples):
        new_sequence = recombine_segments(segments, num_segments)
        augmented_features.append(new_sequence)
    return augmented_features

def augment_dataset(features_list: List[np.ndarray], labels: List[int], segment_length: int, num_augmented_samples: int) -> Tuple[List[np.ndarray], List[int]]:
    augmented_features_list = []
    augmented_labels = []
    for features, label in zip(features_list, labels):
        augmented_sequences = augment_features(features, segment_length, num_augmented_samples)
        augmented_features_list.extend([features] + augmented_sequences)
        augmented_labels.extend([label] * (1 + num_augmented_samples))
    return augmented_features_list, augmented_labels

