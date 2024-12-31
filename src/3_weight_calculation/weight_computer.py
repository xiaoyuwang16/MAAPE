import numpy as np
import pickle
from typing import Dict, List, Tuple
from pprint import pprint

class WeightComputer:
    def __init__(self, window_sizes: List[int], stride: int = 1):
        self.window_sizes = window_sizes
        self.stride = stride
        self.window_size_info = {}

    def generate_windows(self, vector: np.ndarray, window_size: int) -> Tuple[List[np.ndarray], List[int]]:
        vector_length = len(vector)
        windows = []
        indices = []
        for i in range(0, vector_length - window_size + 1, self.stride):
            window = vector[i:i+window_size]
            windows.append(window)
            indices.append(i)
        return windows, indices

    def count_sliced_vectors(self, vectors: np.ndarray) -> Dict[int, int]:
        sliced_vectors_counts = {}
        for window_size in self.window_sizes:
            total_sliced_vectors = 0
            for vector in vectors:
                sliced_vectors, _ = self.generate_windows(vector, window_size)
                total_sliced_vectors += len(sliced_vectors)
            sliced_vectors_counts[window_size] = total_sliced_vectors
        return sliced_vectors_counts

    def count_results_by_window_size(self, search_results: Dict) -> Dict[int, int]:
        result_counts = {}
        for window_size, filtered_vectors_with_indices in search_results.items():
            result_counts[window_size] = len(filtered_vectors_with_indices)
        return result_counts

    def compute_weights(self, vectors: np.ndarray, search_results_path: str) -> np.ndarray:
        with open(search_results_path, "rb") as f:
            search_results = pickle.load(f)

        sliced_vectors_counts = self.count_sliced_vectors(vectors)
        result_counts = self.count_results_by_window_size(search_results)

        # 计算每个窗口大小的信息
        total_ratio = 0
        weights = []
        for window_size in self.window_sizes:
            self.window_size_info[window_size] = {
                'filtered_vectors': result_counts.get(window_size, 0),
                'total_sliced_vectors': sliced_vectors_counts[window_size]
            }
            
            if sliced_vectors_counts[window_size] > 0:
                ratio = result_counts.get(window_size, 0) / sliced_vectors_counts[window_size]
                total_ratio += ratio
                weights.append(ratio)
            else:
                weights.append(0)

        # 归一化权重
        weights = np.array(weights)
        if total_ratio > 0:
            weights = weights / total_ratio

        return weights

    def print_window_info(self):
        print("Window size information:")
        pprint(self.window_size_info, width=100)