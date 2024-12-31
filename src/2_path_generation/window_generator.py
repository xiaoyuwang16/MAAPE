import numpy as np
from typing import Tuple, List

class WindowGenerator:
    def __init__(self, window_sizes: List[int], stride: int = 1):
        """
        Initialize window generator.
        
        Args:
            window_sizes: List of window sizes to use
            stride: Stride for window generation
        """
        self.window_sizes = window_sizes
        self.stride = stride

    def generate_windows(self, vector: np.ndarray, window_size: int) -> Tuple[List[np.ndarray], List[int]]:
        """
        Generate windows from a vector.
        
        Args:
            vector: Input vector
            window_size: Size of the window
            
        Returns:
            Tuple of windows and their starting indices
        """
        vector_length = len(vector)
        windows = []
        indices = []
        
        for i in range(0, vector_length - window_size + 1, self.stride):
            window = vector[i:i+window_size]
            windows.append(window)
            indices.append(i)
            
        return windows, indices

    def create_sliced_vectors(self, input_vectors: np.ndarray, window_size: int) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
        """
        Create sliced vectors from input vectors.
        
        Args:
            input_vectors: Input vectors to slice
            window_size: Size of the window
            
        Returns:
            Tuple of sliced vectors, indices, and input vector indices
        """
        sliced_vectors = []
        sliced_indices = []
        input_vector_indices = []

        for index, input_vector in enumerate(input_vectors):
            windows, start_indices = self.generate_windows(input_vector, window_size)
            sliced_vectors.extend(windows)
            for start_index in start_indices:
                sliced_indices.append(start_index)
                input_vector_indices.append((index, start_index))

        return np.array(sliced_vectors), sliced_indices, input_vector_indices