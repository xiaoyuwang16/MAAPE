import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

class PathFinder:
    def __init__(self, thresholds: np.ndarray):
        """
        Initialize path finder.
        
        Args:
            thresholds: Array of thresholds for different window sizes
        """
        self.thresholds = thresholds

    def split_vectors(self, vectors: List[Tuple], window_size: int, step_size: int) -> List[Tuple]:
        """
        Split vectors into smaller windows.
        """
        split_vectors = []
        for vector, index, similar_indices in vectors:
            for start_index in range(0, len(vector) - window_size + 1, step_size):
                split_vector = vector[start_index:start_index+window_size]
                split_index = (index[0], index[1], index[2] + start_index)
                split_similar_indices = [(si[0], si[1], si[2] + start_index) 
                                       for si in similar_indices]
                split_vectors.append((split_vector, split_index, split_similar_indices))
        return split_vectors

    def find_generation_path(self, search_results: Dict) -> List[List]:
        """
        Find vector generation paths.
        
        Args:
            search_results: Dictionary of search results
            
        Returns:
            List of paths
        """
        window_sizes = sorted(search_results.keys())
        split_vectors_by_window = self._prepare_split_vectors(search_results, window_sizes)
        return self._find_paths(search_results, split_vectors_by_window, window_sizes)

    def _prepare_split_vectors(self, search_results: Dict, window_sizes: List[int]) -> Dict:
        """Prepare split vectors for each window size."""
        split_vectors_by_window = {}
        for i in range(1, len(window_sizes)):
            current_window_size = window_sizes[i]
            prev_window_size = window_sizes[i-1]
            split_vectors_current = []
            for vector, index, similar_indices in search_results[current_window_size]:
                split_vectors_current.extend(
                    self.split_vectors([(vector, index, similar_indices)], 
                                     prev_window_size, 1))
            split_vectors_by_window[current_window_size] = split_vectors_current
        return split_vectors_by_window

    def _find_paths(self, search_results: Dict, 
                   split_vectors_by_window: Dict, 
                   window_sizes: List[int]) -> List[List]:
        """Find paths between vectors of different window sizes."""
        all_paths = []
        for i in tqdm(range(len(window_sizes) - 1), desc="Window Size"):
            current_window_size = window_sizes[i]
            next_window_size = window_sizes[i+1]
            paths = self._process_window_size(
                search_results[current_window_size],
                split_vectors_by_window[next_window_size],
                self.thresholds[i],
                current_window_size
            )
            all_paths.extend(paths)
        return all_paths

    def _process_window_size(self, current_vectors: List, 
                           next_split_vectors: List, 
                           threshold: float,
                           window_size: int) -> List[List]:
        """Process vectors for a specific window size."""
        paths = [[] for _ in range(len(current_vectors))]
        
        for vector_index, (vector, index, similar_indices) in enumerate(current_vectors):
            query_info = (vector, index, similar_indices)
            
            for split_vector, next_index, next_similar_indices in next_split_vectors:
                if index[1:] == next_index[1:]:
                    distance = np.linalg.norm(vector - split_vector)
                    if distance <= threshold:
                        path_info = (split_vector, next_index, next_similar_indices)
                        paths[vector_index].append((query_info, path_info))
                        
            if (vector_index + 1) % 1000 == 0:
                self._print_progress(vector_index + 1, len(current_vectors), window_size)
                
        return paths

    @staticmethod
    def _print_progress(current: int, total: int, window_size: int):
        """Print progress bar."""
        progress = current / total
        print(f"\rProcessing Window Size {window_size}: "
              f"[{int(progress * 50) * '='}"
              f">{(50 - int(progress * 50)) * ' '}] "
              f"{int(progress * 100)}%", end="", flush=True)