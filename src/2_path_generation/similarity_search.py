import numpy as np
import faiss
from tqdm import tqdm
from typing import List, Dict, Any

class SimilaritySearch:
    def __init__(self, threshold: float):
        """
        Initialize similarity search.
        
        Args:
            threshold: Similarity threshold
        """
        self.threshold = threshold

    def find_similar_vectors(self, sliced_vectors: np.ndarray, 
                           sliced_indices: List[int], 
                           input_vector_indices: List[Tuple[int, int]], 
                           window_size: int) -> List[Dict[str, Any]]:
        """
        Find similar vectors using FAISS.
        
        Args:
            sliced_vectors: Sliced vectors to search
            sliced_indices: Indices of sliced vectors
            input_vector_indices: Original vector indices
            window_size: Size of the window
            
        Returns:
            List of vector groups
        """
        index = faiss.IndexFlatL2(sliced_vectors.shape[1])
        index.add(sliced_vectors)
        
        vector_groups = []
        processed_vectors = set()

        for i in tqdm(range(len(sliced_vectors)), desc="Finding similar vectors"):
            query_vector = sliced_vectors[i]
            query_vector_hash = hash(query_vector.tobytes())

            if query_vector_hash in processed_vectors:
                continue

            D, I = index.search(np.array([query_vector]), k=100)
            similar_indices = I[0][D[0] <= self.threshold]
            similar_indices = similar_indices[similar_indices != i]

            if len(similar_indices) > 0:
                group_indices = [input_vector_indices[idx] 
                               for idx in similar_indices 
                               if idx < len(input_vector_indices)]
                
                group = {
                    "vector": query_vector,
                    "indices": group_indices
                }
                vector_groups.append(group)
                processed_vectors.add(query_vector_hash)

        return vector_groups