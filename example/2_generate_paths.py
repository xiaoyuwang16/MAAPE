import pickle
import numpy as np
from src.path_generation.window_generator import WindowGenerator
from src.path_generation.similarity_search import SimilaritySearch
from src.path_generation.path_finder import PathFinder

def main():
    # Load data
    vectors = np.load('P450_embeddings_normalized_pca.npy')
    thresholds = np.load('converted_thresholds_pca.npy')
    
    # Initialize components
    window_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110]
    window_generator = WindowGenerator(window_sizes)
    similarity_search = SimilaritySearch(thresholds[0])
    path_finder = PathFinder(thresholds)

    # Process each window size
    search_results = {}
    for i, window_size in enumerate(window_sizes):
        sliced_vectors, indices, vector_indices = window_generator.create_sliced_vectors(
            vectors, window_size)
        vector_groups = similarity_search.find_similar_vectors(
            sliced_vectors, indices, vector_indices, window_size)
        search_results[window_size] = vector_groups

    # Find paths
    paths = path_finder.find_generation_path(search_results)

    # Save results
    with open("P450_paths_with_similar_indices_pca.pkl", "wb") as f:
        pickle.dump(paths, f)

if __name__ == "__main__":
    main()