class Config:
    DATA_DIR = "data"
    EMBEDDINGS_FILE = f"{DATA_DIR}/P450_embeddings_normalized_pca.npy"
    PATHS_FILE = f"{DATA_DIR}/P450_paths_with_similar_indices_pca.pkl"
    EDGES_FILE = f"{DATA_DIR}/P450_all_edges_pca.pkl"
    EDGES_DATA_FILE = f"{DATA_DIR}/P450_all_edges_data_pca.pkl"
    KNN_EDGES_FILE = f"{DATA_DIR}/P450_knn_graph_edges.txt"
    NEW_WEIGHTS_FILE = f"{DATA_DIR}/P450_new_edge_weights_pca.pkl"