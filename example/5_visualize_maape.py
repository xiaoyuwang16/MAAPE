import numpy as np
import pickle
from src.visualization.maape_visualizer import MAAPEVisualizer

def main():
    # Define color scheme
    color_scheme = {
        'Peronosporales': '#FF6B6B',
        'Saprolegniales': '#4ECDC4',
        'Pythiales': '#9B5DE5'
    }

    # Initialize visualizer
    visualizer = MAAPEVisualizer(color_scheme)

    # Load data
    embeddings = np.load('P450_embeddings_normalized_pca.npy')
    sequence_orders = visualizer.read_order_file('order_index.txt')
    
    with open('P450_new_edge_weights_pca.pkl', 'rb') as f:
        edge_weights = pickle.load(f)

    # Build KNN graph
    knn_graph = visualizer.build_knn_graph(embeddings, k=20)
    
    # Prepare edges
    directed_edges = [(edge[0], edge[1]) for edge in edge_weights.keys()]

    # Visualize
    visualizer.visualize(knn_graph, sequence_orders, directed_edges, edge_weights)

if __name__ == "__main__":
    main()