import numpy as np
from src.graph_construction.knn_graph_builder import KNNGraphBuilder, GraphVisualizer
from src.graph_construction.weight_extractor import WeightExtractor

def main():
    # Load data
    embeddings = np.load('P450_embeddings_normalized_pca.npy')
    
    # Define parameters
    k = 20
    threshold = 0.5
    color_scheme = {
        'Peronosporales': '#FF6B6B',
        'Saprolegniales': '#4ECDC4',
        'Pythiales': '#9B5DE5'
    }

    # Build KNN graph
    graph_builder = KNNGraphBuilder(embeddings, k, threshold)
    adj_list = graph_builder.build_graph()
    graph_builder.save_edges('P450_knn_graph_edges.txt')

    # Visualize graph
    visualizer = GraphVisualizer(color_scheme)
    sequence_orders = visualizer.read_order_file('order_index.txt')
    visualizer.visualize(adj_list, sequence_orders)

    # Extract weights
    weight_extractor = WeightExtractor('P450_knn_graph_edges.txt', 
                                     'P450_all_edges_pca.pkl')
    new_weights = weight_extractor.extract_weights()
    weight_extractor.save_weights('P450_new_edge_weights_pca.pkl')
    weight_extractor.print_statistics()
    weight_extractor.visualize_weights()

if __name__ == "__main__":
    main()