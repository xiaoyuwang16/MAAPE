import numpy as np
import pickle
from src.weight_calculation.weight_computer import WeightComputer
from src.weight_calculation.edge_weight_calculator import EdgeWeightCalculator

def main():
    # 初始化参数
    window_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110]
    
    # 计算权重
    vectors = np.load('P450_embeddings_normalized_pca.npy')
    weight_computer = WeightComputer(window_sizes)
    weights = weight_computer.compute_weights(vectors, 'P450_search_results_all_pca.pkl')
    weight_computer.print_window_info()
    
    # 计算边权重
    with open('P450_processed_paths_pca.pkl', 'rb') as f:
        processed_paths = pickle.load(f)
        
    edge_calculator = EdgeWeightCalculator(window_sizes, weights)
    all_edges, all_edges_data = edge_calculator.calculate_edge_weights(processed_paths)
    edge_calculator.print_statistics(all_edges, all_edges_data)
    
    # 保存结果
    with open('P450_all_edges_data_pca.pkl', 'wb') as f:
        pickle.dump(all_edges_data, f)
    with open('P450_all_edges_pca.pkl', 'wb') as f:
        pickle.dump(all_edges, f)

if __name__ == "__main__":
    main()