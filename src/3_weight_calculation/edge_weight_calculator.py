import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

class EdgeWeightCalculator:
    def __init__(self, window_sizes: List[int], weights: np.ndarray):
        self.window_sizes = window_sizes
        self.weights = weights
        self.co_occurrence_matrices = {
            window_size: defaultdict(lambda: defaultdict(int)) 
            for window_size in window_sizes
        }
        self.total_co_occurrences = {window_size: 0 for window_size in window_sizes}

    def calculate_edge_weights(self, processed_paths: List[Dict]) -> Tuple[List, Dict]:
        # 计算共现矩阵
        self._calculate_co_occurrences(processed_paths)
        
        # 计算总权重矩阵
        total_matrix = self._calculate_total_weights()
        
        # 生成边数据
        all_edges, all_edges_data = self._generate_edge_data(total_matrix)
        
        return all_edges, all_edges_data

    def _calculate_co_occurrences(self, processed_paths: List[Dict]):
        for item in tqdm(processed_paths, desc="Processing paths"):
            query_indices = [vector[1] for vector in item['query_vectors']]
            result_indices = [vector[1] for vector in item['result_vectors']]
            window_size = item['query_vectors'][0][0]

            for query_index in query_indices:
                for result_index in result_indices:
                    self.co_occurrence_matrices[window_size][query_index][result_index] += 1
                    self.total_co_occurrences[window_size] += 1

    def _calculate_total_weights(self) -> Dict:
        total_matrix = {}
        for window_size, co_matrix in tqdm(self.co_occurrence_matrices.items(), 
                                         desc="Calculating total weight matrix"):
            weight = self.weights[self.window_sizes.index(window_size)]
            
            for query_index, result_dict in co_matrix.items():
                if query_index not in total_matrix:
                    total_matrix[query_index] = {}
                    
                for result_index, count in result_dict.items():
                    weight1 = count * weight
                    weight2 = co_matrix[result_index].get(query_index, 0) * weight
                    
                    # 更新权重
                    if result_index not in total_matrix[query_index]:
                        total_matrix[query_index][result_index] = 0
                    total_matrix[query_index][result_index] += weight1
                    
                    if result_index not in total_matrix:
                        total_matrix[result_index] = {}
                    if query_index not in total_matrix[result_index]:
                        total_matrix[result_index][query_index] = 0
                    total_matrix[result_index][query_index] += weight2
                    
        return total_matrix

    def _generate_edge_data(self, total_matrix: Dict) -> Tuple[List, Dict]:
        all_edges = []
        all_edges_data = {}
        
        for query_index in tqdm(total_matrix, desc="Generating edge data"):
            for result_index in total_matrix[query_index]:
                if query_index < result_index:
                    total_weight1 = total_matrix[query_index][result_index]
                    total_weight2 = total_matrix[result_index][query_index]
                    
                    all_edges.append((query_index, result_index, total_weight1, total_weight2))
                    edge_data = self._create_edge_data(query_index, result_index, 
                                                     total_weight1, total_weight2)
                    all_edges_data[(query_index, result_index)] = edge_data
                    
        return all_edges, all_edges_data

    def _create_edge_data(self, query_index: int, result_index: int, 
                         total_weight1: float, total_weight2: float) -> Dict:
        return {
            'total_weight1': total_weight1,
            'total_weight2': total_weight2,
            'window_data': [
                {
                    'window_size': window_size,
                    'count1': self.co_occurrence_matrices[window_size][query_index][result_index],
                    'count2': self.co_occurrence_matrices[window_size][result_index][query_index],
                    'weight': self.weights[self.window_sizes.index(window_size)],
                    'total_count': self.total_co_occurrences[window_size]
                }
                for window_size in self.window_sizes
            ]
        }

    def print_statistics(self, all_edges: List, all_edges_data: Dict):
        print(f"Total number of edges: {len(all_edges)}")
        
        print("\nTotal co-occurrences for each window size:")
        for window_size, total_count in self.total_co_occurrences.items():
            print(f"Window size: {window_size}, Total co-occurrences: {total_count}")
            
        max_edge = max(all_edges_data.items(), 
                      key=lambda x: max(x[1]['total_weight1'], x[1]['total_weight2']))
        min_edge = min(all_edges_data.items(), 
                      key=lambda x: min(x[1]['total_weight1'], x[1]['total_weight2']))
        
        print("\nEdge with maximum weight:")
        self._print_edge_info(max_edge)
        
        print("\nEdge with minimum weight:")
        self._print_edge_info(min_edge)

    def _print_edge_info(self, edge: Tuple):
        print(f"Edge: {edge[0]}")
        print(f"  Total weight1: {edge[1]['total_weight1']:.6f}")
        print(f"  Total weight2: {edge[1]['total_weight2']:.6f}")
        print("  Calculation data:")
        for window_data in edge[1]['window_data']:
            print(f"    Window size: {window_data['window_size']}, "
                  f"Count1: {window_data['count1']}, "
                  f"Count2: {window_data['count2']}, "
                  f"Weight: {window_data['weight']}, "
                  f"Total co-occurrences: {window_data['total_count']}")