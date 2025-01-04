# Modular Assembly Analysis of Protein Embeddings (MAAPE) algorithm

A novel algorithm that integrates a k-nearest neighbour (KNN) similarity network with co-occurrence matrix analysis to extract evolutionary insights from protein language model (PLM) embeddings.
This code implements a five-step pipeline for analyzing protein sequences and constructing directed similarity networks:

1. Embedding Generation\
Processes input protein sequences\
Utilizes the ESM-2 language model to generate sequence embeddings\
Creates high-dimensional vector representations of protein sequences
2. Path Generation\
Segments the embeddings into smaller sub-vectors\
Identifies assembly paths between these sub-vectors\
Maps potential connections between sequence segments
3. Weight and Edge Calculation\
Computes co-assembly relationships between input sequences based on identified paths\
Determines edge directions between sequence pairs\
Calculates edge weights based on sequence relationships\
Generates a weighted, directed edge list

4. Builds a K-nearest neighbor (KNN) similarity network\
Maps previously calculated directions and weights onto KNN edges\
Creates a structured representation of sequence relationships
5. Visualization\
Generates the final MAAPE (Molecular Assembly And Protein Engineering) network
6. Aggregated visualization\
Generates a nodes clustered version of MAAPE graph, to gain condensed version of protein evolution relationships.


## Features
- ESM2-based protein sequence embedding
- Multi-scale sliding window analysis
- Co-occurrence matrix construction
- KNN similarity network construction
- Evolutionary direction detection
- Visualization of protein evolutionary relationships



![graphical](https://github.com/user-attachments/assets/77610421-6d2d-44fb-bcb0-4944b8586c5a)


![MAAPE算法示意图](https://github.com/user-attachments/assets/b36e147d-d28e-4784-9292-de9e3ae33e7a)

##  Requirements
torch,
transformers,
biopython,
numpy,
tqdm,
scikit-learn,
faiss-cpu,
networkx,
matplotlib,
typing-extensions

## Installation
```bash
git clone https://github.com/xiaoyuwang16/MAAPE.git
cd MAAPE
pip install -r requirements.txt
```

## Data Format Requirements

Input files:\
1. Protein sequences in FASTA format, there's a `/path/to/MAAPE/example/test.fasta` in example folder which contains 110 Rubisco protein sequences\
2. Order index file: `/path/to/MAAPE/example/order_index.txt`
   Contains sequence indices and their corresponding protein categories. This information is used for node coloring in the visualization.
3. Similarity threshold file for determining whether sub-vectors of different window sizes are equivalent：`/path/to/MAAPE/example/converted_thresholds_pca.npy`，this file is derived from threshold_window_size_5 = 0.00001, with thresholds for other window sizes converted proportionally using square root scaling.
   
Output:\
Embedding files (.npy), there is a embedding file already L2 normalized and reduced to 110 dimensions: `/path/to/MAAPE/example/output/normalized_pca_embeddings.npy`\
Path information (.pkl)\
Edge weights and graph structure (.pkl, .txt)\
Visualization plots


## Usage
```python
import os
import sys
maape_path = '/path/to/MAAPE' 
sys.path.append(maape_path)

import importlib.util

def import_file(file_path):
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 1. Generate Embeddings
script = import_file('/path/to/MAAPE/example/_1_generate_embeddings.py')
generate_embeddings = script.main
generate_embeddings()

# 2. Generate Paths
script = import_file('/path/to/MAAPE/example/_2_generate_paths.py')
generate_paths = script.main
generate_paths()

# 3. Calculate Weights and Edges
script = import_file('/path/to/MAAPE/example/_3_calculate_weights_and_edges.py')
calculate_weights = script.main
calculate_weights()

# 4. Build and Analyze Graph
script = import_file('/path/to/MAAPE/example/_4_build_and_analyze_graph.py')
build_and_analyze = script.main
build_and_analyze()

# 5. Visualize Results
script = import_file('/path/to/MAAPE/example/_5_visualize_maape.py')
maape_visual = script.main
maape_visual()

# 6. Visualize Aggregated Results
script = import_file('/path/to/MAAPE/example/_6_aggregated_visualization.py')
aggregated_maape = script.main
aggregated_maape()
```
Step 5 & 6 will generate MAAPE graph and its condensed version.

MAAPE graph of ‘/path/to/MAAPE/example/test.fasta’:
![下载](https://github.com/user-attachments/assets/5e1489d7-51e0-4432-8167-75ebf98544d8)
Condensed graph:
![下载 (1)](https://github.com/user-attachments/assets/dcc2c80d-96a2-4f7e-b503-9e086225395f)
