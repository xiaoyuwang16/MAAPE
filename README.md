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

## Usage
1. Generate Embeddings
```python
from example._1_generate_embeddings import main as generate_embeddings
generate_embeddings()
```

3. Generate Paths
```python
from examples._2_generate_paths import main as generate_paths
generate_paths()
```

4. Calculate Weights and Edges
```python
from examples._3_calculate_weights_and_edges import main as calculate_weights
calculate_weights()
```

5. Build and Analyze Graph
```python
from examples._4_build_and_analyze_graph import main as analyze_graph
analyze_graph()
```

6. Visualize Results
```python
from examples._5_visualize_maape import main as visualize
visualize()
```

## Data Format Requirements

Input files:\
Protein sequences in FASTA format\
Order information in text format (for visualization)

Output:\
Embedding files (.npy)\
Path information (.pkl)\
Edge weights and graph structure (.pkl, .txt)\
Visualization plots


