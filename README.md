# Modular Assembly Analysis of Protein Embeddings (MAAPE) algorithm

A novel algorithm that integrates a k-nearest neighbour (KNN) similarity network with co-occurrence matrix analysis to extract evolutionary insights from protein language model (PLM) embeddings.

![graphical](https://github.com/user-attachments/assets/77610421-6d2d-44fb-bcb0-4944b8586c5a)
![MAAPE算法示意图](https://github.com/user-attachments/assets/b36e147d-d28e-4784-9292-de9e3ae33e7a)



## Features
- ESM2-based protein sequence embedding
- Multi-scale sliding window analysis
- Co-occurrence matrix construction
- KNN similarity network construction
- Evolutionary direction detection
- Visualization of protein evolutionary relationships

## Installation
```bash
git clone https://github.com/yourusername/MAAPE.git
cd MAAPE
pip install -r requirements.txt
```

## Usage
1. Generate Embeddings
```python
from examples.generate_embeddings import main as generate_embeddings
generate_embeddings()
```

3. Generate Paths
```python
from examples.generate_paths import main as generate_paths
generate_paths()
```

4. Calculate Weights and Edges
```python
from examples.calculate_weights_and_edges import main as calculate_weights
calculate_weights()
```

5. Build and Analyze Graph
```python
from examples.build_and_analyze_graph import main as analyze_graph
analyze_graph()
```

6. Visualize Results
```python
from examples.visualize_maape import main as visualize
visualize()
```

## Data Format Requirements

Input files:
Protein sequences in FASTA format
Order information in text format (for visualization)

Output:
Embedding files (.npy)
Path information (.pkl)
Edge weights and graph structure (.pkl, .txt)
Visualization plots


