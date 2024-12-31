# MAAPE (Modular Assembly Analysis of Protein Embeddings)

## Overview
MAAPE is an algorithm designed to extract evolutionary insights from protein language model embeddings. It integrates a Euclidean distance-based KNN similarity network with multiscale co-occurrence matrix analysis to capture both traditional sequence similarities and evolutionary directions.

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