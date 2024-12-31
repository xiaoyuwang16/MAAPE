from src.embedding.esm_embedder import ESMEmbedder
from src.embedding.normalizer import EmbeddingNormalizer
from src.embedding.utils import save_embeddings

def main():
    # Initialize embedder
    embedder = ESMEmbedder()
    
    # Load sequences
    sequences = embedder.load_sequences("path/to/your/fasta/file")
    
    # Generate embeddings
    embeddings = embedder.generate_embeddings(sequences)
    save_embeddings(embeddings, "raw_embeddings.npy")
    
    # Initialize normalizer
    normalizer = EmbeddingNormalizer()
    
    # Normalize and transform embeddings
    normalized_embeddings = normalizer.l2_normalize(embeddings)
    save_embeddings(normalized_embeddings, "normalized_embeddings.npy")
    
    # PCA transformation
    pca_embeddings = normalizer.pca_transform(normalized_embeddings)
    save_embeddings(pca_embeddings, "normalized_pca_embeddings.npy")

if __name__ == "__main__":
    main()