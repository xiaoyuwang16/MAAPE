#embedding and normalization



import tensorflow as tf
print("GPU available:", tf.test.is_gpu_available())
!pip install biopython
from Bio import SeqIO
from transformers import EsmModel, EsmTokenizer
import torch
import numpy as np
from tqdm import tqdm
import time
model_name = "facebook/esm2_t33_650M_UR50D"
model = EsmModel.from_pretrained(model_name)
tokenizer = EsmTokenizer.from_pretrained(model_name)
file_path = "/content/drive/MyDrive/P450/P450_mcleaned.fasta"
sequences = []
for record in SeqIO.parse(file_path, "fasta"):
    sequences.append(str(record.seq))

batch_size = 16
embeddings = []

start_time = time.time()

for i in tqdm(range(0, len(sequences), batch_size), desc="Embedding Progress"):
    batch_sequences = sequences[i:i+batch_size]
    inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    embeddings.append(batch_embeddings)

    elapsed_time = time.time() - start_time
    estimated_total_time = elapsed_time / (i+batch_size) * len(sequences)
    remaining_time = estimated_total_time - elapsed_time
    print(f"Estimated remaining time: {remaining_time:.2f} seconds")

embeddings = np.concatenate(embeddings, axis=0)

print(f"Embedding completed. Total time: {time.time() - start_time:.2f} seconds")

np.save('/content/P450_embeddings.npy', embeddings)
# 定义输入和输出文件路径
input_file = '/content/P450_embeddings.npy'
output_file = '/content/P450_embeddings_normalized.npy'

# 从.npy文件中读取向量
vectors = np.load(input_file)

# 计算向量集合的L2范数
norms = np.linalg.norm(vectors, axis=1)

# 对向量进行L2归一化
normalized_vectors = vectors / norms[:, np.newaxis]

# 将归一化后的向量保存到.npy文件
np.save(output_file, normalized_vectors)

import numpy as np
from sklearn.decomposition import PCA


protein_embeddings = np.load('/content/drive/MyDrive/P450/P450_embeddings_normalized.npy')

# 创建PCA对象,指定降维后的目标维度为110
pca = PCA(n_components=110)

# 对蛋白质嵌入向量进行PCA降维
low_dim_embeddings = pca.fit_transform(protein_embeddings)

print(f"Shape of the low-dimensional embeddings: {low_dim_embeddings.shape}")
np.save('/content/drive/MyDrive/P450/P450_embeddings_normalized_pca.npy', low_dim_embeddings)

