import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader  # 添加缺失的导入
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
import math
import os

# ==================== 配置参数 ====================
FILE_PATH = r"D:\课程\大四\大四下\自然语言处理\第二次作业\jyxstxtqj_downcc.com\白马啸西风.txt"
EMBEDDING_DIM = 100
WINDOW_SIZE = 10
X_MAX = 100
ALPHA = 0.75
EPOCHS = 50
BATCH_SIZE = 1024
MIN_FREQ = 5
LR = 0.05

# ==================== 数据预处理 ====================
class GloVeDataset(Dataset):  # 继承自Dataset
    def __init__(self, cooccurrences):
        self.data = [(i, j, cnt) for (i, j), cnt in cooccurrences.items()]
        self.weights = [self.weight_fn(cnt) for (i, j, cnt) in self.data]
        
    def weight_fn(self, x):
        return (x / X_MAX)**ALPHA if x < X_MAX else 1.0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        i, j, x_ij = self.data[idx]
        return (
            torch.LongTensor([i]),
            torch.LongTensor([j]),
            torch.FloatTensor([x_ij]),
            torch.FloatTensor([self.weights[idx]])
        )

def build_cooccurrence(text_path):
    with open(text_path, 'r', encoding='gb18030') as f:
        text = re.sub(r'\s+', '', f.read())
    
    words = list(jieba.cut(text))
    
    word_freq = defaultdict(int)
    for w in words:
        word_freq[w] += 1
    vocab = {'<PAD>':0, '<UNK>':1}
    idx = 2
    for w, cnt in word_freq.items():
        if cnt >= MIN_FREQ:
            vocab[w] = idx
            idx += 1
    id2word = {v:k for k,v in vocab.items()}
    
    cooccurrence = defaultdict(float)
    for center in range(len(words)):
        center_word = words[center]
        if center_word not in vocab:
            continue
        center_idx = vocab[center_word]
        
        start = max(0, center - WINDOW_SIZE)
        end = min(len(words), center + WINDOW_SIZE + 1)
        
        for context in range(start, end):
            if context == center:
                continue
            context_word = words[context]
            if context_word not in vocab:
                continue
            context_idx = vocab[context_word]
            
            distance = abs(context - center)
            cooccurrence[(center_idx, context_idx)] += 1.0 / distance
    
    return vocab, id2word, cooccurrence

# ==================== GloVe模型 ====================
class GloVe(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.wi = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.wj = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.bi = nn.Embedding(vocab_size, 1)
        self.bj = nn.Embedding(vocab_size, 1)
        
        nn.init.uniform_(self.wi.weight, -0.5/EMBEDDING_DIM, 0.5/EMBEDDING_DIM)
        nn.init.uniform_(self.wj.weight, -0.5/EMBEDDING_DIM, 0.5/EMBEDDING_DIM)
        nn.init.constant_(self.bi.weight, 0.0)
        nn.init.constant_(self.bj.weight, 0.0)
    
    def forward(self, i, j, x_ij, weights):
        wi = self.wi(i)
        wj = self.wj(j)
        bi = self.bi(i).squeeze()
        bj = self.bj(j).squeeze()
        
        similarity = torch.sum(wi * wj, dim=1) + bi + bj
        loss = weights * torch.pow(similarity - torch.log(x_ij), 2)
        return torch.mean(loss)

# ==================== 训练过程 ====================
def train_glove(vocab, cooccurrence):
    dataset = GloVeDataset(cooccurrence)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)  # 现在已定义
    
    model = GloVe(len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for i, j, x_ij, weights in dataloader:
            optimizer.zero_grad()
            loss = model(i.squeeze(), j.squeeze(), x_ij.squeeze(), weights.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    
    embeddings = model.wi.weight.data + model.wj.weight.data
    return embeddings.numpy()

# ==================== 验证工具 ====================
class VectorValidator:
    def __init__(self, embeddings, vocab, id2word):
        self.embeddings = embeddings
        self.vocab = vocab
        self.id2word = id2word
        
    def get_vector(self, word):
        if word not in self.vocab:
            return None
        return self.embeddings[self.vocab[word]]
    
    def most_similar(self, word, topn=5):
        vec = self.get_vector(word)
        if vec is None:
            return []
        
        similarities = []
        for idx in range(len(self.embeddings)):
            if idx == self.vocab[word]:
                continue
            sim = cosine_similarity([vec], [self.embeddings[idx]])[0][0]
            similarities.append((self.id2word[idx], sim))
        return sorted(similarities, key=lambda x: -x[1])[:topn]
    
    def visualize_clusters(self, words):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        vectors = []
        labels = []
        for w in words:
            if w in self.vocab:
                vectors.append(self.embeddings[self.vocab[w]])
                labels.append(w)
        
        pca = PCA(n_components=2)
        points = pca.fit_transform(vectors)
        
        plt.figure(figsize=(12,8))
        plt.scatter(points[:,0], points[:,1], c='blue')
        for i, label in enumerate(labels):
            plt.annotate(label, (points[i,0], points[i,1]))
        plt.title("Word Embedding Visualization")
        plt.show()
    
    def paragraph_similarity(self, text1, text2):
        def text2vec(text):
            words = jieba.cut(text)
            vecs = [self.get_vector(w) for w in words if w in self.vocab]
            return np.mean(vecs, axis=0) if vecs else None
        
        vec1 = text2vec(text1)
        vec2 = text2vec(text2)
        if vec1 is None or vec2 is None:
            return 0.0
        return cosine_similarity([vec1], [vec2])[0][0]

# ==================== 主流程 ====================
if __name__ == "__main__":
    # 解决KMeans内存泄漏警告
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # 数据预处理
    vocab, id2word, cooccurrence = build_cooccurrence(FILE_PATH)
    print(f"词汇表大小: {len(vocab)}")
    
    # 训练模型
    embeddings = train_glove(vocab, cooccurrence)
    
    # 初始化验证器
    validator = VectorValidator(embeddings, vocab, id2word)
    
    # 示例验证
    print("\n『白马』相似词:")
    print(validator.most_similar('白马'))
    
    print("\n聚类可视化:")
    validator.visualize_clusters(['李文秀', '苏普', '阿曼', '沙漠', '迷宫', '宝藏'])
    
    para1 = "白马带着她一步步回到中原。白马已经老了，只能慢慢的走，但终是能回到中原的。"
    para2 = "沙漠的月夜依然清冷，李文秀望着星空，想起苏普说的那个迷宫传说。"
    print(f"\n段落相似度: {validator.paragraph_similarity(para1, para2):.4f}")