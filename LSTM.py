import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import re
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题



# ==================== 配置参数 ====================
FILE_PATH = r"D:\课程\大四\大四下\自然语言处理\第二次作业\jyxstxtqj_downcc.com\白马啸西风.txt"
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 64
SEQ_LENGTH = 5  # 序列窗口大小
EPOCHS = 30
MIN_FREQ = 5    # 最低词频

# ==================== 数据预处理 ====================
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.sequences[idx][:-1])
        y = torch.LongTensor([self.sequences[idx][-1]])
        return x, y

def preprocess(text_path):
    # 读取文本并清洗
    with open(text_path, 'r', encoding='gb18030') as f:
        text = f.read()
    text = re.sub(r'\s+', '', text)  # 去除空白字符
    
    # 中文分词
    words = list(jieba.cut(text))
    
    # 构建词汇表
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    vocab = {'<PAD>':0, '<UNK>':1}
    idx = 2
    for word, freq in word_freq.items():
        if freq >= MIN_FREQ:
            vocab[word] = idx
            idx += 1
    
    # 生成训练序列
    sequences = []
    for i in range(len(words)-SEQ_LENGTH):
        seq = [vocab.get(word, 1) for word in words[i:i+SEQ_LENGTH+1]]  # 最后一个是目标词
        sequences.append(seq)
    
    return vocab, sequences

# ==================== 模型定义 ====================
class LSTMLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output

# ==================== 训练过程 ====================
def train_model(vocab, sequences):
    dataset = TextDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMLM(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model

# ==================== 验证方法 ====================
class VectorAnalyzer:
    def __init__(self, model, vocab):
        self.word_vectors = model.embedding.weight.data
        self.idx_to_word = {v:k for k,v in vocab.items()}
        self.word_to_idx = vocab
        
    def find_similar_words(self, word, top_k=5):
        if word not in self.word_to_idx:
            return []
        idx = self.word_to_idx[word]
        vector = self.word_vectors[idx].numpy()
        
        similarities = []
        for i, vec in enumerate(self.word_vectors):
            if i == idx: continue
            sim = cosine_similarity([vector], [vec.numpy()])[0][0]
            similarities.append((self.idx_to_word[i], sim))
        return sorted(similarities, key=lambda x: -x[1])[:top_k]
    
    def visualize_clusters(self, words, n_clusters=2):
        vectors = []
        valid_words = []
        for w in words:
            if w in self.word_to_idx:
                vectors.append(self.word_vectors[self.word_to_idx[w]].numpy())
                valid_words.append(w)
        
        # 降维
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)
        
        # 聚类
        
        os.environ["OMP_NUM_THREADS"] = "1"  # 解决内存泄漏警告

        kmeans = KMeans(n_clusters=2, n_init=10).fit(reduced)  # 显式设置n_init
        
        # 可视化
        plt.figure(figsize=(10,6))
        plt.scatter(reduced[:,0], reduced[:,1], c=kmeans.labels_)
        for i, word in enumerate(valid_words):
            plt.annotate(word, (reduced[i,0], reduced[i,1]))
        plt.title("Word Clusters Visualization")
        plt.show()
    
    def paragraph_similarity(self, para1, para2):
        def get_avg_vector(paragraph):
            words = jieba.lcut(paragraph)
            vectors = []
            for w in words:
                if w in self.word_to_idx:
                    vectors.append(self.word_vectors[self.word_to_idx[w]].numpy())
            return np.mean(vectors, axis=0) if vectors else np.zeros(EMBEDDING_DIM)
        
        vec1 = get_avg_vector(para1)
        vec2 = get_avg_vector(para2)
        return cosine_similarity([vec1], [vec2])[0][0]

# ==================== 执行流程 ====================
if __name__ == "__main__":
    # 数据预处理
    vocab, sequences = preprocess(FILE_PATH)
    print(f"词汇表大小: {len(vocab)}")
    
    # 训练模型
    model = train_model(vocab, sequences)
    
    # 初始化分析器
    analyzer = VectorAnalyzer(model, vocab)
    
    # 示例验证
    print("\n语义相似词示例:")
    print(analyzer.find_similar_words("白马"))
    
    print("\n聚类可视化:")
    analyzer.visualize_clusters(["李文秀", "苏普", "沙漠", "迷宫", "宝藏", "阿曼"])
    
    print("\n段落相似度示例:")
    para1 = "白马带着她一步步回到中原。白马已经老了，只能慢慢的走，但终是能回到中原的。"
    para2 = "沙漠的月夜依然清冷，李文秀望着星空，想起苏普说的那个迷宫传说。"
    print(f"相似度: {analyzer.paragraph_similarity(para1, para2):.4f}")