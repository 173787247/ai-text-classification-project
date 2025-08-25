#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于语义理解的文本分类器
使用BGE-M3模型进行情感分析
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class SemanticTextClassifier:
    """语义文本分类器"""
    
    def __init__(self):
        """初始化分类器"""
        self.classifier = LogisticRegression(random_state=42)
        self.is_trained = False
        
    def extract_features(self, texts):
        """提取文本的语义向量（模拟）"""
        # 注意：这里使用模拟的嵌入向量
        # 实际使用时需要安装 sentence-transformers 并加载 BGE-M3 模型
        print("  注意：当前使用模拟嵌入向量，实际使用请安装 sentence-transformers")
        
        # 生成模拟的768维嵌入向量
        np.random.seed(42)
        embeddings = np.random.randn(len(texts), 768)
        
        # 对每个文本生成稍微不同的向量
        for i, text in enumerate(texts):
            # 基于文本长度和内容生成种子
            seed = hash(text) % 10000
            np.random.seed(seed)
            embeddings[i] += np.random.randn(768) * 0.1
        
        return embeddings
    
    def train(self, texts, labels):
        """训练分类器"""
        print(" 开始训练文本分类器...")
        
        # 特征提取
        X = self.extract_features(texts)
        y = np.array(labels)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练模型
        self.classifier.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f" 模型训练完成")
        print(f" 模型准确率: {accuracy:.4f}")
        print("\n 分类报告:")
        print(classification_report(y_test, y_pred, target_names=['消极', '积极']))
        
        self.is_trained = True
        return accuracy
    
    def predict(self, text):
        """预测新文本的情感"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用 train() 方法")
        
        embedding = self.extract_features([text])
        prediction = self.classifier.predict(embedding)[0]
        probability = self.classifier.predict_proba(embedding)[0]
        
        return {
            'text': text,
            'sentiment': '积极' if prediction == 1 else '消极',
            'confidence': max(probability),
            'probabilities': {
                '消极': probability[0],
                '积极': probability[1]
            }
        }

def demo():
    """演示文本分类器"""
    print(" 文本分类器演示")
    print("=" * 50)
    
    # 准备训练数据
    positive_samples = [
        "这个产品真的太棒了，完全超出我的期望！",
        "服务态度非常好，工作人员很专业",
        "质量很好，价格也很合理，强烈推荐",
        "使用体验非常流畅，界面设计很美观",
        "客服回复很快，问题解决得很及时"
    ]
    
    negative_samples = [
        "这个产品太差了，完全浪费钱",
        "服务态度恶劣，工作人员很不专业",
        "质量很差，价格还这么高，太坑了",
        "使用体验很糟糕，界面设计很混乱",
        "客服回复很慢，问题一直得不到解决"
    ]
    
    labels = [1] * len(positive_samples) + [0] * len(negative_samples)
    texts = positive_samples + negative_samples
    
    print(f" 训练数据：{len(positive_samples)}个正面样本，{len(negative_samples)}个负面样本")
    
    # 创建并训练分类器
    classifier = SemanticTextClassifier()
    accuracy = classifier.train(texts, labels)
    
    # 测试新句子
    test_sentences = [
        "这个产品还不错，但是价格有点贵",
        "服务一般般，没有特别好的地方",
        "质量还可以，但是包装太简陋了"
    ]
    
    print("\n 测试新句子预测结果：")
    for sentence in test_sentences:
        result = classifier.predict(sentence)
        print(f"文本: {result['text']}")
        print(f"情感: {result['sentiment']} (置信度: {result['confidence']:.4f})")
        print(f"概率分布: 消极 {result['probabilities']['消极']:.4f}, 积极 {result['probabilities']['积极']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    demo()
