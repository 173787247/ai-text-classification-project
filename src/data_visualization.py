#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于PCA的用户数据可视化探索
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class UserDataVisualizer:
    def __init__(self):
        self.pca_2d = PCA(n_components=2)
        self.pca_3d = PCA(n_components=3)
        
    def prepare_visualization_data(self, df):
        """准备可视化数据"""
        numeric_features = ['age', 'income', 'purchase_frequency', 'avg_order_value', 
                          'days_since_last_purchase', 'total_purchases', 
                          'customer_satisfaction', 'website_visits', 'mobile_app_usage']
        
        X = df[numeric_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, numeric_features
    
    def perform_pca_and_visualize(self, df):
        """执行PCA降维并可视化"""
        X_scaled, features = self.prepare_visualization_data(df)
        
        # 2D PCA
        X_pca_2d = self.pca_2d.fit_transform(X_scaled)
        
        # 3D PCA
        X_pca_3d = self.pca_3d.fit_transform(X_scaled)
        
        # 创建可视化
        self.create_2d_visualization(X_pca_2d, df)
        self.create_3d_visualization(X_pca_3d, df)
        
        # 打印解释方差比
        self.print_explained_variance()
        
        return X_pca_2d, X_pca_3d
    
    def create_2d_visualization(self, X_pca_2d, df):
        """创建2D散点图"""
        plt.figure(figsize=(12, 8))
        
        colors = ['red', 'blue', 'green']
        cluster_names = ['群体 0', '群体 1', '群体 2']
        
        for cluster_id in range(3):
            mask = df['cluster'] == cluster_id
            plt.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], 
                       c=colors[cluster_id], label=cluster_names[cluster_id], 
                       alpha=0.7, s=50)
        
        plt.xlabel(f'第一主成分 ({self.pca_2d.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'第二主成分 ({self.pca_2d.explained_variance_ratio_[1]:.3f})')
        plt.title('用户数据PCA降维可视化 (2D)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/pca_2d_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_3d_visualization(self, X_pca_3d, df):
        """创建3D散点图"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['red', 'blue', 'green']
        cluster_names = ['群体 0', '群体 1', '群体 2']
        
        for cluster_id in range(3):
            mask = df['cluster'] == cluster_id
            ax.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
                      c=colors[cluster_id], label=cluster_names[cluster_id], 
                      alpha=0.7, s=50)
        
        ax.set_xlabel(f'第一主成分 ({self.pca_3d.explained_variance_ratio_[0]:.3f})')
        ax.set_ylabel(f'第二主成分 ({self.pca_3d.explained_variance_ratio_[1]:.3f})')
        ax.set_zlabel(f'第三主成分 ({self.pca_3d.explained_variance_ratio_[2]:.3f})')
        ax.set_title('用户数据PCA降维可视化 (3D)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/pca_3d_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_explained_variance(self):
        """打印解释方差比"""
        print("=== PCA解释方差比 ===")
        print("2D PCA:")
        for i, ratio in enumerate(self.pca_2d.explained_variance_ratio_):
            print(f"  主成分 {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
        
        print("\n3D PCA:")
        for i, ratio in enumerate(self.pca_3d.explained_variance_ratio_):
            print(f"  主成分 {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
        
        print(f"\n2D PCA累计解释方差: {sum(self.pca_2d.explained_variance_ratio_):.4f} ({sum(self.pca_2d.explained_variance_ratio_)*100:.2f}%)")
        print(f"3D PCA累计解释方差: {sum(self.pca_3d.explained_variance_ratio_):.4f} ({sum(self.pca_3d.explained_variance_ratio_)*100:.2f}%)")

def demo():
    """演示数据可视化"""
    print(" 数据可视化演示")
    print("=" * 50)
    
    # 生成模拟数据
    np.random.seed(42)
    n_users = 100
    
    user_data = {
        'user_id': range(1, n_users + 1),
        'age': np.random.normal(35, 10, n_users).clip(18, 65),
        'income': np.random.normal(50000, 20000, n_users).clip(20000, 150000),
        'purchase_frequency': np.random.poisson(5, n_users),
        'avg_order_value': np.random.normal(200, 80, n_users).clip(50, 500),
        'days_since_last_purchase': np.random.exponential(30, n_users).clip(1, 365),
        'total_purchases': np.random.poisson(20, n_users),
        'customer_satisfaction': np.random.normal(4.2, 0.8, n_users).clip(1, 5),
        'website_visits': np.random.poisson(15, n_users),
        'mobile_app_usage': np.random.normal(0.7, 0.3, n_users).clip(0, 1)
    }
    
    df = pd.DataFrame(user_data)
    
    # 添加聚类标签（模拟）
    df['cluster'] = np.random.randint(0, 3, n_users)
    
    print(f" 生成用户数据：{len(df)}条记录")
    print(" 开始PCA降维和可视化...")
    
    # 执行可视化
    visualizer = UserDataVisualizer()
    X_pca_2d, X_pca_3d = visualizer.perform_pca_and_visualize(df)
    
    print(" 数据可视化完成！")

if __name__ == "__main__":
    demo()
