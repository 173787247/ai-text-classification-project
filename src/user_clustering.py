#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于K-Means的用户分群与画像洞察
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class UserClusteringAnalyzer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_data(self, df):
        """准备聚类数据"""
        numeric_features = ['age', 'income', 'purchase_frequency', 'avg_order_value', 
                          'days_since_last_purchase', 'total_purchases', 
                          'customer_satisfaction', 'website_visits', 'mobile_app_usage']
        
        X = df[numeric_features].copy()
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, numeric_features
    
    def perform_clustering(self, df):
        """执行聚类分析"""
        X_scaled, features = self.prepare_data(df)
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        df['cluster'] = cluster_labels
        
        centroids = self.kmeans.cluster_centers_
        centroids_original = self.scaler.inverse_transform(centroids)
        
        return df, centroids_original, features
    
    def find_representative_users(self, df, centroids_original, features):
        """找到每个群体的代表性用户"""
        representatives = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            cluster_centroid = centroids_original[cluster_id]
            
            distances = []
            for idx, user in cluster_data.iterrows():
                user_features = user[features].values
                distance = np.linalg.norm(user_features - cluster_centroid)
                distances.append((idx, distance))
            
            closest_user_idx = min(distances, key=lambda x: x[1])[0]
            representative_user = df.loc[closest_user_idx]
            
            representatives[cluster_id] = {
                'centroid': dict(zip(features, cluster_centroid)),
                'representative_user': representative_user.to_dict(),
                'cluster_size': len(cluster_data)
            }
        
        return representatives
    
    def analyze_clusters(self, df, representatives):
        """分析聚类结果"""
        print("=== K-Means聚类分析结果 ===\n")
        
        for cluster_id, info in representatives.items():
            print(f"群体 {cluster_id} (共{info['cluster_size']}人):")
            print("群体质心特征:")
            for feature, value in info['centroid'].items():
                print(f"  {feature}: {value:.2f}")
            
            print("\n代表性用户:")
            user = info['representative_user']
            print(f"  用户ID: {user['user_id']}")
            print(f"  年龄: {user['age']:.1f}")
            print(f"  收入: ${user['income']:.0f}")
            print(f"  购买频率: {user['purchase_frequency']:.1f}")
            print(f"  平均订单价值: ${user['avg_order_value']:.0f}")
            print(f"  客户满意度: {user['customer_satisfaction']:.1f}")
            
            self.describe_user_profile(cluster_id, info)
            print("-" * 60)

    def describe_user_profile(self, cluster_id, info):
        """描述用户画像"""
        centroid = info['centroid']
        
        if cluster_id == 0:
            if centroid['income'] > 60000 and centroid['purchase_frequency'] > 6:
                profile = "高收入、高消费的年轻用户群体"
            else:
                profile = "中等收入、稳定消费的用户群体"
        elif cluster_id == 1:
            if centroid['age'] > 40 and centroid['avg_order_value'] > 250:
                profile = "成熟、高价值的中年用户群体"
            else:
                profile = "中等年龄、中等消费的用户群体"
        else:
            if centroid['days_since_last_purchase'] > 60:
                profile = "低活跃度、需要激活的用户群体"
            else:
                profile = "新用户或低价值用户群体"
        
        print(f"\n用户画像: {profile}")

def demo():
    """演示用户聚类分析"""
    print(" 用户聚类分析演示")
    print("=" * 50)
    
    # 生成模拟用户数据
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
    print(f" 生成用户数据：{len(df)}条记录")
    
    # 执行聚类分析
    analyzer = UserClusteringAnalyzer(n_clusters=3)
    df_with_clusters, centroids, features = analyzer.perform_clustering(df)
    representatives = analyzer.find_representative_users(df_with_clusters, centroids, features)
    
    print(" 聚类分析完成，分析结果：")
    analyzer.analyze_clusters(df_with_clusters, representatives)

if __name__ == "__main__":
    demo()
