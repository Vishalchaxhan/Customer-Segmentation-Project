import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Streamlit Title
st.title("Customer Segmentation using K-Means Clustering")

# Step 1: Load Dataset
df = pd.read_csv("Mall_Customers.csv")  # Direct file read
st.write("### 📌 Dataset Preview")
st.write(df.head())

# Step 2: Select Features (Assuming 3rd & 4th columns are relevant)
X = df.iloc[:, [3, 4]].values  

# Step 3: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Finding Optimal Clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
st.write("### 📊 Elbow Method for Optimal K")
fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS")
ax.set_title("Elbow Method for Optimal K")
st.pyplot(fig)

# Step 5: Apply K-Means
optimal_k = 5  # Choose based on elbow graph
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 6: Visualization
st.write("### 🎨 Clustering Visualization")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=y_kmeans, palette="viridis", s=100, ax=ax)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
ax.set_xlabel("Annual Income (scaled)")
ax.set_ylabel("Spending Score (scaled)")
ax.set_title("Customer Segmentation using K-Means")
ax.legend()
st.pyplot(fig)
