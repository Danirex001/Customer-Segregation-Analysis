from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Select relevant features
filtered_df = df[['Value', 'Quantity', 'Payment_Terms']]

# Encode Payment_Terms
le = LabelEncoder()
filtered_df['Payment_Terms'] = le.fit_transform(filtered_df['Payment_Terms'])

# Scale numerical features
scaler = StandardScaler()
dfscaled = scaler.fit_transform(filtered_df.drop('Payment_Terms', axis=1))

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(dfscaled)

# Plot the clusters
plt.scatter(dfscaled[:, 0], dfscaled[:, 1], c=kmeans.labels_)
plt.xlabel('Scaled Value')
plt.ylabel('Scaled Quantity')
plt.title('KMeans Clustering')
plt.show()
