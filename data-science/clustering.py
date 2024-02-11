# Import modules and packages
import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import KMeans

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./dataset/nba_players_shooting.csv")
X = data["X"]
Y = data["Y"]
string_to_binary = {'MADE': 1, 'MISSED': 0}
#string_to_binary = {'TRUE': 1, 'FALSE': 0}
# Apply the mapping to create a new binary column
data['SCORE'] = data['SCORE'].map(string_to_binary)
categorical = {'Seth Curry': 3, 'Chris Paul': 2, 'Russell Westbrook': 1, 'Trae Young': 0}
data['SHOOTER'] = data['SHOOTER'].map(categorical)
data[['first_part', 'second_part']] = data['RANGE'].str.split(',', expand=True)
data = data.drop(columns="RANGE")
data = data.drop(columns="first_part")
data['RANGE'] = data['second_part'].str[:-1]
data = data.drop(columns="second_part")
data = data.drop(columns="DEFENDER")
data['RANGE'] = pd.to_numeric(data['RANGE']).astype('Int64')
condition1 = (data['SHOOTER'] == 3)
#condition2 = (data['SCORE'] == 1)
condition3 = (data['RANGE'] >= 24)

# Combine conditions using logical AND (&)
combined_condition = condition1 & condition3 #& condition2
filtered_df = data[combined_condition]
X = filtered_df["X"]
Y = filtered_df["Y"]

# Apply DBSCAN
dbscan = DBSCAN(eps=10, min_samples=5)
labelss = dbscan.fit_predict(filtered_df)



# # Read data
# data = pd.read_csv('https://raw.githubusercontent.com/sowmyacr/kmeans_cluster/master/CLV.csv')
# X = np.array(data)

# # System constants
# number_of_clusters = 4

# # Apply KMEans to the Data
# kmeans = KMeans(
# 	n_clusters=4,
# 	init='k-means++',
# 	max_iter=300,
# 	n_init=10,
# 	random_state=19890528,
# 	precompute_distances=True)

# kfit = kmeans.fit(X)

# freeze_centroids = kmeans.cluster_centers_
# print(freeze_centroids)
# print(freeze_centroids.shape)

# # Save centroids as pickle file locally
# with open('freezed_centroids.pkl','wb') as f:
#     pickle.dump(freeze_centroids, f)
#     print('Centroids are being saved to pickle file.')