"""
The goal of this script is to use a toy example for clustering analysis.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# set the number of clusters to be studied
NB_CLUSTER = 3
# set the number of principal components to lower dimensional space
NB_COMPONENT = 10


# load data
digits = load_digits(n_class=NB_CLUSTER)
X_digits, y_digits = digits.data, digits.target
# X_digits.shape: (~n_class*180, 64) since ~n_class*180 images of 8x8=64 pixels/features
# y_digits.shape: (~n_class*180,)
print(X_digits.shape)


# lower dimensional space 
estimator = PCA(n_components=NB_COMPONENT)
X_pca = estimator.fit_transform(X_digits)
# X_pca.shape: (~n_class*180, n_components)


# plot clusters (10 colors at most depending on NB_CLUSTER)
colors = ["black", "blue", "purple", "yellow", "pink", "red", "lime", "cyan", "orange", "gray"]

for i in range(len(colors)):
    px = X_pca[:, 0][y_digits == i] # first component
    py = X_pca[:, 1][y_digits == i] # second component
    plt.scatter(px, py, c=colors[i])

plt.legend(digits.target_names)
plt.xlabel("Première composante principale")
plt.ylabel("Deuxième composante principale")

plt.show()


# now we want to see the clusters found by k-mean algorithm
kmeans = KMeans(n_clusters=NB_CLUSTER).fit(X_digits)
kmeans_pred = kmeans.predict(X_digits)
# kmeans_pred.shape == y_digits.shape


# therefore we plot the 2 first components like before but with the clusters learned by k-mean
for i in range(len(colors)):
    px = X_pca[:, 0][kmeans_pred == i]
    py = X_pca[:, 1][kmeans_pred == i]
    plt.scatter(px, py, c=colors[i])

plt.legend(digits.target_names)
plt.xlabel("Première composante principale")
plt.ylabel("Deuxième composante principale")

plt.show()
