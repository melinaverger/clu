"""
The goal of this script is to use a toy example for clustering analysis.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


# set the number of clusters to be studied
NB_CLUSTER = 10
# set the number of principal components to lower dimensional space
# max. 64
NB_COMPONENT = 10


# load data
digits = load_digits(n_class=NB_CLUSTER)
X_digits, y_digits = digits.data, digits.target
# X_digits.shape: (~n_class*180, 64) since ~n_class*180 images of 8x8=64 pixels/features
# y_digits.shape: (~n_class*180,)
print("X:\t", X_digits.shape)


# split data
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, train_size=0.8, random_state=0, shuffle=True)
print("X_train:", X_train.shape)
print("X_test:\t", X_test.shape)
print("\n")


# lower dimensional space of X_train
estimator = PCA(n_components=NB_COMPONENT)
X_pca = estimator.fit_transform(X_train)
# X_pca.shape: (~n_class*180, n_components)

# when we set n_components=0.95 to have a ratio of variance of preserve at 0.95,
# we need 28 PC to reach this 95%
print(estimator.explained_variance_ratio_)
print(len(estimator.explained_variance_ratio_))
print("\n")



# plot clusters (10 colors at most depending on NB_CLUSTER)
colors = ["black", "blue", "purple", "yellow", "pink", "red", "lime", "cyan", "orange", "gray"]

for i in range(NB_CLUSTER):
    px = X_pca[:, 0][y_train == i] # first component
    py = X_pca[:, 1][y_train == i] # second component
    plt.scatter(px, py, c=colors[i])

plt.legend(digits.target_names[:NB_CLUSTER])
plt.xlabel("Première composante principale")
plt.ylabel("Deuxième composante principale")
#plt.title("Clustering avec X_train réduit en fonction de y_train (ground truth)")

plt.show()


""" # now we want to see the clusters found by k-mean algorithm
kmeans = KMeans(n_clusters=NB_CLUSTER).fit(X_train)
kmeans_pred = kmeans.predict(X_train)
# kmeans_pred.shape == y_digits.shape


# therefore we plot the 2 first components like before but with the clusters learned by k-mean
for i in range(NB_CLUSTER):
    px = X_pca[:, 0][kmeans_pred == i]
    py = X_pca[:, 1][kmeans_pred == i]
    plt.scatter(px, py, c=colors[i])

plt.legend(digits.target_names[:NB_CLUSTER])
plt.xlabel("Première composante principale")
plt.ylabel("Deuxième composante principale")
plt.title("Clustering avec X_train réduit en fonction de kmeans_pred (clustering initial)")

plt.show()


print("Rand index (ground truth vs. kmeans predictions): ", metrics.rand_score(y_train, kmeans_pred))
print("\n") """

# ===========================
# now we want a predictive model to be trained on X_digits and kmeans_pred as labeled data
# DT or LR


lr = LogisticRegression(penalty="none", random_state=0)
lr.fit(X_train, y_train)
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)


print("LR accuracy - train", metrics.accuracy_score(y_train, y_pred_train))
print("LR accuracy - test", metrics.accuracy_score(y_test, y_pred_test))


for i in range(NB_CLUSTER):
    px = X_pca[:, 0][y_pred_train == i] # first component
    py = X_pca[:, 1][y_pred_train == i] # second component
    plt.scatter(px, py, c=colors[i])

plt.legend(digits.target_names[:NB_CLUSTER])
plt.xlabel("Première composante principale")
plt.ylabel("Deuxième composante principale")
#plt.title("[LR] Clustering avec X_train réduit en fonction de y_pred_train (clustering avec prédictions d'entraînement)")

plt.show()


print("Rand index (ground trut vs. LR predictions) - train: ", metrics.rand_score(y_train, y_pred_train))
print("Rand index (ground trut vs. LR predictions) - test: ", metrics.rand_score(y_test, y_pred_test))

print("\n")


""" dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, kmeans_pred)
y_pred2_train = dt.predict(X_train)
y_pred2_test = dt.predict(X_test)

print("DT accuracy - train", metrics.accuracy_score(kmeans_pred, y_pred2_train))
print("DT accuracy - test", metrics.accuracy_score(kmeans_pred, y_pred2_test))

for i in range(NB_CLUSTER):
    px = X_pca[:, 0][y_pred2_train == i] # first component
    py = X_pca[:, 1][y_pred2_train == i] # second component
    plt.scatter(px, py, c=colors[i])

plt.legend(digits.target_names[:NB_CLUSTER])
plt.xlabel("Première composante principale")
plt.ylabel("Deuxième composante principale")
plt.title("[DT] Clustering avec X_train réduit en fonction de y_pred2_train (clustering avec prédictions d'entraînement)")

plt.show()


print("Rand index (kmeans predictions vs. DT predictions): ", metrics.rand_score(kmeans_pred, y_pred2_train))
print("Rand index (ground truth vs. LR predictions): ", metrics.rand_score(y_train, y_pred2_train))
print("\n") """
