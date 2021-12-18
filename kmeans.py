from sklearn import cluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_KMeans(max_k, data):
    max_k += 1
    kmeans_results = dict()
    for k in range(2, max_k):
        kmeans = cluster.KMeans(n_clusters=k
                                , init='k-means++'
                                , n_init=10
                                , tol=0.0001
                                , random_state=1
                                , algorithm='full')
        #fit_predict when we want silhouettte score
        #kmeans_results.update({k: kmeans.fit(data)})
        kmeans_results.update({k: kmeans.fit_predict(data)})

    return kmeans_results


def run_DBSan(data):
    return cluster.DBSCAN(eps=1.2, min_samples=1).fit_predict(data)
    # for printing clusters
    # return cluster.DBSCAN(eps=1.2, min_samples=1).fit(data)


def run_Birch(data):
    return cluster.Birch(threshold=0.01, n_clusters=2).fit(data)


def get_top_features_cluster(tf_idf_array, prediction, n_feats, vectorizer):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction == label)  # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis=0)  # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats]  # indices with top 20 scores
        features = vectorizer.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns=['features', 'score'])
        dfs.append(df)
    return dfs


def plotWords(dfs, n_feats):
    plt.figure(figsize=(8, 4))
    for i in range(0, len(dfs)):
        plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
        sns.barplot(x='score', y='features', orient='h', data=dfs[i][:n_feats])
        plt.show()
