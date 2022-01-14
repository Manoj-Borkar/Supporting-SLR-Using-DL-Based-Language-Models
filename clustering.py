from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

kmeans_kwargs = {
    "init": "k-means++",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}

def kneelocator(clustering_embeddings):
    sse = []
    
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(clustering_embeddings)
        sse.append(kmeans.inertia_)
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    if not kl.elbow:
        return 2
    else:
        return kl.elbow

def clustering_func(data, doc_embeddings):
    
    no_clusters = kneelocator(doc_embeddings)
    kmeans = KMeans(n_clusters=no_clusters, random_state=0).fit(doc_embeddings)
    true_labels = np.array(data['label'])
    alllabels = [true_labels[ind] for ind in range(len(kmeans.labels_))]
    allpreds = [kmeans.labels_[ind] for ind in range(len(kmeans.labels_))]
    data["Preds"]= allpreds
    return data
    
