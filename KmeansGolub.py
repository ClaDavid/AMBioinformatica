# coding=utf-8
#IMPORTATE: a linha 1 é extremamente necessaria para a realizacao da leitura do arquivo
# Adaptado de http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html


#imports
from __future__ import print_function

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)

# Bases
ds = np.loadtxt('golub.csv', delimiter = ",")

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Plotar grafico
    fig, (ax1) = plt.subplots()
    fig.set_size_inches(5, 5)

    
    # Setar eixos do grafico
    ax1.set_xlim([-0.1, 1])
    
    # Colocar espaço em branco entre os clustes
    ax1.set_ylim([0, len(ds) + (n_clusters + 1) * 10])

    #Roda o k means
    clusterer = KMeans(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(ds)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(ds, cluster_labels)
    print("Para n_clusters =", n_clusters,
          "Silhouette_score is :", silhouette_avg)

    # Guarda o score de cada silhueta
    sample_silhouette_values = silhouette_samples(ds, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Plot da silhueta dos clusters.")
    ax1.set_xlabel("Coeficiente de silhueta")
    ax1.set_ylabel("Cluster Numero")

    # The vertical line for average silhoutte score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.annotate('Coeficiente Medio da Silhueta', xy=(silhouette_avg, 1.5), xytext=(silhouette_avg+0.1, 1.5),
            arrowprops=dict(facecolor='blue', shrink=0.05),
            )

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

 
    plt.suptitle(("Analise de silhueta para KMeans"
                  "k = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    
    plt.show()

