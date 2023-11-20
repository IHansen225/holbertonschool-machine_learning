#!/usr/bin/env python3
"""
    K-means module
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
        Performs K-means on a dataset
    """
    try:
        centroids = initialize(X, k)
        for _ in range(iterations):
            c = centroids.copy()
            distance = np.sqrt(np.sum(
                (X - centroids[:, np.newaxis]) ** 2, axis=2))
            clss = np.argmin(distance, axis=0)

            for j in range(k):
                if len(X[clss == j]) == 0:
                    centroids[j] = initialize(X, 1)
                else:
                    centroids[j] = np.mean(X[clss == j], axis=0)

            distance = np.sqrt(np.sum(
                (X - centroids[:, np.newaxis]) ** 2, axis=2))
            clss = np.argmin(distance, axis=0)
            if np.all(c == centroids):
                break

        return centroids, clss
    except Exception:
        return None, None


def initialize(X, k):
    """
        Initializes cluster centroids for K-means
    """
    if not isinstance(k, int) or k <= 0:
        return
    try:
        min = np.min(X, axis=0)
        max = np.max(X, axis=0)

        # Obtain the centroids
        return np.random.uniform(min, max, size=(k, X.shape[1]))
    except Exception:
        pass
