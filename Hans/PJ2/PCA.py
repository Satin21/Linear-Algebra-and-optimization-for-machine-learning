import numpy as np
import matplotlib.pyplot as plt

from main import read_csv, mean_centered, split_data, PCA, pdf2cdf
from sklearn.preprocessing import StandardScaler


def PCA_own(data, dim=2):

    features = np.array(data)[:, :-1]
    label = np.array(data)[:, -1]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    scaled_features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    covariance = (1 / features.shape[0]) * (scaled_features.T @ scaled_features)

    eigval, eigvec = np.linalg.eigh(covariance)
    eigval = eigval[::-1]
    eigvec = eigvec[:, ::-1]

    reduced_data = scaled_features @ eigvec[:, :dim]

    variances = eigval / sum(eigval)

    total_variances = list(sum(variances[:i]) for i in range(1, len(variances) + 1))

    ax.plot(range(1, len(variances) + 1), total_variances, "*", c="r", label="own PCA")

    # do PCA using sklearn
    X = mean_centered(np.array(data)[:, :13])

    # Scale the input data/features
    sc = StandardScaler()
    X = sc.fit_transform(X)

    pca = PCA(n_components=2)
    pc = pca.fit_transform(X)  # Compute the principal components & transform the data

    n_features = len(X[0])
    pca = PCA(n_components=n_features)
    pca.fit_transform(X)  # Compute the principal components & transform the data
    cdf = pdf2cdf(pca.explained_variance_ratio_)

    ax.plot(cdf, "-", c="b", label="sklearn-PCA")

    ax.set_ylabel("Variance (%)")
    ax.set_xlabel("Number of principal components")

    ax.set_title("Explained variance CDF")

    ax.legend()

    # ax[1].scatter(reduced_data[:, 0], reduced_data[:, 1], c = label);
    # ax[1].set_xlabel('Principal component 1');
    # ax[1].set_ylabel('Principal component 2');

    # ax[1].set_title('2D feature space')

    fig.savefig("PCA comparison.pdf", format="pdf", bbox_inches="tight")
