import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array([[0.2, -0.3], [-1.1, 2], [1, -2.2], [0.5, -1], [-0.6, 1]])
num_components = 1
print("data:\n", data)

scaler = StandardScaler()
scaler.fit(data)
scaled_data = scaler.transform(data)
print("\nscaled:\n", scaled_data)

covariance_matrix = np.cov(scaled_data.T)
print("\ncovariance matrix:\n", covariance_matrix)

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print("\neigenvalues:\n", eigenvalues)
print("\neigenvectors:\n", eigenvectors)

print(
    "\nProportion of variance:\n", sum(eigenvalues[:num_components]) / sum(eigenvalues)
)

projection_matrix = (eigenvectors.T[:][:num_components]).T
print("\nprojection matrix:\n", projection_matrix)

data_pca = scaled_data.dot(projection_matrix)
print("\ndata_pca:\n", data_pca)

# compare to scikit-learn version
from sklearn.decomposition import PCA

pca = PCA(n_components=num_components)

pca.fit(scaled_data)
print("\nscikit-learn version:\n", pca.transform(scaled_data))
