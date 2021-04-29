import numpy as np

from .tree import Tree

class IsolationForest(object):
	def __init__(self, n_estimators=100, max_depth=5, max_iter=50, random_state=None):
		self.estimators_ = [Tree(max_depth, max_iter, random_state) for _ in range(n_estimators)]

	def fit(self, X):
		for e in self.estimators_:
			e.fit(X)

	def predict(self, X):
		acc = np.empty(shape=(len(self.estimators_), X.shape[0]), dtype=np.float64)

		for i, e in enumerate(self.estimators_):
			acc[i, :] = e.predict(X).reshape(-1)

		return np.mean(acc, axis=0)
