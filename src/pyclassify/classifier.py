from pyclassify import utils
import line_profiler
import numpy as np
from .module import distance_numba

LINE_PROFILE = 1

class kNN():

	def __init__(self, k, backhand="plain"):
		if not isinstance(k, int):
			raise TypeError("k must be an int")
		if backhand not in ["plain", "numpy", "numba"]:
			raise ValueError("backhand must either be numpy or plain")
		self.k = k
		self.backhand = backhand
		if self.backhand == "plain":
			self.distance = utils.distance
		elif self.backhand == "numpy":
			self.distance = utils.distance_numpy
		elif self.backhand == "numba":
			self.distance = utils.distance_numba

	@line_profiler.profile
	def _get_k_nearest_neighbors(self, X, y, x):
		dist_list = [(idx, self.distance(p,x)) for idx, p in enumerate(X)]    #utils.distance
		dist_list.sort(key=lambda d: d[1])
		idx_list = [t[0] for t in dist_list[:self.k]]
		return [y[i] for i in idx_list]

	@line_profiler.profile
	def __call__(self, data, new_points):
		X = data[0]
		y = data[1]
		y_pred = []
		if self.backhand == "numpy":
			X = np.array(X)
			new_points = np.array(new_points)
		for x in new_points:
			y_k_ngh = self._get_k_nearest_neighbors(X, y, x)
			y_label = utils.majority_vote(y_k_ngh)
			y_pred.append(y_label)
		return y_pred
