from pyclassify import utils

class kNN():
	def __init__(self, k):
		if not isinstance(k, int):
			raise TypeError("k must be an int")
		self.k = k
	
	def _get_k_nearest_neighbors(self, X, y, x):
		dist_list = [(idx, utils.distance(p,x)) for idx, p in enumerate(X)]
		dist_list.sort(key=lambda d: d[1])
		idx_list = [t[0] for t in dist_list[:self.k]]
		return [y[i] for i in idx_list]

	def __call__(self, data, new_points):
		X = data[0]
		y = data[1]
		y_pred = []
		for x in new_points:
			y_k_ngh = self._get_k_nearest_neighbors(X, y, x)
			y_label = utils.majority_vote(y_k_ngh)
			y_pred.append(y_label)
		return y_pred
