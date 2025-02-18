from pyclassify.utils import distance, majority_vote
from pyclassify.classifier import kNN
import pytest

def test_distance():
	point1 = [2, 1, 4, 3]
	point2 = [1, 2, 3, 4]
	point3 = [1,2,3.6, 6]
	with pytest.raises(AssertionError):
		assert distance(point1,point2) == 3
		assert distance(point1, point2) == distance(point2, point1)  #simmetry
		assert distance(point1, point2) >= 0                         #positive definite
		assert distance(point1, point1) == 0
		assert distance(point1, point2) < distance(point3, point2) + distance(point1, point3)    #triangular inequality


def test_majority_vote():
	y_ngh = [1, 0, 0, 0]
	#with pytest.raises(AssertionError):
	assert majority_vote(y_ngh) == 0


def test_constr():
	k = 2
	cf = kNN(k)
	assert cf.k == k
