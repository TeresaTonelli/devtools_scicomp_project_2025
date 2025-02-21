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

@pytest.mark.parametrize(
	"k, backhand, expected_k, expected_backhand",
	[
		(5, "plain", 5, "plain"),
		(5, "numpy", 5, "numpy"),
	]
)
def test_kNN_constructor(k, backhand, expected_k, expected_backhand):
	cf = kNN(k, backhand)
	assert cf.k == expected_k
	assert cf.backhand == expected_backhand


@pytest.mark.parametrize(
        "k, backhand",
        [
                ("5", "plain"),
                (5.25, "numpy"),
        ]
)
def test_kNN_TypeError(k, backhand):
	with pytest.raises(TypeError):
		cf = kNN(k, backhand)


@pytest.mark.parametrize(
        "k, backhand",
        [
                (5, "p"),
        ]
)
def test_kNN_ValueError(k, backhand):
        with pytest.raises(ValueError):
                cf = kNN(k, backhand)
