from math import sqrt
import yaml
import os
import line_profiler
import numpy as np
from numba import jit
from numba.pycc import CC
from numba import prange

LINE_PROFILE = 1

@line_profiler.profile
def distance(point1, point2):
	n = max(len(point1), len(point2))
	d = 0
	n = len(point1)
	for i in range(n):
		d += (point1[i] - point2[i])**2
	return sqrt(d)

@line_profiler.profile
def distance_numpy(point1, point2):
        tmp_d = point1 - point2
        tmp_d = np.square(tmp_d)
        tmp_d = np.sum(tmp_d)
        return np.sqrt(tmp_d)


# AOT compilation.
cc = CC('module')
@cc.export('distance_numba', 'f8(f8[:], f8[:])')
@jit(parallel=True)
def distance_numba(point1, point2):
	n = max(len(point1), len(point2))
	d = 0
	n = len(point1)
	for i in prange(n):
		d += (point1[i] - point2[i])**2
	return sqrt(d)

@line_profiler.profile
def majority_vote(neighbors):
	unique_labels = set(neighbors)
	max_label = None
	max_count = 0
	for label in unique_labels:
		tmp = 0
		for element in neighbors:
			if element == label:
				tmp += 1
		if tmp > max_count:
			max_count = tmp
			max_label = label
	return max_label


def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs


def read_file(file_dir):
	features = []
	labels = []
	with open(file_dir, 'r') as file:
        	lines = file.readlines()
	for line in lines:
		feature = line.strip().split(",")[:-1]
		feature = [float(f) for f in feature]
		label = line.strip().split(",")[-1]
		if label in ["g", "0"]:
			labels.append(0)
		else:
			labels.append(1)
		features.append(feature)
	return features, labels


cc.compile()
