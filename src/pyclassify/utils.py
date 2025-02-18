from math import sqrt
import yaml
import os

def distance(point1, point2):
	n = max(len(point1), len(point2))
	d = 0
	n = len(point1)
	for i in range(n):
		d += (point1[i] - point2[i])**2
	return sqrt(d)

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
		if label == "g":
			labels.append(0)
		else:
			labels.append(1)
		features.append(feature)
	return features, labels
