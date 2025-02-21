from pyclassify.classifier import kNN
from pyclassify.utils import read_config, read_file
import argparse
import numpy as np

parser = argparse.ArgumentParser(description = "Practical session 2")
parser.add_argument("--config", type=str, default=r"experiments/config")
args = parser.parse_args()

config_dir = args.config
#config_dir = fr"/root/devtools_scicomp_teresa/devtools_scicomp_project_2025/experiments/config"
parameters = read_config(config_dir)
#print("parameters", parameters)

k = parameters["k"]
pth = parameters["dataset"]
try:
	backhand = parameters["backhand"]
except:
	backhand = "plain"

#total_features, total_labels = read_file(fr"/root/devtools_scicomp_teresa/devtools_scicomp_project_2025/data/ionosphere.data")
total_features, total_labels = read_file(pth)

#shuffle dataset with indexes
indices = np.arange(len(total_features))
permuted_indices = np.random.permutation(indices)

#split 80% and 20 % train and test
idx_20 = int(0.2*len(total_features))
train_features = [total_features[i] for i in permuted_indices[:idx_20]]
train_labels = [total_labels[i] for i in permuted_indices[:idx_20]]
test_features = [total_features[i] for i in permuted_indices[idx_20:]]
test_labels = [total_labels[i] for i in permuted_indices[idx_20:]]

k = parameters["k"]
my_kNN = kNN(k, backhand)
test_pred = my_kNN.__call__((train_features, train_labels), test_features)
#print("test pred", test_pred)

#accuracy
accuracy_counter = 0
for i in range(len(test_labels)):
	if test_labels[i] == test_pred[i]:
		accuracy_counter += 1
print("accuracy", accuracy_counter / len(test_labels))
