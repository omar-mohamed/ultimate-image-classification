import pandas as pd
import os
import numpy as np
from configs import argHandler  # Import the default arguments

dataset_df = pd.read_csv('./data/all_data.csv')

test_set_fraction=0.2

shuffle=True

if shuffle:
    dataset_df = dataset_df.sample(frac=1., random_state=np.random.randint(1,100))

train_dict = {"path":[],"class":[]}
test_dict = {"path":[],"class":[]}

class_size = {}
test_sizes = {}
for index, row in dataset_df.iterrows():
    label = row['class']
    if label not in class_size:
        class_size[label] = 0
        test_sizes[label] = 0
    class_size[label] +=1

for index, row in dataset_df.iterrows():
    label = row['class']
    class_test_limit = test_set_fraction * class_size[label]
    if test_sizes[label] < class_test_limit:
        test_sizes[label] +=1
        test_dict["path"].append(row['path'])
        test_dict["class"].append(row['class'])
    else:
        train_dict["path"].append(row['path'])
        train_dict["class"].append(row['class'])

print("Number of records for each class: {}".format(class_size))
print("Number of records for each class in test set: {}".format(test_sizes))

training_df=pd.DataFrame(train_dict)
testing_df=pd.DataFrame(test_dict)

training_df.to_csv(os.path.join("./data","training_set.csv"), index=False)

testing_df.to_csv(os.path.join("./data","testing_set.csv"), index=False)