import pandas as pd
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--csv_path', nargs='?', const="./data/all_data.csv", default="./data/all_data.csv", help='The path to the csv representing all the data')
parser.add_argument('--test_split_fraction', nargs='?',type=float, const=0.2, default=0.2, help='The split fraction for the test set. It will automatically calculate this fraction for each class')
parser.add_argument('--shuffle', nargs='?',type=bool,const=True, default=True, help='Shuffle the data or not.')



args = parser.parse_args()
CSV_PATH, SPLIT_FRACTION, SHUFFLE = args.csv_path, args.test_split_fraction, args.shuffle


dataset_df = pd.read_csv(CSV_PATH)

if SHUFFLE:
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
    class_test_limit = SPLIT_FRACTION * class_size[label]
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

base_dire = os.path.dirname(CSV_PATH)
training_df.to_csv(os.path.join(base_dire,"training_set.csv"), index=False)

testing_df.to_csv(os.path.join(base_dire,"testing_set.csv"), index=False)