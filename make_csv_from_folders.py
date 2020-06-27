from glob import glob
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--folder_path', nargs='?', const="./data/images", default="./data/images", help='The path to the folder containing the class folders whic contaon the images')
parser.add_argument('--write_path', nargs='?', const="./data/all_data.csv", default="./data/all_data.csv", help='The path to write the csv to. Include the name of the csv')

args = parser.parse_args()
FOLDER_PATH, WRITE_PATH = args.folder_path, args.write_path

image_paths = glob(FOLDER_PATH + "/*/*")
csv_dict={"path":[],"class":[]}

for img_path in image_paths:
    label =os.path.basename(os.path.dirname(img_path))
    csv_dict['path'].append(img_path)
    csv_dict['class'].append(label)

df = pd.DataFrame(csv_dict)

df.to_csv(WRITE_PATH, index=False)



