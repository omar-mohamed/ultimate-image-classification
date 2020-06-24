from glob import glob
import pandas as pd
import os


path = "./data/images"
image_paths = glob(path+"/*/*")
csv_dict={"path":[],"class":[]}
write_path = './data/all_data.csv'

for img_path in image_paths:
    label =os.path.basename(os.path.dirname(img_path))
    csv_dict['path'].append(img_path)
    csv_dict['class'].append(label)

df = pd.DataFrame(csv_dict)

df.to_csv(write_path, index=False)



