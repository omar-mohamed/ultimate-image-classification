import pandas as pd
from configs import argHandler  # Import the default arguments
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
FLAGS = argHandler()
FLAGS.setDefaults()

defualt_path = os.path.join(FLAGS.save_model_path, 'training_log.csv')

parser.add_argument('--train_log_path', nargs='?', const=defualt_path, default=defualt_path, help='The path to the training log csv')

args = parser.parse_args()

TRAIN_LOG_PATH = args.train_log_path

if not os.path.exists(TRAIN_LOG_PATH):
    print(f"No file in {TRAIN_LOG_PATH}")
    exit()

df= pd.read_csv(TRAIN_LOG_PATH)
df = df.drop(['epoch','lr','loss','val_loss'],axis=1)

ax = df.plot()
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')

plt.show()