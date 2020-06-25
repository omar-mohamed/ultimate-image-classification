import pandas as pd
from configs import argHandler  # Import the default arguments
import os
import matplotlib.pyplot as plt

FLAGS = argHandler()
FLAGS.setDefaults()

TRAIN_LOG_PATH = ''

if TRAIN_LOG_PATH == '':
    TRAIN_LOG_PATH =  os.path.join(FLAGS.save_model_path, 'training_log.csv')

if not os.path.exists(TRAIN_LOG_PATH):
    print(f"No file in {TRAIN_LOG_PATH}")
    exit()

df= pd.read_csv(TRAIN_LOG_PATH)
df = df.drop(['epoch','lr','loss','val_loss'],axis=1)

ax = df.plot()
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')

plt.show()