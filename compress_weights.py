from tensorflow.keras.models import model_from_json
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from utils import custom_save_model
from configs import argHandler  # Import the default arguments
import os

COMPRESSED_MODEL_NAME = ''
LOAD_MODEL_PATH = ''

FLAGS = argHandler()
FLAGS.setDefaults()
if LOAD_MODEL_PATH == '':
    LOAD_MODEL_PATH = FLAGS.load_model_path


base_model_name = os.path.basename(LOAD_MODEL_PATH)
base_model_name, file_extension = os.path.splitext(base_model_name)
base_model_name = base_model_name + "_compressed"
dir_model_path = os.path.basename(LOAD_MODEL_PATH)

if COMPRESSED_MODEL_NAME == '':
    COMPRESSED_MODEL_NAME = base_model_name

if not os.path.exists(LOAD_MODEL_PATH):
    print(f"No file in {LOAD_MODEL_PATH}")
    exit()

model = load_model(LOAD_MODEL_PATH)
model.summary()

custom_save_model(model,dir_model_path,COMPRESSED_MODEL_NAME)