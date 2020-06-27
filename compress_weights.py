from tensorflow.keras.models import model_from_json
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from utils import custom_save_model
from configs import argHandler  # Import the default arguments
import os
import argparse


FLAGS = argHandler()
FLAGS.setDefaults()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', nargs='?', const=FLAGS.load_model_path, default=FLAGS.load_model_path, help='The path to the model')

args = parser.parse_args()
LOAD_MODEL_PATH = args.model_path



base_model_name = os.path.basename(LOAD_MODEL_PATH)
base_model_name, file_extension = os.path.splitext(base_model_name)
base_model_name = base_model_name + "_compressed"
dir_model_path = os.path.basename(LOAD_MODEL_PATH)


if not os.path.exists(LOAD_MODEL_PATH):
    print(f"No file in {LOAD_MODEL_PATH}")
    exit()

model = load_model(LOAD_MODEL_PATH)
model.summary()

custom_save_model(model,dir_model_path, base_model_name)