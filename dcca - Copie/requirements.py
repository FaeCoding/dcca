import matplotlib.pyplot as plt

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
from random import randint
import string
import numpy as np
from scipy import signal
import os
from sklearn.cluster import KMeans
import tensorflow as tf
import pickle

from tensorflow import keras
from keras.saving import register_keras_serializable
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, AveragePooling1D, Input, Lambda, BatchNormalization, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import layers, models
from keras.activations import *

from tqdm import tqdm

import argparse