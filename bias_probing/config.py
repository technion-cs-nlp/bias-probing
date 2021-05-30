# Global directory configuration for the project
#
import os
from os.path import join

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = join(ROOT_DIR, '../models')
CACHE_DIR = join(ROOT_DIR, '../cache')
DATA_DIR = join(ROOT_DIR, '../data')
TEMP_DIR = join(ROOT_DIR, '../temp')
FIGURES_DIR = join(ROOT_DIR, '../figures')
RESULTS_DIR = join(ROOT_DIR, '../results')
ENCODED_DATA_DIR = join(CACHE_DIR, '_encodings')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ENCODED_DATA_DIR, exist_ok=True)
