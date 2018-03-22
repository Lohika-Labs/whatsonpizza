import os

BASEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '')
PROJECT_BASE = os.path.join(BASEDIR, '..', '')
DATASET_BASE = os.path.join(BASEDIR, '..', 'dataset_18k')
TAXONOMY_DIR = os.path.join(BASEDIR, '..', 'taxonomy')
TAXONOMY_FILE = os.path.join(TAXONOMY_DIR, 'pizza_types.json')
TAXONOMY_IMAGES = os.path.join(TAXONOMY_DIR, 'images', '')
