import os

current_directory = os.path.dirname(__file__)
# /root/autodl-tmp/workspace/sEMG-classification
PROJECT_PATH = os.path.dirname(current_directory)


def create_dir(p):
    if os.path.exists(p):
        return 
    os.makedirs(p)