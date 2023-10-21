import os

current_directory = os.path.dirname(__file__)
# /root/autodl-tmp/workspace/sEMG-classification
PROJECT_PATH = os.path.dirname(current_directory)

# /root/autodl-tmp/workspace/sEMG-classification/runs/data
DATA_PATH = os.path.join(PROJECT_PATH, 'runs/data')


def create_dir(p):
    if os.path.exists(p):
        return p
    os.makedirs(p)
    return p


# print(DATA_PATH)