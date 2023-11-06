import os

current_directory = os.path.dirname(__file__)
# /root/autodl-tmp/workspace/sEMG-classification
PROJECT_PATH = os.path.dirname(current_directory)

# /root/autodl-tmp/workspace/sEMG-classification/runs/data
DATA_PATH = os.path.join(PROJECT_PATH, 'runs/data')
ACC_LOSS_PATH = os.path.join(PROJECT_PATH, 'runs/acc_loss')
MODELS_PATH = os.path.join(PROJECT_PATH, 'runs/models')

SEMG_FEATURE_SELECT_PATH = os.path.join(PROJECT_PATH, 'runs/feature_select/semg')
ACC_FEATURE_SELECT_PATH = os.path.join(PROJECT_PATH, 'runs/feature_select/acc')

def create_dir(p):
    if os.path.exists(p):
        return p
    os.makedirs(p)
    return p


def exist_path(p):
    return os.path.exists(p)
# print(DATA_PATH)