# Configuration parameters
BATCH_SIZE = 32
PRE_EPOCHS = 200
POST_EPOCHS = 400
READ_BASE_PATH = '/mnt/storage/franklhtan/projects/LearnCsld2025'
MODEL_LOAD_BASE_PATH = '/mnt/storage/franklhtan/projects/LearnCsld2025'
WRITE_BASE_PATH = '/mnt/storage/franklhtan/projects/LearnCsld2025'
DATASET_NAME = "data_CSLD"
CONDITION_SUFFIX = "equal"
CSV_PATH = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_train_{CONDITION_SUFFIX}/metadata_train_{CONDITION_SUFFIX}.csv'
CSV_PATH2 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_train_{CONDITION_SUFFIX}/metadata_train_{CONDITION_SUFFIX}.csv'
CSV_PATH3 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_test_{CONDITION_SUFFIX}/metadata_test_{CONDITION_SUFFIX}.csv'
CSV_PATH4 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_test_{CONDITION_SUFFIX}/metadata_test_{CONDITION_SUFFIX}.csv'
DEVICE = 'cuda'
IN_FEATURES = 15
HID_FEATURES = 2
OUT_FEATURES = 2

L1_MANIPULANT_SELECT = ['s', 'c']  # L1: s vs c
L2_MANIPULANT_SELECT = ['tsh', 'sh']  # L2: VERT still tsh vs sh. 

# Seed control
BASE_SEED = 20260603          # shared across the entire project
DETERMINISTIC = False         # True only if you need strict determinism

SIMILARITY = "euclidean"
RUN_NAMES = ['0603_RCRC_VERT_EWCp3_1e4_HID4', '0603_RCRC_VERT_EWCn3_1e4_HID4']
WRITE_RUN_NAMES = RUN_NAMES
RUN_TIMES_START = 1
RUN_TIMES_END = RUN_TIMES_START+3
