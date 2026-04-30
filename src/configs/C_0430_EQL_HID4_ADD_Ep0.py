# Configuration parameters
BATCH_SIZE = 32
PRE_EPOCHS = 100
POST_EPOCHS = 200
READ_BASE_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025'
MODEL_LOAD_BASE_PATH = '/mnt/storage/franklhtan/projects/LearnCsld2025'
WRITE_BASE_PATH = '/mnt/storage/franklhtan/projects/LearnCsld2025'
DATASET_NAME = "data_VWR"
CONDITION_SUFFIX = "equal"
CSV_PATH = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_train_{CONDITION_SUFFIX}/metadata_train_{CONDITION_SUFFIX}.csv'
CSV_PATH2 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_train_{CONDITION_SUFFIX}/metadata_train_{CONDITION_SUFFIX}.csv'
CSV_PATH3 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_test_{CONDITION_SUFFIX}/metadata_test_{CONDITION_SUFFIX}.csv'
CSV_PATH4 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_test_{CONDITION_SUFFIX}/metadata_test_{CONDITION_SUFFIX}.csv'
DEVICE = 'cuda'
IN_FEATURES = 15
HID_FEATURES = 4
OUT_FEATURES = 4

L1_MANIPULANT_SELECT = ['e', 'o', 'i', 'u', 'ih', 'uh', 'iS', 'uS', 'ihL', 'uhL']  # L1: s vs c
L2_MANIPULANT_SELECT = ['e', 'o', 'i', 'u', 'ih', 'uh', 'iS', 'uS', 'ihL', 'uhL']  # L2: VERT still tsh vs sh. 

# Seed control
BASE_SEED = 20260430          # shared across the entire project
DETERMINISTIC = False         # True only if you need strict determinism

SIMILARITY = "euclidean"
RUN_NAMES = ['0430_RCCL_EQL_EWCp0_1e4_HID4', '0430_RCRC_EQL_EWCp0_1e4_HID4']
WRITE_RUN_NAMES = RUN_NAMES
RUN_TIMES_START = 4
RUN_TIMES_END = RUN_TIMES_START+3
