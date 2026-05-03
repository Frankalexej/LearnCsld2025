# 0429: Equal distribution. 

# Configuration parameters
BATCH_SIZE = 32
PRE_EPOCHS = 100
POST_EPOCHS = 200
LR = 1e-4
L2_LR = 1e-4

DATASET_NAME = "data_VN"
CONDITION_SUFFIX = "equal"
CSV_PATH = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_train_{CONDITION_SUFFIX}/metadata_train_{CONDITION_SUFFIX}.csv'
CSV_PATH2 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_train_{CONDITION_SUFFIX}/metadata_train_{CONDITION_SUFFIX}.csv'
CSV_PATH3 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_test_{CONDITION_SUFFIX}/metadata_test_{CONDITION_SUFFIX}.csv'
CSV_PATH4 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_test_{CONDITION_SUFFIX}/metadata_test_{CONDITION_SUFFIX}.csv'  # NEW 20260203, I just suddenly found that we didn't have testing on L1. 

DEVICE = 'cuda'
IN_FEATURES = 15
HID_FEATURES = 4
OUT_FEATURES = 4
OUT_FEATURES_2 = 6
FORCE_ALL_LOAD = True
# Notice that because we are desigining this to be only either two identical CLs or RCCL or RCRC, so number of features do not differ. 
PRE_METHOD = "SC"
POST_METHOD = "SC"
SUPCON_TEMPERATURE = 0.07
L1_MANIPULANT_SELECT = ['i', 'u', 'e', 'o']  # L1: i u e o
L2_MANIPULANT_SELECT = ['iL', 'uL', 'ihS', 'uhS']  # L2: i u ih uh
FREEZE_FOR_L2 = False            # whether to freeze the L1 encoder when training on L2
CONSOLIDATION_METHOD = "EWC"
CONSOLIDATION_STRENGTH = 1e3   # trying stronger consolidation strength. p9 = 1e9. 

# Seed control
BASE_SEED = 20260504          # shared across the entire project
DETERMINISTIC = False         # True only if you need strict determinism

SIMILARITY = "euclidean"
RUN_NAME = '0504_SCSC_EQL_EWCp3_1e4_HID4'
RUN_TIMES_START = 1
RUN_TIMES_END = RUN_TIMES_START+6
SAMPLE_LIST = []
