# 0203: new training config: L1 = (s-c), L2 = (sh-ch) / (s-ts) / (z-j). BS32, LR1E4, F4, RCCL, CNN, ADAM optimizer. 
# PARA: S-C -> SH-CH

# Configuration parameters
BATCH_SIZE = 32
PRE_EPOCHS = 100
POST_EPOCHS = 200
LR = 1e-4
L2_LR = 1e-4

DATASET_NAME = "data_VW"
CONDITION_SUFFIX = "natural"
CSV_PATH = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_train_{CONDITION_SUFFIX}/metadata_train_{CONDITION_SUFFIX}.csv'
CSV_PATH2 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_train_{CONDITION_SUFFIX}/metadata_train_{CONDITION_SUFFIX}.csv'
CSV_PATH3 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_test_{CONDITION_SUFFIX}/metadata_test_{CONDITION_SUFFIX}.csv'
CSV_PATH4 = f'/mnt/storage/franklhtan/projects/LearnCsld2025/{DATASET_NAME}/data_test_{CONDITION_SUFFIX}/metadata_test_{CONDITION_SUFFIX}.csv'  # NEW 20260203, I just suddenly found that we didn't have testing on L1. 

DEVICE = 'cuda'
IN_FEATURES = 12
HID_FEATURES = 2
OUT_FEATURES = 8
# Notice that because we are desigining this to be only either two identical CLs or RCCL or RCRC, so number of features do not differ. 
PRE_METHOD = "RC"
POST_METHOD = "RC"
L1_MANIPULANT_SELECT = ['i', 'u']  # L1: s vs c
L2_MANIPULANT_SELECT = ['i', 'u', 'ih', 'uh']  # L2: VERT still tsh vs sh. 
FREEZE_FOR_L2 = False            # whether to freeze the L1 encoder when training on L2
CONSOLIDATION_METHOD = "EWC"
CONSOLIDATION_STRENGTH = 10.0   # trying smaller consolidation strength. p3 = 1e3. 

# Seed control
BASE_SEED = 20260425          # shared across the entire project
DETERMINISTIC = False         # True only if you need strict determinism

SIMILARITY = "euclidean"
RUN_NAME = '0425_RCRC_AMP_EWCp1_1e4_HID2'
RUN_TIMES_START = 1
RUN_TIMES_END = RUN_TIMES_START+10
SAMPLE_LIST = []