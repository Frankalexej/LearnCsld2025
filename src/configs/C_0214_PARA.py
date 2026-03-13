# Configuration parameters
BATCH_SIZE = 32
PRE_EPOCHS = 100
POST_EPOCHS = 200
READ_BASE_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025'
MODEL_LOAD_BASE_PATH = '/mnt/storage/franklhtan/projects/LearnCsld2025'
WRITE_BASE_PATH = '/mnt/storage/franklhtan/projects/LearnCsld2025'
CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/metadata_train_phase1.csv'
CSV_PATH2 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase2_a/metadata_train_phase2_a.csv'
CSV_PATH3 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/metadata_test_a.csv'
CSV_PATH4 = '/mnt/storage/franklhtan/projects/LearnCsld2025/data/data_test_1/metadata_test_1.csv'
DEVICE = 'cuda'
IN_FEATURES = 51
HID_FEATURES = 4
OUT_FEATURES = 8

L1_CONSONANT_SELECT = ['s', 'c']  # L1: s vs c
L2_CONSONANT_SELECT = ['sh', 'ch']  # L2: sh vs ch

# Seed control
BASE_SEED = 20260214          # shared across the entire project
DETERMINISTIC = False         # True only if you need strict determinism

SIMILARITY = "euclidean"
RUN_NAMES = ['0312_RCRC_PARA_EWC_1e3']  # '0214_RCRC_PARA_NFZ_1e4', '0214_RCRC_PARA_NFZ_1e5', '0214_RCRC_PARA_EWC_1e4'
RUN_TIMES_START = 1
RUN_TIMES_END = RUN_TIMES_START+10