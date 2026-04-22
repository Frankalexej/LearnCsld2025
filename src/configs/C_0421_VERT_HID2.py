# Configuration parameters
BATCH_SIZE = 32
PRE_EPOCHS = 200
POST_EPOCHS = 400
READ_BASE_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025'
MODEL_LOAD_BASE_PATH = '/mnt/storage/franklhtan/projects/LearnCsld2025'
WRITE_BASE_PATH = '/mnt/storage/franklhtan/projects/LearnCsld2025'
CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/metadata_train_phase1.csv'
CSV_PATH2 = '/mnt/storage/franklhtan/projects/LearnCsld2025/data/data_train_phase2_a/data_train_phase2_a.csv'
CSV_PATH3 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/metadata_test_a.csv'
CSV_PATH4 = '/mnt/storage/franklhtan/projects/LearnCsld2025/data/data_test_1/metadata_test_1.csv'
DEVICE = 'cuda'
IN_FEATURES = 51
HID_FEATURES = 2
OUT_FEATURES = 8

L1_CONSONANT_SELECT = ['s', 'c']  # L1: s vs c
L2_CONSONANT_SELECT = ['tsh', 'sh']  # L2: sh vs ch

# Seed control
BASE_SEED = 20260421          # shared across the entire project
DETERMINISTIC = False         # True only if you need strict determinism

SIMILARITY = "euclidean"
RUN_NAMES = ['0421_RCRC_VERT_EWCp0_1e3_HID2', '0421_RCRC_VERT_EWCp3_1e3_HID2']
RUN_TIMES_START = 1
RUN_TIMES_END = RUN_TIMES_START+10