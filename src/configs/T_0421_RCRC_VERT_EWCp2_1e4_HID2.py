# 0203: new training config: L1 = (s-c), L2 = (sh-ch) / (s-ts) / (z-j). BS32, LR1E4, F4, RCCL, CNN, ADAM optimizer. 
# VERT: S-C -> TSH-SH

# Configuration parameters
BATCH_SIZE = 32
PRE_EPOCHS = 200
POST_EPOCHS = 400
LR = 1e-3
L2_LR = 1e-4

CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/metadata_train_phase1.csv'
CSV_PATH2 = '/mnt/storage/franklhtan/projects/LearnCsld2025/data/data_train_phase2_a/data_train_phase2_a.csv'
CSV_PATH3 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/metadata_test_a.csv'
CSV_PATH4 = '/mnt/storage/franklhtan/projects/LearnCsld2025/data/data_test_1/metadata_test_1.csv'  # NEW 20260203, I just suddenly found that we didn't have testing on L1. 

DEVICE = 'cuda'
IN_FEATURES = 51
HID_FEATURES = 2
OUT_FEATURES = 8
# Notice that because we are desigining this to be only either two identical CLs or RCCL or RCRC, so number of features do not differ. 
PRE_METHOD = "RC"
POST_METHOD = "RC"
L1_CONSONANT_SELECT = ['s', 'c']  # L1: s vs c
L2_CONSONANT_SELECT = ['tsh', 'sh']  # L2: VERT still tsh vs sh. 
FREEZE_FOR_L2 = False            # whether to freeze the L1 encoder when training on L2
CONSOLIDATION_METHOD = "EWC"
CONSOLIDATION_STRENGTH = 100.0   # trying smaller consolidation strength. p2 = 1e2. 

# Seed control
BASE_SEED = 20260421          # shared across the entire project
DETERMINISTIC = False         # True only if you need strict determinism

SIMILARITY = "euclidean"
RUN_NAME = '0421_RCRC_VERT_EWCp2_1e4_HID2'
RUN_TIMES_START = 1
RUN_TIMES_END = RUN_TIMES_START+10
SAMPLE_LIST = []