# 0203: new training config: L1 = (s-c), L2 = (sh-ch) / (s-ts) / (z-j). BS32, LR1E4, F4, RCCL, CNN, ADAM optimizer. 
# PARA: S-C -> SH-CH

# Configuration parameters
BATCH_SIZE = 32
PRE_EPOCHS = 100
POST_EPOCHS = 200
LR = 1e-3
L2_LR = 1e-3

CSV_PATH = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/metadata_train_phase1.csv'
CSV_PATH2 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase2_a/metadata_train_phase2_a.csv'
CSV_PATH3 = '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/metadata_test_a.csv'
CSV_PATH4 = '/mnt/storage/franklhtan/projects/LearnCsld2025/data/data_test_1/metadata_test_1.csv'  # NEW 20260203, I just suddenly found that we didn't have testing on L1. 

DEVICE = 'cuda'
IN_FEATURES = 51
HID_FEATURES = 4
OUT_FEATURES = 8
# Notice that because we are desigining this to be only either two identical CLs or RCCL or RCRC, so number of features do not differ. 
PRE_METHOD = "RC"
POST_METHOD = "RC"
L1_CONSONANT_SELECT = ['s', 'c']  # L1: s vs c
L2_CONSONANT_SELECT = ['tsh', 'sh']  # L2: sh vs ch
FREEZE_FOR_L2 = False            # whether to freeze the L1 encoder when training on L2
CONSOLIDATION_METHOD = "NA"
CONSOLIDATION_STRENGTH = 0.0

# Seed control
BASE_SEED = 20260214          # shared across the entire project
DETERMINISTIC = False         # True only if you need strict determinism

SIMILARITY = "euclidean"
RUN_NAME = '0214_RCRC_VERT_NFZ_1e3'
RUN_TIMES_START = 7
RUN_TIMES_END = RUN_TIMES_START+3
SAMPLE_LIST = ['/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/aca/aca_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_train_phase1/asa/asa_0001.npy',

                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/acha/acha_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/acha/acha_0002.npy',

                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/atsha/atsha_0001.npy',
                '/mnt/storage/ldl_linguistics/PhonGen2025/data_251011/data_test_a/asha/asha_0001.npy',
               ]