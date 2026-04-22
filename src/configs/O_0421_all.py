# Configuration parameters
READ_BASE_PATH = '/mnt/storage/franklhtan/projects/LearnCsld2025/eval_outputs'
WRITE_BASE_PATH = '/mnt/storage/franklhtan/projects/LearnCsld2025/observe_outputs'
EPOCH_START = 10
EPOCH_END = 601 # changed to 201 for NPT (no-pretrain) runs, which have 200 epochs total
EPOCH_INTERVAL = 10
RUN_TIMES_START = 1
RUN_TIMES_END = 11
RUN_NAMES = ['0421_RCRC_PARA_EWCp0_1e3_HID2', '0421_RCRC_PARA_EWCp3_1e3_HID2', '0421_RCRC_VERT_EWCp0_1e3_HID2', '0421_RCRC_VERT_EWCp3_1e3_HID2']
RUN_NAMES_SHORT = RUN_NAMES