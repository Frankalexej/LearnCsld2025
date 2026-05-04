import argparse
import importlib.util
import os
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from model import SimpleResNet1Dv2 as ThisEncoder
from torch.utils.data import DataLoader
from dataset import NPYDatasetInfoCollect_AMP_CNN as CollectDataset
import numpy as np
import sys

# Check for existing checkpoints
# Function to extract epoch number from filename
def get_epoch_number(filename):
    # Assumes filename format: 'checkpoint_epoch_<number>.pt'
    return int(filename.split('_')[-1].split('.')[0])

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            features = model(inputs)
            features = features.unsqueeze(1)
            loss = criterion(features, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss

def batch_meta_to_df(batch_meta: dict) -> pd.DataFrame:
    """
    Convert a batch metadata dict (lists + tensors) into a pandas DataFrame.
    """
    clean_dict = {}
    for k, v in batch_meta.items():
        if torch.is_tensor(v):
            # 1D tensor -> list of scalars
            clean_dict[k] = v.detach().cpu().tolist()
        else:
            # already list of strings, bools, etc.
            clean_dict[k] = v
    return pd.DataFrame(clean_dict)

@torch.no_grad()
def evaluate_collect_outputs(
    model,
    data_loader,
    device,
    npy_path="outputs.npy",
    csv_path="outputs_meta.csv",
    to_float32=True
):
    """
    Runs the model over data_loader, stacks all output vectors into a single .npy file,
    and writes a CSV with metadata + the index of each vector in the .npy array.

    Assumptions:
      - Each batch item is a dict with {input_key: tensor, ...metadata...}.
      - model(inputs) returns a 2D tensor [B, D] (if 1D, it will be unsqueezed).
    """
    model.eval()
    model.to(device)

    all_vecs = []
    csv_frames = []   # list to hold batch-level DataFrames

    for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(data_loader)):
        inputs = inputs.to(device)

        # forward pass -> vectors
        vec = model(inputs)  # expected [B, D]
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)

        vec = vec.detach().cpu()
        if to_float32 and vec.dtype != torch.float32:
            vec = vec.float()

        # append vectors
        all_vecs.append(vec)

        # append metadata DataFrame
        batch_df = batch_meta_to_df(targets)
        csv_frames.append(batch_df)

    if not all_vecs:
        raise RuntimeError("No vectors collected. Check your loader and input_key.")
    mat = torch.cat(all_vecs, dim=0).numpy()
    np.save(npy_path, mat)

    # save CSV
    csv_df = pd.concat(csv_frames, ignore_index=True)
    csv_df.to_csv(csv_path, index=False)

    print(f"Saved vectors: {mat.shape} -> {os.path.abspath(npy_path)}")
    print(f"Saved metadata rows: {len(csv_df)} -> {os.path.abspath(csv_path)}")

    return


def main(config_path, run_name=None, write_run_name=None, run_time=0):
    config = load_config(config_path)
    if run_name is not None:
        config.RUN_NAME = run_name
    else: 
        raise ValueError("Please provide run_name argument.")
    if write_run_name is not None:
        config.WRITE_RUN_NAME = write_run_name
    else: 
        config.WRITE_RUN_NAME = config.RUN_NAME

    print("Running evaluation for:", config.RUN_NAME)
    
    #weight dir
    save_dir = os.path.join(config.MODEL_LOAD_BASE_PATH, 'weights', config.RUN_NAME, f'{run_time}')
    # os.makedirs(save_dir, exist_ok=True)
    eval_save_dir_hyper = os.path.join(config.WRITE_BASE_PATH, 'eval_outputs', config.WRITE_RUN_NAME, f'{run_time}')
    os.makedirs(eval_save_dir_hyper, exist_ok=True)

    global_mean = 1.0

    L1_manipulant_select = config.L1_MANIPULANT_SELECT  # e.g., ['s','c']
    L2_manipulant_select = config.L2_MANIPULANT_SELECT  # e.g., ['sh','ch'] or ['s','ts'] or ['z', 'j']

    # Load dataset
    dataset_L1 = CollectDataset(
        csv_path=config.CSV_PATH4,  # L1 testing set
        global_mean=global_mean, 
        manipulant_select=L1_manipulant_select
    )
    dataset_L2 = CollectDataset(
        csv_path=config.CSV_PATH3,  # L2 testing set
        global_mean=global_mean, 
        manipulant_select=L2_manipulant_select
    )
    dataloader_L1 = DataLoader(dataset_L1, batch_size=config.BATCH_SIZE, shuffle=False)
    dataloader_L2 = DataLoader(dataset_L2, batch_size=config.BATCH_SIZE, shuffle=False)

    # PreMethod, PostMethod
    # pre_method = config.PRE_METHOD  # NOTE: This shall not be customized unless it is CL->CL, but just in case we will need to in the future. Currently, this function is not supported by the model IN TERMS OF LEARNING QUALITY (not in terms of run time error): the main concern is that for CNN, the AE decoder is much larger than a simple CL decoder, which means, if we first train RC then CL, it is okay, but the reverse might not work properly simply because in the reverse way, the huge decoder for RC is supposedly much harder to train than CL decoder. In addition, it is not so justifiable to do the CL->RC training in terms of L1->L2. But CL->CL, RC->RC should be working. This is also the reason why we have to include pre_method customization. 
    # post_method = config.POST_METHOD
    # similarity_config = config.SIMILARITY

    # Initialize model, loss, optimizer
    model = ThisEncoder(out_features=config.HID_FEATURES).to(config.DEVICE)

    # Get and sort checkpoint files based on epoch number
    checkpoint_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')],
        key=get_epoch_number
    )
    if checkpoint_files:
        for check_point_file in checkpoint_files: 
            this_checkpoint = os.path.join(save_dir, check_point_file)
            print(f"Loading checkpoint from {this_checkpoint}")
            model.load_state_dict(torch.load(this_checkpoint), strict=False)
            this_epoch = get_epoch_number(check_point_file)
            # eval_save_dir = os.path.join(eval_save_dir_hyper, f"epoch_{this_epoch}")
            eval_save_dir = eval_save_dir_hyper
            os.makedirs(eval_save_dir, exist_ok=True)
            # Run evaluation and collect outputs
            # evaluate_collect_outputs(
            #     model=model,
            #     data_loader=dataloader_L1,
            #     device=config.DEVICE,
            #     npy_path=os.path.join(eval_save_dir, f"vec_E{this_epoch}_L1.npy"),
            #     csv_path=os.path.join(eval_save_dir, f"meta_E{this_epoch}_L1.csv"),
            #     to_float32=True
            # )
            evaluate_collect_outputs(
                model=model,
                data_loader=dataloader_L2,
                device=config.DEVICE,
                npy_path=os.path.join(eval_save_dir, f"vec_E{this_epoch}_L2.npy"),
                csv_path=os.path.join(eval_save_dir, f"meta_E{this_epoch}_L2.csv"),
                to_float32=True
            )
    else:
        print("No checkpoint found, end running evaluation.")
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script with config path')
    parser.add_argument('--config', type=str, required=True, help='Path to config.py')
    args = parser.parse_args()

    pre_load_config = load_config(args.config)
    if not hasattr(pre_load_config, 'RUN_NAMES') or not hasattr(pre_load_config, 'RUN_TIMES_START') or not hasattr(pre_load_config, 'RUN_TIMES_END'):
        raise ValueError("Config file must define RUN_NAMES, RUN_TIMES_START, and RUN_TIMES_END.")
    if not hasattr(pre_load_config, 'WRITE_RUN_NAMES'):
        pre_load_config.WRITE_RUN_NAMES = pre_load_config.RUN_NAMES

    for run_name, write_run_name in zip(pre_load_config.RUN_NAMES, pre_load_config.WRITE_RUN_NAMES):
        for run_time in range(pre_load_config.RUN_TIMES_START, pre_load_config.RUN_TIMES_END):
            print("Collecting: ", run_name, run_time)
            main(args.config, run_name=run_name, write_run_name=write_run_name, run_time=run_time)