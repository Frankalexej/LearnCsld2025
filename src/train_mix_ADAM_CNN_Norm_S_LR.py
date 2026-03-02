  # train.py
import argparse
import importlib.util
import os
import torch
import torch.optim as optim
import tqdm
from model import SimpleResNet1DClass, SimpleResNet1DRecon
from torch.utils.data import DataLoader
from dataset import NPYDatasetCL_CNN_Norm as ThisCLDataset
from dataset import NPYDatasetRC_CNN_Norm as ThisRCDataset
import wandb
import numpy as np
import pandas as pd
import sys
from utils_seed import *
from loss import *

"""
Mix: we can manipulate L1 and L2 learning mechanisms. Simply configure in the config file. 
Single: 20260203 new design: L1 = s-c, L2 = sh-ch or s-ts. We will handle this in config. 
"""

# Check for existing checkpoints
# Function to extract epoch number from filename
def get_epoch_number(filename):
    # Assumes filename format: 'checkpoint_epoch_<number>.pt'
    return int(filename.split('_')[-1].split('.')[0])

def epoch_in_pre(epoch, pre): 
    if epoch <= pre: 
        return True
    else: 
        return False
    
def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True

def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def sample_test_wrapper(sample_list,model,device,similarity_config,epoch, global_mean=1.0):
    length = len(sample_list)
    left_index = 0
    for i in range(length//2):
        # Log start of processing pair
        #print(f"Processing pair {left_index}-{left_index+1}")
        left_name = sample_list[left_index].split('/')[-1].split('.')[0]
        right_name = sample_list[left_index+1].split('/')[-1].split('.')[0]
        #print(f"{left_name}*{right_name}")
        similarity = sample_test(sample_list[left_index], sample_list[left_index+1],model,similarity_config,device, global_mean)
        wandb.log({f"{left_name}*{right_name}": similarity,"epoch": epoch})
        left_index = left_index + 2 # 0vs1, 2vs3, 4vs5

def sample_test(path1,path2,model,similarity_config,device, global_mean=1.0):
    array1 = np.load(path1)
    array1_flat = array1.flatten()
    array1_input = array1_flat.reshape(1, 1, 51)
    tensor_input1 = torch.from_numpy(array1_input).float()
    tensor_input1 = tensor_input1 / global_mean  # normalize by dataset global mean
    tensor_input1 = tensor_input1.to(device)
    with torch.no_grad():
        feature1 = model.encode(tensor_input1)

    array2 = np.load(path2)    
    array2_flat = array2.flatten() 
    array2_input = array2_flat.reshape(1, 1, 51)
    tensor_input2 = torch.from_numpy(array2_input).float()
    tensor_input2 = tensor_input2 / global_mean  # normalize by dataset global mean
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_input2 = tensor_input2.to(device)
    with torch.no_grad():
        feature2 = model.encode(tensor_input2)

    # print(feature1, feature2)
    
    if similarity_config == 'cosine':
        feat1 = feature1.cpu().numpy().flatten()
        norm_feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feature2.cpu().numpy().flatten()
        norm_feat2 = feat2 / np.linalg.norm(feat2)
        cosine_sim = np.dot(norm_feat1, norm_feat2)
        similarity = cosine_sim.item()
    if similarity_config == 'euclidean':
        feat1 = feature1.flatten()
        feat2 = feature2.flatten()
        dist_squared = torch.sum(feat1 ** 2) - 2 * torch.dot(feat1, feat2) + torch.sum(feat2 ** 2)
        euclidean_dist = torch.sqrt(torch.clamp(dist_squared, min=1e-8))
        similarity = euclidean_dist.cpu().numpy().item()
    return similarity


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            features = model(inputs)
            # features = features.unsqueeze(1)
            loss = criterion(features, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    print(f'Validation Loss: {avg_loss:.4f}')
    return avg_loss

def main(config_path, run_time=0, this_seed=0):
    config = load_config(config_path)
    
    #weight dir
    save_dir = os.path.join('..', 'weights', config.RUN_NAME, f"{run_time}")
    os.makedirs(save_dir, exist_ok=True)
    
    #sample test
    sample_list = config.SAMPLE_LIST

    # Initialize wandb
    wandb.init(
        project="LearnCsld2025",
        name=config.RUN_NAME + f"_{run_time}",
        reinit=True  # allow reinitializing within the same process
    )

    # Here we collect the global means, this collection ensures that both L1 and L2 share the same normalization statistics. 
    df1 = pd.read_csv(config.CSV_PATH)
    df2 = pd.read_csv(config.CSV_PATH2)
    eps = 1e-9
    df = pd.concat([df1, df2], ignore_index=True)
    # get the by dimension mean of dataset, in normal scale, use numpy for easier calculation
    data_sum = None
    count = 0

    for _, row in df.iterrows():
        x = np.load(row['path'])          # (3, 17)
        x = x.reshape(-1)                 # (51,)
        x = x + eps                    # avoid dividing by 0

        if data_sum is None:
            data_sum = x.copy()
        else:
            data_sum += x
        count += 1

    global_mean = torch.tensor(
        data_sum / count, dtype=torch.float32
    )


    # PreMethod, PostMethod
    pre_method = config.PRE_METHOD  # NOTE: This shall not be customized unless it is CL->CL, but just in case we will need to in the future. Currently, this function is not supported by the model IN TERMS OF LEARNING QUALITY (not in terms of run time error): the main concern is that for CNN, the AE decoder is much larger than a simple CL decoder, which means, if we first train RC then CL, it is okay, but the reverse might not work properly simply because in the reverse way, the huge decoder for RC is supposedly much harder to train than CL decoder. In addition, it is not so justifiable to do the CL->RC training in terms of L1->L2. But CL->CL, RC->RC should be working. This is also the reason why we have to include pre_method customization. 
    post_method = config.POST_METHOD

    L1_consonant_select = config.L1_CONSONANT_SELECT  # e.g., ['s','c']
    L2_consonant_select = config.L2_CONSONANT_SELECT  # e.g., ['sh','ch'] or ['s','ts'] or ['z', 'j']

    freeze_for_L2 = config.FREEZE_FOR_L2  # whether to freeze encoder when training L2 (only designated layers)
    consolidation_method = config.CONSOLIDATION_METHOD  # currently either EWC or no consolidation (freezing is controlled separately)

    l1_lr = config.LR
    l2_lr = config.L2_LR

    print(f"PARAMS: L1={L1_consonant_select}; L2={L2_consonant_select}; freeze={freeze_for_L2}. ")

    # Load dataset
    if pre_method == "RC": 
        dataset1 = ThisRCDataset(
            csv_path=config.CSV_PATH, 
            global_mean=global_mean, 
            consonant_select=L1_consonant_select
        )
        testset1 = ThisRCDataset(
            csv_path=config.CSV_PATH4, # test_data_1, corresponding to train_data_phase1
            global_mean=global_mean, 
            consonant_select=L1_consonant_select
        )
        model1 = SimpleResNet1DRecon(hid_features=config.HID_FEATURES).to(config.DEVICE)
        criterion1 = torch.nn.MSELoss(reduction="mean") # Using MSE for reconstruction. 
        optimizer1 = optim.Adam(model1.parameters(), lr=l1_lr)
    elif pre_method == "CL": 
        dataset1 = ThisCLDataset(
            csv_path=config.CSV_PATH, 
            global_mean=global_mean, 
            consonant_select=L1_consonant_select
        )
        testset1 = ThisCLDataset(
            csv_path=config.CSV_PATH4, 
            global_mean=global_mean, 
            consonant_select=L1_consonant_select
        )
        model1 = SimpleResNet1DClass(hid_features=config.HID_FEATURES, out_features=config.OUT_FEATURES).to(config.DEVICE)
        criterion1 = torch.nn.CrossEntropyLoss(reduction="mean")
        optimizer1 = optim.Adam(model1.parameters(), lr=l1_lr)
    else: 
        raise AssertionError("Pre method not existing! ")
    
    if post_method == "RC": 
        dataset2 = ThisRCDataset(
            csv_path=config.CSV_PATH2, 
            global_mean=global_mean, 
            consonant_select=L2_consonant_select
        )
        testset2 = ThisRCDataset(
            csv_path=config.CSV_PATH3, 
            global_mean=global_mean, 
            consonant_select=L2_consonant_select
        )
        model2 = SimpleResNet1DRecon(hid_features=config.HID_FEATURES).to(config.DEVICE)
        criterion2 = torch.nn.MSELoss(reduction="mean") # Using MSE for reconstruction. 
        optimizer2 = optim.Adam(model2.parameters(), lr=l2_lr)
    elif post_method == "CL": 
        dataset2 = ThisCLDataset(
            csv_path=config.CSV_PATH2, 
            global_mean=global_mean, 
            consonant_select=L2_consonant_select
        )
        testset2 = ThisCLDataset(
            csv_path=config.CSV_PATH3, 
            global_mean=global_mean, 
            consonant_select=L2_consonant_select
        )
        model2 = SimpleResNet1DClass(hid_features=config.HID_FEATURES, out_features=config.OUT_FEATURES).to(config.DEVICE)
        criterion2 = torch.nn.CrossEntropyLoss(reduction="mean") # Using MSE for reconstruction. 
        optimizer2 = optim.Adam(model2.parameters(), lr=l2_lr)
    else: 
        raise AssertionError("Post method not existing! ")

    dataloader1 = make_loader(dataset1, batch_size=config.BATCH_SIZE, shuffle=True, seed=this_seed)
    dataloader2 = make_loader(dataset2, batch_size=config.BATCH_SIZE, shuffle=True, seed=this_seed)
    testloader1 = make_loader(testset1, batch_size=config.BATCH_SIZE, shuffle=False, seed=this_seed+1)
    testloader2 = make_loader(testset2, batch_size=config.BATCH_SIZE, shuffle=False, seed=this_seed+1)

    # here we deal with EWC settings
    if consolidation_method == "EWC": 
        print("EWC activated")
        fimloader = make_loader(dataset1, batch_size=config.BATCH_SIZE, shuffle=True, seed=this_seed)   # only have dataset1, because fim is calculated based on L1 learning results
        ewc = EWC(
            dataloader=fimloader, 
            ewc_lambda=config.CONSOLIDATION_STRENGTH, 
            estimate_sample_size=None, 
            device=config.DEVICE
            )
        
    # In this way, we can designate the training methods of L1 and L2. 

    # Initialize model, loss, optimizer
    # model = SimpleResNet1DRecon(hid_features=config.HID_FEATURES).to(config.DEVICE)
    similarity_config = config.SIMILARITY
    
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=config.LR,
    #     momentum=0.0,
    #     weight_decay=0.0,
    #     nesterov=False
    # )
    # optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # Get and sort checkpoint files based on epoch number
    checkpoint_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')],
        key=get_epoch_number
    )

    if checkpoint_files:
        latest_checkpoint = os.path.join(save_dir, checkpoint_files[-1])
        print(f"Loading checkpoint from {latest_checkpoint}")
        last_epoch = get_epoch_number(checkpoint_files[-1])
        if epoch_in_pre(last_epoch, config.PRE_EPOCHS): 
            model1.load_state_dict(torch.load(latest_checkpoint))
        else: 
            model2.load_state_dict(torch.load(latest_checkpoint))
            
    else:
        print("No checkpoint found, starting training from scratch.")
        last_epoch = 0 
         #test the sample before the training
        sample_test_wrapper(sample_list,model1,config.DEVICE,similarity_config,0, global_mean=global_mean)   

    start_epoch = last_epoch + 1
    first_end_epoch = min(config.PRE_EPOCHS+1, start_epoch+config.PRE_EPOCHS+config.POST_EPOCHS)
    if start_epoch <= config.PRE_EPOCHS :
        print("first phase")
        print(f"first_end_epoch:{first_end_epoch}")
        for epoch in range(start_epoch, first_end_epoch):
            epoch_loss = 0.0
            model1.train()
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(dataloader1, desc=f'd1_Epoch {epoch}/{start_epoch+config.PRE_EPOCHS+config.POST_EPOCHS}')):
                inputs = inputs.to(config.DEVICE)
                targets = targets.to(config.DEVICE)

                features = model1(inputs)
                # features = features.unsqueeze(1)  # Add view dimension if needed
                loss = criterion1(features, targets)

                optimizer1.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
                optimizer1.step()

                epoch_loss += loss.item()

                #wandb.log({"batch_loss": loss.item(), "epoch": epoch})
            avg_loss = epoch_loss / (len(dataloader1))
            print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
            wandb.log({"train_loss": avg_loss, "epoch": epoch})
            test_loss = evaluate(model1, testloader1, criterion1, config.DEVICE)
            wandb.log({"test_loss": test_loss, "epoch": epoch})
            
            # Save the latest checkpoint, overwriting the previous one
            checkpoint_path_latest = os.path.join(save_dir, 'checkpoint_latest.pt')
            torch.save(model1.state_dict(), checkpoint_path_latest)

            # Save a checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path_epoch = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                torch.save(model1.state_dict(), checkpoint_path_epoch)
            
            #test some samples
            sample_test_wrapper(sample_list,model1,config.DEVICE,similarity_config,epoch+1, global_mean=global_mean)
    
    if start_epoch <= config.PRE_EPOCHS + 1: # +1 allows the case of stopping between after 100 and before 110
        # NOTE: here we transfer the model. For good formatting, we will transfer regardless of whether pre and post shares learning method. 
        # NOTE: THIS IS AN ASSUMPTION, CHECK AFTER RUNNING: here we supposedly only load the encoder part, REGARDLESS of whether or not pre and post share model structure. This is to be fair to the case where they do not share model structure, so that training on L2 does not give advantage to not changing model structure. 
        state = model1.state_dict()
        encoder_prefixes = ("layer1.", "layer2.", "layer3.", "fc.")
        enc_state = {k: v for k, v in state.items() if k.startswith(encoder_prefixes)}
        model2.load_state_dict(enc_state, strict=False)  # strict=False: otherwise it will check whether the whole target model is matched. 
        print("Model 1 encoder loaded to Model 2. ")

        if freeze_for_L2: 
            # Freeze the encoder
            freeze_module(model2.layer1)
            freeze_module(model2.layer2)
            freeze_module(model2.layer3)
            print("Freeze Conducted. ")
            # freeze_module(model2.fc)

        # Build optimizer on trainable params only
        optimizer2 = torch.optim.Adam(
            (p for p in model2.parameters() if p.requires_grad),
            lr=l2_lr,
        )

        if consolidation_method == "EWC": 
            print("EWC params loaded")
            fim, old_params = ewc.calculate_fim(model1, criterion1, optimizer1)
            ewc.fim = fim
            ewc.old_params = old_params

    second_start_epoch = max(start_epoch,first_end_epoch)
    if second_start_epoch < start_epoch+config.PRE_EPOCHS+config.POST_EPOCHS:
        print("second phase:")
        print(f"second_start_epoch:{second_start_epoch}")
        for epoch in range(second_start_epoch, start_epoch+config.PRE_EPOCHS+config.POST_EPOCHS):
            epoch_loss = 0.0
            model2.train()
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(dataloader2, desc=f'd2_Epoch {epoch}/{start_epoch+config.PRE_EPOCHS+config.POST_EPOCHS}')):
                inputs = inputs.to(config.DEVICE)
                targets = targets.to(config.DEVICE)

                features = model2(inputs)
                # features = features.unsqueeze(1)  # Add view dimension if needed
                if consolidation_method == "EWC": 
                    loss = criterion2(features, targets) + ewc.penalty(model2)
                else: 
                    loss = criterion2(features, targets)

                optimizer2.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
                optimizer2.step()

                epoch_loss += loss.item()

                #wandb.log({"batch_loss": loss.item(), "epoch": epoch})
            # avg_loss = epoch_loss / (len(dataloader1)+len(dataloader2))
            avg_loss = epoch_loss / (len(dataloader2))
            print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
            wandb.log({"train_loss": avg_loss, "epoch": epoch})
            test_loss = evaluate(model2, testloader2, criterion2, config.DEVICE)
            wandb.log({"test_loss": test_loss, "epoch": epoch})
            
            # Save the latest checkpoint, overwriting the previous one
            checkpoint_path_latest = os.path.join(save_dir, 'checkpoint_latest.pt')
            torch.save(model2.state_dict(), checkpoint_path_latest)

            # Save a checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path_epoch = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                torch.save(model2.state_dict(), checkpoint_path_epoch)
            
            #test some samples
            sample_test_wrapper(sample_list,model2,config.DEVICE,similarity_config,epoch+1, global_mean=global_mean)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script with config path')
    parser.add_argument('--config', type=str, required=True, help='Path to config.py')
    args = parser.parse_args()

    config = load_config(args.config)
    run_times_start, run_times_end = config.RUN_TIMES_START, config.RUN_TIMES_END
    run_name = config.RUN_NAME

    for run_time in range(run_times_start, run_times_end): 
        print(f"NOW TRAINING: RUN {run_time}")
        # 1) derive per-job seed
        seed = make_seed(config.BASE_SEED, 
                         condition="", 
                         run_id=run_time)
        # 2) apply it early
        seed_everything(seed, deterministic=config.DETERMINISTIC)

        print(f"[SEED] BASE_SEED={config.BASE_SEED} condition={run_name} run_id={run_time} -> seed={seed}")
        
        main(args.config, run_time, this_seed=seed)