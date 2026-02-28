  # train.py
import argparse
import importlib.util
import os
import torch
import torch.optim as optim
import tqdm
from model import SimpleNonlinearFCClassifier
from loss import SimpleMSE
from torch.utils.data import DataLoader
from dataset import NPYDatasetCL
import wandb
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

def sample_test_wrapper(sample_list,model,device,similarity_config,epoch):
    length = len(sample_list)
    left_index = 0
    for i in range(length//2):
        # Log start of processing pair
        #print(f"Processing pair {left_index}-{left_index+1}")
        left_name = sample_list[left_index].split('/')[-1].split('.')[0]
        right_name = sample_list[left_index+1].split('/')[-1].split('.')[0]
        #print(f"{left_name}*{right_name}")
        similarity = sample_test(sample_list[left_index], sample_list[left_index+1],model,similarity_config,device)
        wandb.log({f"{left_name}*{right_name}": similarity,"epoch": epoch})
        left_index = left_index + 2 # 0vs1, 2vs3, 4vs5

def sample_test(path1,path2,model,similarity_config,device):
    array1 = np.load(path1)
    array1_flat = array1.flatten()
    array1_input = array1_flat.reshape(1, 51)
    # array1_input = array1[1]
    tensor_input1 = torch.from_numpy(array1_input).float()
    tensor_input1 = tensor_input1.to(device)
    with torch.no_grad():
        feature1 = model.encode(tensor_input1)

    array2 = np.load(path2)    
    array2_flat = array2.flatten() 
    array2_input = array2_flat.reshape(1, 51)
    # array2_input = array2[1]
    tensor_input2 = torch.from_numpy(array2_input).float()
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

def main(config_path, run_time=0):
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

    # Load dataset
    dataset1 = NPYDatasetCL(
        csv_path=config.CSV_PATH
    )
    dataset2 = NPYDatasetCL(
        csv_path=config.CSV_PATH2
    )
    testset = NPYDatasetCL(
        csv_path=config.CSV_PATH3
    )

    dataloader1 = DataLoader(dataset1, batch_size=config.BATCH_SIZE, shuffle=True)
    dataloader2 = DataLoader(dataset2, batch_size=config.BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize model, loss, optimizer
    model = SimpleNonlinearFCClassifier(in_features=config.IN_FEATURES, hid_features=config.HID_FEATURES, out_features=config.OUT_FEATURES).to(config.DEVICE)
    similarity_config = config.SIMILARITY
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LR,
        momentum=0.0,
        weight_decay=0.0,
        nesterov=False
    )
    # optimizer = optim.Adam(model.parameters(), lr=config.LR)

    # Get and sort checkpoint files based on epoch number
    checkpoint_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')],
        key=get_epoch_number
    )

    if checkpoint_files:
        latest_checkpoint = os.path.join(save_dir, checkpoint_files[-1])
        print(f"Loading checkpoint from {latest_checkpoint}")
        model.load_state_dict(torch.load(latest_checkpoint))
        last_epoch = get_epoch_number(checkpoint_files[-1])
            
    else:
        print("No checkpoint found, starting training from scratch.")
        last_epoch = 0 
         #test the sample before the training
        sample_test_wrapper(sample_list,model,config.DEVICE,similarity_config,0)   

    start_epoch = last_epoch + 1
    first_end_epoch = min(config.PRE_EPOCHS+1, start_epoch+config.PRE_EPOCHS+config.POST_EPOCHS)
    if start_epoch < config.PRE_EPOCHS :
        print("first phase")
        print(f"first_end_epoch:{first_end_epoch}")
        for epoch in range(start_epoch, first_end_epoch):
            epoch_loss = 0.0
            model.train()
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(dataloader1, desc=f'd1_Epoch {epoch}/{start_epoch+config.PRE_EPOCHS+config.POST_EPOCHS}')):
                inputs = inputs.to(config.DEVICE)
                targets = targets.to(config.DEVICE)

                features = model(inputs)
                # features = features.unsqueeze(1)  # Add view dimension if needed
                loss = criterion(features, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

                #wandb.log({"batch_loss": loss.item(), "epoch": epoch})
            avg_loss = epoch_loss / (len(dataloader1))
            print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
            wandb.log({"train_loss": avg_loss, "epoch": epoch})
            test_loss = evaluate(model, testloader, criterion, config.DEVICE)
            wandb.log({"test_loss": test_loss, "epoch": epoch})
            
            # Save the latest checkpoint, overwriting the previous one
            checkpoint_path_latest = os.path.join(save_dir, 'checkpoint_latest.pt')
            torch.save(model.state_dict(), checkpoint_path_latest)

            # Save a checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path_epoch = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                torch.save(model.state_dict(), checkpoint_path_epoch)
            
            #test some samples
            sample_test_wrapper(sample_list,model,config.DEVICE,similarity_config,epoch+1)
    
    second_start_epoch = max(start_epoch,first_end_epoch)
    if second_start_epoch < start_epoch+config.PRE_EPOCHS+config.POST_EPOCHS:
        print("second phase:")
        print(f"second_start_epoch:{second_start_epoch}")
        for epoch in range(second_start_epoch, start_epoch+config.PRE_EPOCHS+config.POST_EPOCHS):
            epoch_loss = 0.0
            model.train()
            # for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(dataloader1, desc=f'd1_Epoch {epoch}/{last_epoch+config.EPOCHS}')):
            #     inputs = inputs.to(config.DEVICE)
            #     targets = targets.to(config.DEVICE)

            #     features = model(inputs)
            #     # features = features.unsqueeze(1)  # Add view dimension if needed
            #     loss = criterion(features, targets)

            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
            for batch_idx, (inputs, targets) in enumerate(tqdm.tqdm(dataloader2, desc=f'd2_Epoch {epoch}/{start_epoch+config.PRE_EPOCHS+config.POST_EPOCHS}')):
                inputs = inputs.to(config.DEVICE)
                targets = targets.to(config.DEVICE)

                features = model(inputs)
                # features = features.unsqueeze(1)  # Add view dimension if needed
                loss = criterion(features, targets)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

                #wandb.log({"batch_loss": loss.item(), "epoch": epoch})
            # avg_loss = epoch_loss / (len(dataloader1)+len(dataloader2))
            avg_loss = epoch_loss / (len(dataloader2))
            print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
            wandb.log({"train_loss": avg_loss, "epoch": epoch})
            test_loss = evaluate(model, testloader, criterion, config.DEVICE)
            wandb.log({"test_loss": test_loss, "epoch": epoch})
            
            # Save the latest checkpoint, overwriting the previous one
            checkpoint_path_latest = os.path.join(save_dir, 'checkpoint_latest.pt')
            torch.save(model.state_dict(), checkpoint_path_latest)

            # Save a checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path_epoch = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
                torch.save(model.state_dict(), checkpoint_path_epoch)
            
            #test some samples
            sample_test_wrapper(sample_list,model,config.DEVICE,similarity_config,epoch+1)
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script with config path')
    parser.add_argument('--config', type=str, required=True, help='Path to config.py')
    args = parser.parse_args()

    config = load_config(args.config)
    run_times_start, run_times_end = config.RUN_TIMES_START, config.RUN_TIMES_END

    for run_time in range(run_times_start, run_times_end): 
        print(f"NOW TRAINING: RUN {run_time}")
        main(args.config, run_time)