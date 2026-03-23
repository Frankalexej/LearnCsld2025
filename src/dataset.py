import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class NPYDataset(Dataset):
    """TODO: 20251221 Under construction, must check before running. """
    def __init__(self, csv_path):
        """
        Args:
            csv_path (str): Path to the CSV metadata file.
            base_path (str): Base directory where .npy files are stored.
            max_samples (int): Maximum number of samples to load.
            train_only (bool): If True, load only training data; else test data.
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        #df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data


        # Store DataFrame
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row['path'])  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D NOTE: currently we borrow the VCV structure, anyway only C is changing. 
        # data_tensor = data_tensor.unsqueeze(0)  # add channel dimension (1, N)
        # Now instead of using VCV, for testing, I will use only the C. 
        # data_tensor = data_tensor    # Just take the second element. 

        label = row['word']
        return data_tensor, data_tensor

class NPYDatasetCL(Dataset):
    """TODO: 20251221 Under construction, must check before running. """
    def __init__(self, csv_path, consonant_select=None):
        """
        Args:
            csv_path (str): Path to the CSV metadata file.
            base_path (str): Base directory where .npy files are stored.
            max_samples (int): Maximum number of samples to load.
            train_only (bool): If True, load only training data; else test data.
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        #df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data
        # NOTE: new feture 20260203 to select only specific consonant data.
        if consonant_select is not None: 
            df = df[df['consonant'].isin(consonant_select)].reset_index(drop=True)

        # Store DataFrame
        self.df = df

        unique_labels = ["asa", "aca", "atsa", "atca", "asha", "acha", "atsha", "atcha"]
        self.label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
        self.index_to_label = {i: lab for lab, i in self.label_to_index.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row['path'])  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D NOTE: currently we borrow the VCV structure, anyway only C is changing. 
        # data_tensor = data_tensor.unsqueeze(0)  # add channel dimension (1, N)
        # Now instead of using VCV, for testing, I will use only the C. 
        # data_tensor = data_tensor    # Just take the second element. 

        label_str = row['word']
        label = self.label_to_index[label_str]
        return data_tensor, label

class NPYDatasetCL_CNN(Dataset):
    """CNN version simply adds a channel dimension. """
    def __init__(self, csv_path, consonant_select=None):
        """
        Args:
            csv_path (str): Path to the CSV metadata file.
            base_path (str): Base directory where .npy files are stored.
            max_samples (int): Maximum number of samples to load.
            train_only (bool): If True, load only training data; else test data.
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        #df = df.sample(frac=1).reset_index(drop=True)  # Shuffle data
        # NOTE: new feture 20260203 to select only specific consonant data.
        if consonant_select is not None:
            df = df[df['consonant'].isin(consonant_select)].reset_index(drop=True)

        # Store DataFrame
        self.df = df

        unique_labels = ["asa", "aca", "atsa", "atca", "asha", "acha", "atsha", "atcha"]
        self.label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
        self.index_to_label = {i: lab for lab, i in self.label_to_index.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row['path'])  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D NOTE: currently we borrow the VCV structure, anyway only C is changing. 
        data_tensor = data_tensor.unsqueeze(0)  # add channel dimension (1, N)
        # Now instead of using VCV, for testing, I will use only the C. 
        # data_tensor = data_tensor    # Just take the second element. 

        label_str = row['word']
        label = self.label_to_index[label_str]
        return data_tensor, label
    
class NPYDatasetCL_CNN_Norm(Dataset):
    """CNN version simply adds a channel dimension. """
    def __init__(self, csv_path, global_mean=1.0, consonant_select=None):
        """
        Args:
            csv_path (str): Path to the CSV metadata file.
            base_path (str): Base directory where .npy files are stored.
            max_samples (int): Maximum number of samples to load.
            train_only (bool): If True, load only training data; else test data.

            NOTE: currently normalization is not implemented
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        # NOTE: new feture 20260203 to select only specific consonant data.
        if consonant_select is not None:
            df = df[df['consonant'].isin(consonant_select)].reset_index(drop=True)

        # Store DataFrame
        self.df = df
        self.global_mean = global_mean

        # unique_labels = ["asa", "aca", "atsa", "atca", "asha", "acha", "atsha", "atcha"]
        unique_labels = consonant_select
        self.label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
        self.index_to_label = {i: lab for lab, i in self.label_to_index.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row['path'])  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D NOTE: currently we borrow the VCV structure, anyway only C is changing. 
        data_tensor = data_tensor / self.global_mean  # normalize by dataset global mean
        data_tensor = data_tensor.unsqueeze(0)  # add channel dimension (1, N)
        # Now instead of using VCV, for testing, I will use only the C. 
        # data_tensor = data_tensor    # Just take the second element. 

        label_str = row['consonant']
        label = self.label_to_index[label_str]
        return data_tensor, label

class NPYDatasetRC_CNN(Dataset):
    """CNN version simply adds a channel dimension. """
    def __init__(self, csv_path, consonant_select=None):
        """
        Args:
            csv_path (str): Path to the CSV metadata file.
            base_path (str): Base directory where .npy files are stored.
            max_samples (int): Maximum number of samples to load.
            train_only (bool): If True, load only training data; else test data.
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        # NOTE: new feture 20260203 to select only specific consonant data.
        if consonant_select is not None:
            df = df[df['consonant'].isin(consonant_select)].reset_index(drop=True)

        # Store DataFrame
        self.df = df

        unique_labels = ["asa", "aca", "atsa", "atca", "asha", "acha", "atsha", "atcha"]
        self.label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
        self.index_to_label = {i: lab for lab, i in self.label_to_index.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row['path'])  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D NOTE: currently we borrow the VCV structure, anyway only C is changing. 
        data_tensor = data_tensor.unsqueeze(0)  # add channel dimension (1, N)
        # Now instead of using VCV, for testing, I will use only the C. 
        # data_tensor = data_tensor    # Just take the second element. 

        label_str = row['word']
        label = self.label_to_index[label_str]
        return data_tensor, data_tensor
    

class NPYDatasetRC_CNN_Norm(Dataset):
    """CNN version simply adds a channel dimension. """
    def __init__(self, csv_path, global_mean=1.0, consonant_select=None):
        """
        Args:
            csv_path (str): Path to the CSV metadata file.
            base_path (str): Base directory where .npy files are stored.
            max_samples (int): Maximum number of samples to load.
            train_only (bool): If True, load only training data; else test data.
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        # NOTE: new feture 20260203 to select only specific consonant data.
        if consonant_select is not None:
            df = df[df['consonant'].isin(consonant_select)].reset_index(drop=True)
        # Store DataFrame
        self.df = df
        self.global_mean = global_mean
        unique_labels = ["asa", "aca", "atsa", "atca", "asha", "acha", "atsha", "atcha"]
        self.label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
        self.index_to_label = {i: lab for lab, i in self.label_to_index.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row['path'])  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D NOTE: currently we borrow the VCV structure, anyway only C is changing. 
        data_tensor = data_tensor / self.global_mean  # normalize by dataset global mean

        data_tensor = data_tensor.unsqueeze(0)  # add channel dimension (1, N)
        # Now instead of using VCV, for testing, I will use only the C. 
        # data_tensor = data_tensor    # Just take the second element. 

        label_str = row['word']
        label = self.label_to_index[label_str]
        return data_tensor, data_tensor
    

class NPYDatasetInfoCollect(Dataset): 
    """
    Dataset class that returns data along with metadata information.
    For EVLUATION only. 
    """
    def __init__(self, csv_path, global_mean=1.0, consonant_select=None):
        """
        Args:
            csv_path (str): Path to the CSV metadata file.
            global_mean (float): Global mean for normalization.
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        # NOTE: new feture 20260203 to select only specific consonant data.
        if consonant_select is not None:
            df = df[df['consonant'].isin(consonant_select)].reset_index(drop=True)

        # Store DataFrame
        self.df = df
        self.global_mean = global_mean

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = np.load(row['path'])  # shape: (3, 17)

        # Convert numpy array to tensor
        data_tensor = torch.tensor(data, dtype=torch.float32)

        # Reshape as needed: flatten and add channel dimension
        data_tensor = data_tensor.reshape(-1)  # flatten to 1D NOTE: currently we borrow the VCV structure, anyway only C is changing. 
        data_tensor = data_tensor / self.global_mean  # normalize by dataset global mean

        data_tensor = data_tensor.unsqueeze(0)  # add channel dimension (1, N)
        # Now instead of using VCV, for testing, I will use only the C. 
        # data_tensor = data_tensor    # Just take the second element. 

        info = {
            "uid": row["uid"],
            "path": row["path"],
            "cog": row["cog"],
            "fri_dur": row["fri_dur"],
            "word": row["word"],
            "consonant": row["consonant"],
            "vowel": row["vowel"],
            # "train": row["train"],
        }

        return data_tensor, info