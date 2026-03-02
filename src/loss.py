import torch
import torch.nn as nn
from copy import deepcopy
from typing import List, Sequence, Tuple, Union, Optional

class SimpleMSE(nn.Module): 
    """
    This does not include any consolidation. 
    """
    def __init__(self, reduction="none"): 
        super().__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
    def forward(self, y_pred, y_true): 
        loss = self.criterion(y_pred, y_true)
        return loss

class EWC: 
    def __init__(self, dataloader, ewc_lambda, estimate_sample_size=None, device=None):
        self.device = device
        self.old_params = None
        self.fim = None
        self.ewc_lambda = ewc_lambda
        self.dataloader = dataloader
        if estimate_sample_size is None: 
            self.estimate_sample_size = len(dataloader.dataset)
        else: 
            self.estimate_sample_size = estimate_sample_size

    def penalty(self, new_params):
        loss = torch.tensor(0.0).to(self.device)
        for idx, weight in enumerate(self.old_params):
            loss += (((self.old_params[weight] - new_params[idx]) ** 2) * self.fim[weight]).flatten().sum(dim=-1)
        return loss * self.ewc_lambda
    
    def calculate_fim(self, model, criterion, optimizer): 
        print("Calculating FIM")
        model_to_use = deepcopy(model.state_dict()) # ensures that we don't touch the real model
        fim = {}
        for batch_idx, (inputs, targets) in enumerate(self.dataloader): 
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            features = model_to_use(inputs)

            # The negative log likelihood
            loss = criterion(features, targets)

            optimizer.zero_grad()
            loss.backward()

            layer_names = list(model.state_dict().keys())
            for p_idx, p in enumerate(model.parameters()):
                if p.grad != None:
                    if layer_names[p_idx] not in fim:
                        fim[layer_names[p_idx]] = torch.zeros_like(p.grad).to(self.device)
                        fim[layer_names[p_idx]] += p.grad ** 2
            
            if batch_idx >= self.estimate_sample_size - 1: 
                break

        for p in fim:
            fim[p] = fim[p] / self.estimate_sample_size
        
        return fim, model.state_dict()
