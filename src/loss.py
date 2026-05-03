from unicodedata import name

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, dataloader, ewc_lambda, estimate_num_batches=None, device=None):
        self.device = device
        self.old_params = None
        self.fim = None
        self.ewc_lambda = ewc_lambda
        self.dataloader = dataloader
        if estimate_num_batches is None: 
            self.estimate_num_batches = len(dataloader)
        elif estimate_num_batches > len(dataloader): 
            # this ensures that we do not accidentally use more samples than we have, which would lead to wrongly calculated FIM. 
            self.estimate_num_batches = len(dataloader)
        else: 
            self.estimate_num_batches = estimate_num_batches

    def use_for_ewc(self, name, p): 
        # NOTE: this time we only regularize the encoder part, not the whole model. 
        return p.requires_grad and ("encoder" in name)

    def penalty(self, new_model): 
        if self.fim is None or self.old_params is None:
            raise ValueError("FIM and old parameters must be calculated before calling penalty.")
        
        new_params = new_model.named_parameters()
        loss = torch.tensor(0.0).to(self.device)
        for name, p in new_params: 
            if (name not in self.fim) or (name not in self.old_params): 
                continue
            if p.shape != self.old_params[name].shape: 
                print(f"Warning: shape mismatch for parameter {name}. Skipping EWC penalty for this parameter. ")
                continue
            
            loss += (((self.old_params[name] - p) ** 2) * self.fim[name]).flatten().sum()
        return loss * self.ewc_lambda
    
    def calculate_fim(self, model, criterion): 
        """
        Calculate the Fisher Information Matrix (FIM) for the given model. 
        model: please use deepcopy to ensure that the model is not changed by this function. 
        """
        print("Calculating FIM")
        model.eval()
        fim = {
            name: torch.zeros_like(p, device=self.device)
            for name, p in model.named_parameters()
            if (p.requires_grad and self.use_for_ewc(name, p))
        }

        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(self.dataloader): 
            if batch_idx >= self.estimate_num_batches: 
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            model.zero_grad()

            features = model(inputs)

            # The negative log likelihood
            loss = criterion(features, targets)

            loss.backward()

            for name, p in model.named_parameters(): 
                if (p.requires_grad and p.grad is not None and self.use_for_ewc(name, p)): 
                    fim[name] += p.grad.detach() ** 2
            
            num_batches += 1

        if num_batches > 0: 
            for p in fim:
                fim[p] = fim[p] / num_batches
        
        theta_star = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if (p.requires_grad and self.use_for_ewc(name, p))
        }

        # Debugging: print out the FIM values to check if they are reasonable.
        print("Number of FIM params:", len(fim))
        for name, val in fim.items():
            print(name, val.abs().mean().item(), val.abs().max().item())
            break

        self.fim = fim
        self.old_params = theta_star
        return 
    

class SupConLoss(nn.Module):
    """
    Supervised contrastive loss for one view per sample.

    features: [batch, hidden_dim]
    labels:   [batch]
    """
    def __init__(self, temperature=0.07, eps=1e-12):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, features, labels):
        device = features.device

        labels = labels.contiguous().view(-1, 1)

        # Normalize hidden vectors so the loss uses cosine similarity
        features = F.normalize(features, dim=1)

        batch_size = features.shape[0]

        # similarity matrix: [B, B]
        sim = torch.matmul(features, features.T) / self.temperature

        # remove self-comparisons
        logits_mask = torch.ones_like(sim, device=device)
        logits_mask.fill_diagonal_(0)

        # positive mask: same label, excluding self
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask = pos_mask * logits_mask

        # numerical stability
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + self.eps)

        # number of positives for each anchor
        pos_count = pos_mask.sum(dim=1)

        # skip anchors with no positive example in the batch
        valid = pos_count > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + self.eps)

        loss = -mean_log_prob_pos[valid].mean()

        return loss