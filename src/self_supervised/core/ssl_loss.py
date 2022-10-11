'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
sys.path.append('/content/drive/MyDrive/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')

from self_supervised.apply import config
sys.path.append(os.path.dirname(__file__))

class SimCLR_loss(nn.Module):
    
    # Based on the implementation of SupContrast
    def __init__(self, gpu, temperature):
        super(SimCLR_loss, self).__init__()
        self.gpu = gpu
        self.temperature = temperature
   
    def forward(self, features, label, weight):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda(self.gpu)

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]
   
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(self.gpu), 0)
        mask = mask * logits_mask 
        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        temp_log_prob = - ((mask * log_prob).sum(1) / mask.sum(1))
        temp = [weight[int(x)] for x in label]
        temp = torch.FloatTensor(temp).to(self.gpu)
        loss = torch.dot(temp,temp_log_prob) / b
       
        # Mean log-likelihood for positive
        # loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss

