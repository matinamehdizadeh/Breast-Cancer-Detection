from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, gpu, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = 'one'
        self.gpu = gpu
        self.base_temperature = temperature

    def forward(self, features, labels, weight):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = self.gpu
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
          
        malignant = features[labels == 1]
        benign = features[labels == 0]
        
        malignant_contrast_features = torch.cat(torch.unbind(malignant, dim=1), dim=0)
        malignant_anchor = malignant[:, 0]

        benign_contrast_features = torch.cat(torch.unbind(benign, dim=1), dim=0)
        benign_anchor = benign[:, 0]

        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
       

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        loss_elimination = 0

        

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        temp = [weight[int(x)] for x in labels]
        temp = torch.FloatTensor(temp).to(self.gpu)
        #loss = torch.dot(temp,temp_log_prob) / b
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # print(loss.view(anchor_count, batch_size)[0])
        # print(batch_size)
        # print('---------------')
        loss = loss.view(anchor_count, batch_size)[0]
        loss = torch.dot(temp,loss)
        if ((len(labels[labels == 1]) != batch_size) and (len(labels[labels == 0]) != batch_size)):
          mask_elimination = torch.eye(batch_size, dtype=torch.float32).cuda(self.gpu)
          mask_elimination = mask_elimination.repeat(1, 2)
          logits_mask_elimination = torch.scatter(torch.ones_like(mask_elimination), 1, torch.arange(batch_size).view(-1, 1).cuda(self.gpu), 0)
          mask_elimination = mask_elimination * logits_mask_elimination

          dot_product_benign = torch.matmul(benign_anchor, malignant_contrast_features.T) / self.temperature
          dot_product_malignant = torch.matmul(malignant_anchor, benign_contrast_features.T) / self.temperature
          # Log-sum trick for numerical stability
          logits_max_benign, _ = torch.max(dot_product_benign, dim=1, keepdim=True)
          logits_benign = dot_product_benign - logits_max_benign.detach()
          exp_logits_benign = torch.exp(logits_benign)

          logits_max_malignant, _ = torch.max(dot_product_malignant, dim=1, keepdim=True)
          logits_malignant = dot_product_malignant - logits_max_malignant.detach()
          exp_logits_malignant = torch.exp(logits_malignant)
          
          
          
          first = (mask_elimination * logits).sum(1, keepdim=True)
          mal_counter = 0
          ben_counter = 0
          for i in range(batch_size):
            if labels[i] == 1:
              loss_elimination += (-1 * torch.log(torch.exp(first[i])/exp_logits_malignant.sum(1, keepdim=True)[mal_counter])) * weight[1]
              mal_counter += 1
            else:
              loss_elimination += (-1 * torch.log(torch.exp(first[i])/exp_logits_benign.sum(1, keepdim=True)[ben_counter])) * weight[0]
              ben_counter += 1
        else:
          loss_elimination = loss

        loss_elimination = loss_elimination / (batch_size*2)
        loss = loss / batch_size
        
        return loss + loss_elimination