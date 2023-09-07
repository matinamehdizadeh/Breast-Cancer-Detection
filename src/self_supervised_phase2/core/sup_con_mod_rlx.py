from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class SupConModRLX(nn.Module):
    def __init__(self, gpu, temperature=0.07):
        super(SupConModRLX, self).__init__()
        self.temperature = temperature
        self.gpu = gpu
        self.base_temperature = temperature
        self.name = {}
        file1 = open('sample.txt', 'r')
        Lines = file1.readlines()
        file1.close()
        for i in range(len(Lines)):
            l = Lines[i].split(' ')
            if l[3] in self.name.keys():
                self.name[l[3]].append([l[6], float(l[9].strip())])
            else:
                self.name[l[3]] = [[l[6], float(l[9].strip())]]

    def forward(self, output_view1, output_view2, labels, weight, p1):
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
        features = torch.cat(
                [output_view1.unsqueeze(1),
                 output_view2.unsqueeze(1)], dim=1)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
          
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = features[:, 0]
        anchor_count = 1

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask2 = torch.eq(labels, labels.T).float().to(device)
        mask = mask2.clone().detach()
        
        for i in range(batch_size):
          arrayN =list(map(float, np.array(self.name[p1[i]])[:,1]))
          arrayP =np.array(self.name[p1[i]])[:,0]          
          for j in range(batch_size):
            if (p1[j] in arrayP):
              if (arrayN[list(arrayP).index(p1[j])] < 0.5):
                if labels[i] == 1:
                  mask2[i][j] = 0
            else:
              if labels[i] == 1:
                mask2[i][j] = 0

        mask2 = mask2.repeat(anchor_count, contrast_count)
        mask = mask.repeat(anchor_count, contrast_count)

        
          # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask2),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask2 = mask2 * logits_mask
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * (2*logits_mask - mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask2 * log_prob).sum(1) / mask2.sum(1)
        temp = [weight[int(x)] for x in labels]
        temp = torch.FloatTensor(temp).to(self.gpu)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size)[0]
        loss = torch.dot(temp,loss) / batch_size

        return loss