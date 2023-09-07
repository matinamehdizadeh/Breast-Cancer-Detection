from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class ConLoss(nn.Module):
    def __init__(self, gpu, temperature=0.07):
        super(ConLoss, self).__init__()
        self.gpu = gpu

    def forward(self, output_view1, output_view2, labels, labels2, weight, p1, p2, mag):
        device = self.gpu

        file_object = open('sample.txt', 'a')

        for i in range(len(output_view1)):
          if labels[i] == labels2[i]:
            temp = torch.dot(output_view1[i], output_view2[i])
            file_object.write('mag ' + mag[i] + ' P1 ' + str(p1[i]) + ' '+ str(labels[i]) +' P2 ' + str(p2[i]) +  ' '+ str(labels2[i]) + ' sim ' + str(temp.item()) + '\n')
        file_object.close()

        return output_view1[0][0]