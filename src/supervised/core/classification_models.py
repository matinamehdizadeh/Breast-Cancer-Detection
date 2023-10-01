from numpy.core.numeric import outer
import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch.autograd import Function
import sys
import numpy as np
sys.path.append('~/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')
from supervised.core.models import EfficientNet_Model
from self_supervised_phase1.core.models import EfficientNet_MLP
from supervised.utils.transform import train_transform, resize_transform2


class domain_predictor(torch.nn.Module):

  def __init__(self, n_centers, l1, l2, l3):
    super(domain_predictor, self).__init__()
    self.n_centers = n_centers
    self.rfc1 = nn.Linear(1408, l1)
    self.rfc2 = nn.Linear(l1, l2)
    self.rfc3 = nn.Linear(l2, l3)	
    self.domain_classifier = torch.nn.Linear(l3, self.n_centers)
  
  def forward(self, x):

    dropout = torch.nn.Dropout(p=0.3)
    m_binary = torch.nn.Sigmoid()

    domain_emb = self.rfc1(x)
    domain_emb = F.relu(domain_emb)
    domain_emb = dropout(domain_emb)
    domain_emb = self.rfc2(domain_emb)
    domain_emb = F.relu(domain_emb)
    domain_emb = dropout(domain_emb)
    domain_emb = self.rfc3(domain_emb)
    domain_emb = F.relu(domain_emb)    


    domain_prob = self.domain_classifier(domain_emb)
    domain_prob = m_binary(domain_prob)

    return domain_prob

class ReverseLayerF(Function):

	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha

		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha

		return output, None

class classifier(torch.nn.Module):
    def __init__(self, l1, l2, l3, prob, path='', device='cuda:0'):
        #super(EfficientNet_Model, self).__init__()
        super(classifier, self).__init__()
        self.prob = prob
        self.efficient = EfficientNet_Model(pretrained=False)
        # 2. Initialized and load SSL pretraiend model which include backbone, MLP, and head
        pretrained_model = EfficientNet_MLP()
       # comment next line for test
        pretrained_model.load_state_dict(torch.load(path, map_location=device))
        # 3. Use backbone part of pretrained model
        self.efficient.model = pretrained_model.backbone
        for param in self.efficient.parameters():
          param.requires_grad = False
        l1 = np.power(2, l1)
        l2 = np.power(2, l2)
        l3 = np.power(2, l3)
        self.rfc1 = nn.Linear(1408, l1)
        self.rfc2 = nn.Linear(l1, l2)
        self.rfc3 = nn.Linear(l2, l3)        
        self.final_fc2 = nn.Sequential(nn.Dropout(self.prob), nn.Linear(l3, 1))
        self.sig = nn.Sigmoid()
        self.domain_predictor = domain_predictor(6, l1, l2, l3)


    def forward(self, x, alpha, trainBool = True):
        out = self.efficient(x)
        out = F.relu(out)
        if trainBool:
          reverse_feature = ReverseLayerF.apply(out, alpha)

          output_domain = self.domain_predictor(reverse_feature)
          out = self.rfc1(out)
          out = F.relu(out) 
          out = F.dropout(out, p=self.prob)
          out = self.rfc2(out)
          out = F.relu(out) 
          out = F.dropout(out, p=self.prob)
          out = self.rfc3(out)
          out = F.relu(out) 
          output = self.sig(self.final_fc2(out))  
          return output, output_domain
        out = self.rfc1(out)
        out = F.relu(out) 
        out = F.dropout(out, p=self.prob)
        out = self.rfc2(out)
        out = F.relu(out) 
        out = F.dropout(out, p=self.prob)
        out = self.rfc3(out)
        out = F.relu(out) 
        output = self.sig(self.final_fc2(out))  
        
        return output