import os,sys
import logging
from os import get_exec_path
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from tqdm import tqdm
import numpy as np
import torch
import sys
import tensorflow
sys.path.append('/content/drive/MyDrive/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')

from self_supervised_phase2.utils import config
sys.path.append(os.path.dirname(__file__))
from self_supervised_phase2.utils import config
from self_supervised_phase2.core import sup_con_mod_rlx
from torch.utils.tensorboard import SummaryWriter

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

#Trainer - Magnification Prior Contrastive Similarity Method
class Trainer:
    def __init__(self,
                 experiment_description,
                 dataloader,
                 model,
                 optimizer,
                 scheduler,
                 epochs,
                 batch_size,
                 gpu,
                 criterion):
        self.experiment_description = experiment_description
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.previous_model = model
        self.current_model = model
        self.criterion = sup_con_mod_rlx.SupConModRLX(gpu=gpu, temperature=0.1)
        self.scheduler = scheduler
        self.gpu = gpu

        self.epochs = epochs
        self.current_epoch = 11
        self.lowest_loss = 10000
        self.cmd_logging = logging
        self.cmd_logging.basicConfig(level=logging.INFO,
                                     format='%(levelname)s: %(message)s')
        self.batch_size = batch_size
        self.loss_list = []
        self.loss_list.append(0)
        self.writer = SummaryWriter(log_dir=config.tensorboard_path+experiment_description)
        self.input_images = []
        file2 = open("file.txt", "r")
        weights = file2.read()
        self.weight = {0:int(weights.split()[1]) / (int(weights.split()[1]) + int(weights.split()[0])) ,
                        1:int(weights.split()[0]) / (int(weights.split()[1]) + int(weights.split()[0]))}
        file2.close()

    def train(self):
        for epoch in range(1, self.epochs + 1):

            self.current_epoch = epoch
            self.previous_model = self.current_model

            epoch_response_dir = self.train_epoch(
                gpu = self.gpu,
                current_epoch=self.current_epoch,
                epochs=self.epochs,
                batch_size=self.batch_size,
                train_loader=self.dataloader,
                model=self.current_model,
                optimizer=self.optimizer,
                criterion=self.criterion,
                weight=self.weight)

            self.current_model = epoch_response_dir['model']
            self.loss_list.append(epoch_response_dir['loss'])
            if epoch % 10 == 0:
              logging.info(
                  f'{self.experiment_description} epoch: {epoch} simCLR loss: {self.loss_list[self.current_epoch]}'
              )
              print(epoch)
            self.input_images = epoch_response_dir['image_pair']

            #Logging tensor board
            self.tensorboard_analytics()

            #Save model - conditional
            self.save_model()

    def tensorboard_analytics(self):

        self.writer.add_scalar('SimCLR-Contrastive-Loss/Epoch',
                               self.loss_list[self.current_epoch],
                               self.current_epoch)

        self.writer.add_scalar('Learning_Rate/Epoch',
                               self.optimizer.param_groups[0]['lr'],
                               self.current_epoch)

        self.writer.add_image('View1/Aug',
                              self.input_images[0].detach().cpu().numpy()[0],
                              self.current_epoch)
        self.writer.add_image('View2/Aug',
                              self.input_images[1].detach().cpu().numpy()[0],
                              self.current_epoch)

    def save_model(self):
        if self.loss_list[self.current_epoch] < self.lowest_loss:
            os.makedirs(f"{config.result_path+self.experiment_description}",
                        exist_ok=True)
            torch.save(
                self.current_model.state_dict(),
                f"{config.result_path+self.experiment_description}/best.pth"
            )
            self.lowest_loss = self.loss_list[self.current_epoch]

    def train_epoch(self, gpu, current_epoch, epochs, batch_size, train_loader,
                            model, optimizer, criterion, weight):

        model.train()
        total_loss = 0
        epoch_response_dir = {}
        with tqdm(total=batch_size * len(train_loader),
                desc=f'Epoch {current_epoch}/{epochs}',
                unit='img') as (pbar):
            for idx, batch in enumerate(train_loader):
                view1, label, view2, p1 = batch[0], batch[1], batch[2], batch[3]
                b, c, h, w = view1.size()
                #for pytorch tranform
                view1 = view1.cuda(gpu, non_blocking=True)
                view2 = view2.cuda(gpu, non_blocking=True)
                output_view1 = model(view1)
                output_view2 = model(view2)
                label = np.array(label)

                label[label == 'M'] = 1
                label[label == 'B'] = 0

                label = torch.tensor(label.astype(float))
                loss = criterion(output_view1, output_view2, label, weight, p1)
                
                curr_loss = loss.item()

                total_loss += curr_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                '''logging'''
                #logging.info('minibatch: {idx} simCLR running_loss: {loss.item()}')
                (pbar.set_postfix)(**{'loss (batch)': loss.item()})
                pbar.update(view1.shape[0])


            # Prepare epoch reponse and return
            epoch_response_dir['model'] = model
            epoch_response_dir['loss'] = total_loss/(batch_size*len(train_loader))
            epoch_response_dir['image_pair'] = [view1, view2]

        return epoch_response_dir