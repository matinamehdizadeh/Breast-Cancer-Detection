'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import numpy as np, json, argparse, time
from numpy.core.fromnumeric import size
from tqdm import tqdm
import cv2, logging
from pathlib import Path
import torch, torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, matthews_corrcoef, classification_report, confusion_matrix, accuracy_score
import sys
sys.path.append('~/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')
from supervised.utils.utils import *
from supervised.core.classification_models import classifier
from supervised.core.rmseLoss import RMSELoss
import bc_config



class Train_Util:

    def __init__(self, experiment_description, epochs, model, device, train_loader, val_loader, optimizer, criterion, batch_size, scheduler, num_classes, writer, threshold = 0.2):
        self.experiment_description = experiment_description
        self.epochs = epochs
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.scheduler = scheduler
        self.writer = writer
        self.num_classes = num_classes
        self.threshold = threshold
        file2 = open("file.txt", "r")
        weights = file2.read()
        self.weights = [int(weights.split()[1])/(int(weights.split()[1]) + int(weights.split()[0])), int(weights.split()[0])/(int(weights.split()[1]) + int(weights.split()[0]))]
        file2.close()
        

    def train_epoch(self, epoch, epochs):
        self.model.train()
        loss_agg = Aggregator()
        confusion_matrix_epoch = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
        kappa =[]
        tot_iterations = int(len(self.train_loader)/bc_config.batch_size)
        patient_confusion_matrix_val = {}
        criterion_domain = RMSELoss()
        i = 0
        lambda_val = 0.5
        with tqdm(total=(len(self.train_loader))) as (t):
            for patient_id, magnification, item_dict, binary_label, multi_label, h_e_matrices in tqdm(self.train_loader):
                view = item_dict[magnification[0]]
                view = view.cuda(self.device, non_blocking=True)
                h_e_matrices = h_e_matrices.type(torch.FloatTensor).to(self.device)
		
                p = float(i + epoch * tot_iterations) / epochs / tot_iterations
                i+=1
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                target = binary_label.to(self.device)
                outputs, pred_domain = self.model(view, alpha, True)
                outputs = outputs.squeeze(1)
                
                target = target.type_as(outputs)
                loss = self.criterion(outputs, target)
                patient_id = list(patient_id)
                loss_domains = lambda_val * criterion_domain(pred_domain.squeeze(1), h_e_matrices)
                loss = loss + loss_domains
                predicted = (outputs > self.threshold).int()
                predicted = predicted.to(self.device)

                for targetx, predictedx,idx in zip(target.view(-1), predicted.view(-1), patient_id):
             
                    confusion_matrix_epoch[(targetx.long(), predictedx.long())] += 1
                    if idx not in patient_confusion_matrix_val.keys():
                      patient_confusion_matrix_val[idx] = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
                    patient_confusion_matrix_val[idx][(targetx.long(), predictedx.long())] += 1
               
                

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_agg.update(loss.item())
                t.set_postfix(loss=('{:05.3f}'.format(loss_agg())))
                t.update()
        weighted_f1_patient = []
        accuracy_patient = []
        classwise_precision_patient= []
        classwise_recall_patient= []
        classwise_f1_patient = []
        bal_acc_patient = []
        dice_patient = []
        kappa_patient = []
        for patient in patient_confusion_matrix_val.keys():
          weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, bal_acc, dice, tpr, fpr, kappa = self.get_metrics_from_confusion_matrix(patient_confusion_matrix_val[patient])
          weighted_f1_patient.append(weighted_f1)
          accuracy_patient.append(accuracy)
          classwise_precision_patient.append(classwise_precision)
          classwise_recall_patient.append(classwise_recall)
          classwise_f1_patient.append(classwise_f1)
          bal_acc_patient.append(bal_acc)
          dice_patient.append(dice)
          kappa_patient.append(kappa)
        weighted_f1_patient = np.mean(weighted_f1_patient)
        accuracy_patient = np.mean(accuracy_patient)
        classwise_precision_patient= np.mean(classwise_precision_patient)
        classwise_recall_patient= np.mean(classwise_recall_patient)
        classwise_f1_patient = np.mean(classwise_f1_patient)
        bal_acc_patient = np.mean(bal_acc_patient)
        dice_patient = np.mean(dice_patient)
        kappa_patient = np.mean(kappa_patient)

        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, bal_acc, dice, tpr, fpr, kappa = self.get_metrics_from_confusion_matrix(confusion_matrix_epoch)
        print(f'{self.experiment_description}:classwise precision', classwise_precision)
        print(f'{self.experiment_description}: classwise recall', classwise_recall)
        print(f'{self.experiment_description}: classwise f1', classwise_f1)
        print(f'{self.experiment_description}: Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: kappa', np.mean(kappa))
        print(f'{self.experiment_description}: Accuracy', accuracy)
        print(f'{self.experiment_description}: Balanced Accuracy', bal_acc)
        print(f'{self.experiment_description}: Dice', dice)

        print(f'{self.experiment_description}: patient level classwise precision', classwise_precision_patient)
        print(f'{self.experiment_description}: patient level classwise recall', classwise_recall_patient)
        print(f'{self.experiment_description}: patient level classwise f1', classwise_f1_patient)
        print(f'{self.experiment_description}: patient level Weighted F1', weighted_f1_patient)
        print(f'{self.experiment_description}: patient level kappa', kappa_patient)
        print(f'{self.experiment_description}: patient level Accuracy', accuracy_patient)
        print(f'{self.experiment_description}: patient level Balanced Accuracy', bal_acc_patient)
        print(f'{self.experiment_description}: patient level Dice', dice_patient)
        return (weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, loss_agg(), kappa, bal_acc, dice, tpr, fpr)
    
    def evaluate_validation_set(self):
        confusion_matrix_val = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
        patient_confusion_matrix_val = {}
        self.model.eval()
        val_loss_avg = Aggregator()
        val_kappa = []
        with torch.no_grad():
            for patient_id, magnification, item_dict, binary_label, multi_label, _ in tqdm(self.val_loader):
                view = item_dict[magnification[0]]
                view = view.cuda(self.device, non_blocking=True)
                
                
                target = binary_label.to(self.device)

                outputs = self.model(view, 0, False)
                outputs = outputs.squeeze(1)
                target = target.type_as(outputs)
                loss = self.criterion(outputs, target)
                patient_id = list(patient_id)
                predicted = (outputs > self.threshold).int()
                predicted = predicted.to(self.device)

                

                for targetx, predictedx, idx in zip(target.view(-1), predicted.view(-1), patient_id):
                    confusion_matrix_val[(targetx.long(), predictedx.long())] += 1
                    if idx not in patient_confusion_matrix_val.keys():
                      patient_confusion_matrix_val[idx] = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
                    patient_confusion_matrix_val[idx][(targetx.long(), predictedx.long())] += 1

                else:
                    val_loss_avg.update(loss.item())
        weighted_f1_patient = []
        accuracy_patient = []
        classwise_precision_patient= []
        classwise_recall_patient= []
        classwise_f1_patient = []
        bal_acc_patient = []
        dice_patient = []
        kappa_patient = []
        for patient in patient_confusion_matrix_val.keys():
          weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, bal_acc, dice, tpr, fpr, kappa = self.get_metrics_from_confusion_matrix(patient_confusion_matrix_val[patient])
          weighted_f1_patient.append(weighted_f1)
          accuracy_patient.append(accuracy)
          classwise_precision_patient.append(classwise_precision)
          classwise_recall_patient.append(classwise_recall)
          classwise_f1_patient.append(classwise_f1)
          bal_acc_patient.append(bal_acc)
          dice_patient.append(dice)
          kappa_patient.append(kappa)
        weighted_f1_patient = np.mean(weighted_f1_patient)
        accuracy_patient = np.mean(accuracy_patient)
        classwise_precision_patient= np.mean(classwise_precision_patient)
        classwise_recall_patient= np.mean(classwise_recall_patient)
        classwise_f1_patient = np.mean(classwise_f1_patient)
        bal_acc_patient = np.mean(bal_acc_patient)
        dice_patient = np.mean(dice_patient)
        kappa_patient = np.mean(kappa_patient)
        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, bal_acc, dice, tpr, fpr, kappa = self.get_metrics_from_confusion_matrix(confusion_matrix_val)
        
        print(f'{self.experiment_description}: Validation classwise precision', classwise_precision)
        print(f'{self.experiment_description}: Validation classwise recall', classwise_recall)
        print(f'{self.experiment_description}: Validation classwise f1', classwise_f1)
        print(f'{self.experiment_description}: Validation Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Validation kappa', kappa)
        print(f'{self.experiment_description}: Validation Accuracy', accuracy)
        print(f'{self.experiment_description}: Validation Balanced Accuracy', bal_acc)
        print(f'{self.experiment_description}: Validation Dice', dice)

        print(f'{self.experiment_description}: Validation patient level classwise precision', classwise_precision_patient)
        print(f'{self.experiment_description}: Validation patient level classwise recall', classwise_recall_patient)
        print(f'{self.experiment_description}: Validation patient level classwise f1', classwise_f1_patient)
        print(f'{self.experiment_description}: Validation patient level Weighted F1', weighted_f1_patient)
        print(f'{self.experiment_description}: Validation patient level kappa', kappa_patient)
        print(f'{self.experiment_description}: Validation patient level Accuracy', accuracy_patient)
        print(f'{self.experiment_description}: Validation patient level Balanced Accuracy', bal_acc_patient)
        print(f'{self.experiment_description}: Validation patient level Dice', dice_patient)
        return (weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, val_loss_avg(), kappa, bal_acc, dice, tpr, fpr)

    
    def test_model(self):
        confusion_matrix_val = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
        self.model.eval()
        val_loss_avg = Aggregator()
        val_kappa = []
        with torch.no_grad():
            for _, magnification, item_dict, binary_label, multi_label in tqdm(self.val_loader):
                view = item_dict[magnification[0]]
                view = view.cuda(self.device, non_blocking=True)
                
                
                target = binary_label.to(self.device)

                outputs = self.model(view)
                outputs = outputs.squeeze(1)
                target = target.type_as(outputs)

                
                predicted = (outputs > self.threshold).int()

                predicted = predicted.to(self.device)

                for targetx, predictedx in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix_val[(targetx.long(), predictedx.long())] += 1
                
        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1 = self.get_metrics_from_confusion_matrix(confusion_matrix_val)
        
        
        print(f'{self.experiment_description}: Test classwise precision', classwise_precision)
        print(f'{self.experiment_description}: Test classwise recall', classwise_recall)
        print(f'{self.experiment_description}: Test classwise f1', classwise_f1)
        print(f'{self.experiment_description}: Test Weighted F1', weighted_f1)
        print(f'{self.experiment_description}: Test Accuracy', accuracy)
        return (weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1)
    
    
    def test_class_probabilities(self, model, device, test_loader, which_class):
        model.Test()
        actuals = []
        probabilities = []
        with torch.no_grad():
            for Testage, label in test_loader:
                image, label = image.to(device), label.to(device)
                output = torch.sigmoid(model(image))
                prediction = output.argmax(dim=1, keepdim=True)
                actuals.extend(label.view_as(prediction) == which_class)
                output = output.cpu()
                probabilities.extend(np.exp(output[:, which_class]))

        return (
         [i.item() for i in actuals], [i.item() for i in probabilities])

    def train_and_evaluate(self):
        
        best_f1 = 0.0
        for epoch in range(self.epochs):
            #train epoch
            weighted_f1, accuracy,classwise_precision,classwise_recall,classwise_f1, loss, kappa, bal_acc, dice, tpr, fpr = self.train_epoch(epoch, self.epochs)
            #evaluate on validation set
            val_weighted_f1, val_accuracy, val_classwise_precision,val_classwise_recall,val_classwise_f1, val_loss, val_kappa, val_bal_acc, val_dice, val_tpr, val_fpr = self.evaluate_validation_set()
                        
            print("Epoch {}/{} Train Loss:{}, Val Loss: {}".format(epoch, self.epochs, loss, val_loss))

            if best_f1 < val_weighted_f1:
                best_train = weighted_f1
                best_f1 = val_weighted_f1
                result_path = f"{bc_config.result_path}{self.experiment_description}"
                Path(result_path).mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), f"{result_path}/best.pth")
                self.scheduler.step(val_loss)

            elif (best_train < weighted_f1) and (best_f1 <= val_weighted_f1):
                best_train = weighted_f1
                best_f1 = val_weighted_f1
                result_path = f"{bc_config.result_path}{self.experiment_description}"
                Path(result_path).mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), f"{result_path}/best.pth")


                file_object = open('sample.txt', 'a')
                file_object.write('\n')
                file_object.write(str(best_f1))
                file_object.close()
                self.scheduler.step(val_loss)

            #Tensorboard
            self.writer.add_scalar('Loss/Validation_Set', val_loss, epoch)
            self.writer.add_scalar('Loss/Training_Set', loss, epoch)

            self.writer.add_scalar('Accuracy/Validation_Set', val_accuracy, epoch)
            self.writer.add_scalar('Accuracy/Training_Set', accuracy, epoch)
            
            self.writer.add_scalar('Dice/Validation_Set', val_dice, epoch)
            self.writer.add_scalar('Dice/Training_Set', dice, epoch)

            self.writer.add_scalar('Balance Acc/Validation_Set', val_bal_acc, epoch)
            self.writer.add_scalar('Balance Acc/Training_Set', bal_acc, epoch)

            self.writer.add_scalar('Kappa/Validation_Set', val_kappa, epoch)
            self.writer.add_scalar('Kappa/Training_Set', kappa, epoch)

            self.writer.add_scalar('Weighted F1/Validation_Set', val_weighted_f1, epoch)
            self.writer.add_scalar('Weighted F1/Training_Set', weighted_f1, epoch)

            self.writer.add_scalar('ROC/Validation_Set', val_tpr, val_fpr)
            self.writer.add_scalar('ROC/Training_Set', tpr, fpr)
            
            
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)

            #Classwise metrics logging
            if 2 == self.num_classes:
                for index in range(0,len(bc_config.binary_label_list)):
                    
                    self.writer.add_scalar(f'F1/Validation_Set/{bc_config.binary_label_list[index]}', val_classwise_f1[index], epoch)
                    self.writer.add_scalar(f'F1/Training_Set/{bc_config.binary_label_list[index]}', classwise_f1[index], epoch)

                    self.writer.add_scalar(f'Precision/Validation_Set/{bc_config.binary_label_list[index]}', val_classwise_precision[index], epoch)
                    self.writer.add_scalar(f'Precision/Training_Set/{bc_config.binary_label_list[index]}', classwise_precision[index], epoch)

                    self.writer.add_scalar(f'Recall/Validation_Set/{bc_config.binary_label_list[index]}', val_classwise_recall[index], epoch)
                    self.writer.add_scalar(f'Recall/Training_Set/{bc_config.binary_label_list[index]}', classwise_recall[index], epoch)
                
        return val_weighted_f1
    def process_classification_report(self, report):
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            row = {}
            row_data = line.split(' ')
            row_data = list(filter(None, row_data))
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            report_data.append(row)
        else:
            return report_data

    def get_metrics_from_confusion_matrix(self, confusion_matrix_epoch):
        epoch_classwise_precision_manual_cpu = np.array(confusion_matrix_epoch.diag().cpu()) / np.array(confusion_matrix_epoch.cpu()).sum(axis=0)
        epoch_classwise_precision_manual_cpu = np.nan_to_num(epoch_classwise_precision_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_classwise_recall_manual_cpu = np.array(confusion_matrix_epoch.diag().cpu()) / np.array(confusion_matrix_epoch.cpu()).sum(axis=1)
        epoch_classwise_recall_manual_cpu = np.nan_to_num(epoch_classwise_recall_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_classwise_f1_manual_cpu = 2 * (epoch_classwise_precision_manual_cpu * epoch_classwise_recall_manual_cpu) / (epoch_classwise_precision_manual_cpu + epoch_classwise_recall_manual_cpu)
        epoch_classwise_f1_manual_cpu = np.nan_to_num(epoch_classwise_f1_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_avg_f1_manual = np.sum(epoch_classwise_f1_manual_cpu * np.array(confusion_matrix_epoch.cpu()).sum(axis=1)) / np.array(confusion_matrix_epoch.cpu()).sum(axis=1).sum()
        epoch_acc_manual = 100 * np.sum(np.array(confusion_matrix_epoch.diag().cpu())) / np.sum(np.array(confusion_matrix_epoch.cpu()))
        epoch_dice_manual_cpu = 2*confusion_matrix_epoch[1][1] / (2*confusion_matrix_epoch[1][1] + confusion_matrix_epoch[0][1] + confusion_matrix_epoch[1][0])
        epoch_dice_manual_cpu = np.nan_to_num(epoch_dice_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_FPR_manual_cpu = confusion_matrix_epoch[0][1] / (confusion_matrix_epoch[0][1] + confusion_matrix_epoch[0][0])
        epoch_FPR_manual_cpu = np.nan_to_num(epoch_FPR_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_TNR_manual_cpu = confusion_matrix_epoch[0][0] / (confusion_matrix_epoch[0][0] + confusion_matrix_epoch[0][1])
        epoch_TNR_manual_cpu = np.nan_to_num(epoch_TNR_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_TPR_manual_cpu = confusion_matrix_epoch[1][1] / (confusion_matrix_epoch[1][1] + confusion_matrix_epoch[1][0])
        epoch_TPR_manual_cpu = np.nan_to_num(epoch_TPR_manual_cpu, nan=0, neginf=0, posinf=0)
        epoch_bal_acc_manual_cpu = (epoch_TPR_manual_cpu + epoch_TNR_manual_cpu)/2
        epoch_kappa_manual_cpu = 2*(confusion_matrix_epoch[1][1] * confusion_matrix_epoch[0][0] - confusion_matrix_epoch[1][0] * confusion_matrix_epoch[0][1])/((confusion_matrix_epoch[1][1] + confusion_matrix_epoch[0][1])*(confusion_matrix_epoch[0][1] + confusion_matrix_epoch[0][0])+(confusion_matrix_epoch[1][1] + confusion_matrix_epoch[1][0])*(confusion_matrix_epoch[1][0]+confusion_matrix_epoch[0][0]))
        epoch_kappa_manual_cpu = np.nan_to_num(epoch_kappa_manual_cpu, nan=0, neginf=0, posinf=0)

        return (
         epoch_avg_f1_manual, epoch_acc_manual, epoch_classwise_precision_manual_cpu, epoch_classwise_recall_manual_cpu, epoch_classwise_f1_manual_cpu,
          epoch_bal_acc_manual_cpu, epoch_dice_manual_cpu, epoch_TPR_manual_cpu, epoch_FPR_manual_cpu, epoch_kappa_manual_cpu)
