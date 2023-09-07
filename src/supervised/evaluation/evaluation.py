'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import numpy as np
import json
import argparse
import time
from tqdm import tqdm
import cv2
import logging
import sys, os

import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

from sklearn.metrics import f1_score,matthews_corrcoef
import sys
sys.path.append('~/matinaMehdizadeh/Magnification-Prior-Self-Supervised-Method-main/src/')

from supervised.core.models import EfficientNet_Model
from self_supervised.core.models import EfficientNet_MLP


from supervised.apply.datasets import get_BreakHis_data_loader, get_BreakHis_testdata_loader
#for bach test uncomment the next line
#from supervised.bach.dataset import get_BreakHis_data_loader, get_BreakHis_testdata_loader
from supervised.apply.transform import resize_transform
from supervised.core.train_util import Train_Util
from supervised.core.classification_models import classifier

import bc_config


def get_metrics_from_confusion_matrix(confusion_matrix_epoch):

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


def test(model, test_loader, device, threshold, magnification):
        confusion_matrix_val = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
        patient_confusion_matrix_val = {}
        model.eval()
        writer = SummaryWriter()
        kappa = []
        with torch.no_grad(): 
            for patient_id, _, item_dict, binary_label, _, _ in tqdm(test_loader):
                view = item_dict[magnification]
                view = view.cuda(device, non_blocking=True)


                target = binary_label.to(device)

                outputs = model(view, 0, False)
                outputs = outputs.squeeze(1)
                target = target.type_as(outputs)

                predicted = (outputs > threshold).int()
                predicted = predicted.to(device)

                for targetx, predictedx,idx in zip(target.view(-1), predicted.view(-1), patient_id):
                    confusion_matrix_val[(targetx.long(), predictedx.long())] += 1
                    if idx not in patient_confusion_matrix_val.keys():
                      patient_confusion_matrix_val[idx] = torch.zeros(len(bc_config.binary_label_list), len(bc_config.binary_label_list))
                    patient_confusion_matrix_val[idx][(targetx.long(), predictedx.long())] += 1
               
        weighted_f1_patient = []
        accuracy_patient = []
        classwise_precision_patient= []
        classwise_recall_patient= []
        classwise_f1_patient = []
        bal_acc_patient = []
        dice_patient = []
        kappa_patient = []
        for patient in patient_confusion_matrix_val.keys():
          weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, bal_acc, dice, tpr, fpr, kappa = get_metrics_from_confusion_matrix(patient_confusion_matrix_val[patient])
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



        weighted_f1, accuracy, classwise_precision, classwise_recall, classwise_f1, bal_acc, dice, tpr, fpr, kappa = get_metrics_from_confusion_matrix(confusion_matrix_val)


        print('pretrained fine-tuned on validation set: ')
        print('Testset classwise precision', classwise_precision)
        print('Testset classwise recall', classwise_recall)
        print('Testset classwise f1', classwise_f1)


        print('Testset Weighted F1',weighted_f1)
        print('Testset Accuracy', accuracy)
        print('Testset kappa', np.mean(kappa))
        print('Testset Balanced Accuracy', bal_acc)
        print('Testset Dice', dice)

        print('patient level classwise precision', classwise_precision_patient)
        print('patient level classwise recall', classwise_recall_patient)
        print('patient level classwise f1', classwise_f1_patient)
        print('patient level Weighted F1', weighted_f1_patient)
        print('patient level kappa', kappa_patient)
        print('patient level Accuracy', accuracy_patient)
        print('patient level Balanced Accuracy', bal_acc_patient)
        print('patient level Dice', dice_patient)
        return weighted_f1, accuracy,classwise_precision,classwise_recall,classwise_f1

def test_model(data_path, magnification, model_path):

    threshold = 0.5
    device = "cuda:9"
    model_path = model_path
    magnification = magnification
    data_path = data_path



    test_loader = get_BreakHis_testdata_loader(data_path, transform = resize_transform,pre_processing=[], image_type_list= [magnification])

    model = classifier(10,13,11,0, model_path, device)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)


    test(model=model, test_loader=test_loader, device=device, threshold=threshold, magnification = magnification)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test_data_fold', type=str, required=True, help='The path for fold of testset'
    )
    parser.add_argument(
        '--magnification', type=str, required=True, help='Choose magnification for model fine-tuning, Example - 40x'
    )
    parser.add_argument(
        '--model_path', type=str, required=True, help='The path for pretrained model'
    )
    args = parser.parse_args()
    test_model(args.test_data_fold, args.magnification, args.model_path)
