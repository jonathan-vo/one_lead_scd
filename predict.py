import glob
import shutil
from collections import namedtuple
from pathlib import Path
from math import sqrt

import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from data import ECGDataset
from models import EffNet
from training_models import BinaryClassificationModel
import matplotlib.pyplot as plt
import sklearn

code_path = Path('/Users/jonathanvo/Documents/ECG Single Lead ML/Code')
model_path = code_path / 'lightning_logs/version_61/checkpoints/epoch=45-step=276.ckpt'


project = code_path / "Dataset"
targets_file = project / "Manifest" / "DigitizedECGs_ExternalReplication_5_10_2022_removed.csv"
# targets_file = project / "Manifest" / "DigitizedECGs_4_13_2022_removed.csv"
targets = pd.read_csv(targets_file)['case']

data_path = project / "test_output"
# data_path = project / "output"

def roc_auc_ci(y_true, y_score, positive=1):
    #Calculating AUC 95% CI
    AUC = sklearn.metrics.roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper)

if __name__ == '__main__':
    test_ds = ECGDataset(
            data_path=data_path,
            manifest_path=targets_file,
            labels="case",
            first_lead_only=False
        )
    
    test_dl = DataLoader(
            test_ds, num_workers=0, batch_size=500, drop_last=False, shuffle=False
        )
    trainer = Trainer()
    classifier = BinaryClassificationModel()
    preds = trainer.predict(model=classifier, ckpt_path=model_path,dataloaders=test_dl)
    preds_df = pd.DataFrame(preds)
    ecg_preds = []
    
    for i in preds_df.iterrows():
        ecg_preds.append(i[1][1])
        # ecg_name = preds_df[0][0] + preds_df[0][1] + preds_df[0][2]
    ecg_preds = np.concatenate(ecg_preds)
    
    print(f"AUC = {sklearn.metrics.roc_auc_score(targets,ecg_preds)}")
    print(f"95% CI = {roc_auc_ci(targets,ecg_preds)}")
    fpr, tpr, threshold = sklearn.metrics.roc_curve(targets,ecg_preds)
    plt.plot(fpr,tpr, label= f"AUC = {sklearn.metrics.roc_auc_score(targets,ecg_preds)}")
    # plt.plot(fpr,tpr, label= "External Set: AUC = 0.655 (95% CI: 0.628, 0.683)")
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.legend()