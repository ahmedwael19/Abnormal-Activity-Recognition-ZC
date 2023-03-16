import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np


from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #  root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def inference(dataloader, model, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0, device=device)
        scores = []
        for i, input in enumerate(dataloader):

            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes, idx_abn = model(inputs=input)
            scores.append(score_abnormal.cpu().detach().numpy())
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))
        np.save('score_abn.npy',scores)
        idx_abn = idx_abn.cpu().detach().numpy()

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        np.save(ROOT/ 'pred.npy',pred)
        np.save(ROOT/ 'feat_magnitudes.npy',list(feat_magnitudes.cpu().detach().numpy())) ## Added

        return idx_abn, pred

