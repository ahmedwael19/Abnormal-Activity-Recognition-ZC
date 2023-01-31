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

def test(dataloader, model, args, viz, device):
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
        if args.dataset == 'shanghai':
            gt = np.load(ROOT/ 'list/gt-sh.npy')
        
        elif args.dataset == "ZC":
            gt = np.load(ROOT/ 'gt_aba2.npy')
            
        else:
            gt = np.load(ROOT/ 'list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)
        np.save(ROOT/ 'pred.npy',pred)
        np.save(ROOT/ 'gt.npy',list(gt))
        np.save(ROOT/ 'feat_magnitudes.npy',list(feat_magnitudes.cpu().detach().numpy())) ## Added
        fpr, tpr, threshold = roc_curve(list(gt), pred)
        np.save(ROOT/ 'fpr.npy', fpr)
        np.save(ROOT/ 'tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save(ROOT/'precision.npy', precision)
        np.save(ROOT/'recall.npy', recall)
        viz.plot_lines('pr_auc', pr_auc)
        viz.plot_lines('auc', rec_auc)
        viz.lines('scores', pred)
        viz.lines('roc', tpr, fpr)
        return rec_auc, idx_abn

