from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
from inference import inference
from tqdm import tqdm
from utils import Visualizer
from config import *
import argparse
from datetime import datetime, timezone
import matplotlib.pyplot as plt

from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  #  root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



import psycopg2
from datetime import datetime, timezone

print('Connecting to the PostgreSQL database...')
conn = psycopg2.connect(
    host="localhost",
    database="action_activity",
    user="postgres",
    password=" ")

cur = conn.cursor()

viz = Visualizer(env='shanghai tech 10 crop', use_incoming_socket=False)


def parse_opts():
    parser = argparse.ArgumentParser(description='RTFM')
    parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
    parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
    parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
    parser.add_argument('--rgb-list', default=ROOT/'list/ZC-i3d-test-10crop.list', help='list of rgb features ')
    parser.add_argument('--test-rgb-list', default=ROOT/'list/ZC-i3d-test-10crop.list', help='list of test rgb features ')
    parser.add_argument('--gt', default=ROOT/'gt_aba2.npy', help='file of ground truth ') ## Changed
    parser.add_argument('--gpus', default=1, type=int, choices=[0], help='gpus')
    parser.add_argument('--lr', type=str, default='[0.001]*15000', help='learning rates for steps(list form)')
    parser.add_argument('--batch-size', type=int, default=8, help='number of instances in a batch of data (default: 16)')
    parser.add_argument('--workers', default=4, help='number of workers in dataloader')
    parser.add_argument('--model-name', default='rtfm', help='name to save model')
    parser.add_argument('--pretrained', default=ROOT/'ckpt/rtfmfinal.pkl', help='ckpt for pretrained model')
    parser.add_argument('--num-classes', type=int, default=1, help='number of class')
    parser.add_argument('--dataset', default='ZC', help='dataset to train on (default: )')
    parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
    parser.add_argument('--max-epoch', type=int, default=15000, help='maximum iteration to train (default: 100)')
    parser.add_argument('--features', default=ROOT/'a.npy', help='I3D Features path')
    parser.add_argument('--inference', default = True, action='store_true', help='run in inference mode')
    parser.add_argument('--training',default=False, action='store_true', help='run in inference mode')
    parser.add_argument('--testing',default=False, action='store_true', help='run in inference mode')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
  #  args = option.parser.parse_args()

    args = parse_opts()
    rgb_list = ( args.rgb_list )
    config = Config(args)
    with open( rgb_list , 'w') as f:
        f.write(args.features)
    if args.training:
        train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                                   batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
        train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    model = Model(args.feature_size, args.batch_size)
    model.load_state_dict(torch.load(args.pretrained)) ## Load the pretrained model :)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')
    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
  ##  output_path = 'ZC/'   # put your own path here
    if args.testing:
        auc, idx_abn, pred = test(test_loader, model, args,viz, device)

    if args.inference:
        idx_abn, pred = inference(test_loader, model, device)
      #  pred*=1000
        print(pred.shape)
        x = np.arange(0,len(pred)).T

        plt.plot(x,pred)
        plt.xlabel("Frame Number")
        plt.ylabel("Score")
        plt.title("Abnormal Activity Detection")
        plt.show()



"""
    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device)

        if step % 5 == 0 and step > 200:

            auc = test(test_loader, model, args, viz, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model.state_dict(), './ckpt/' + args.model_name + '{}-i3d.pkl'.format(step))
                save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step)))
    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
"""