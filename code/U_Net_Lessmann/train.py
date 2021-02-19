import os
import time
import argparse
import numpy as np
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# import the torch tensorboard 
from torch.utils.tensorboard import SummaryWriter

from data.dataset import CSIDataset
from utils.metrics import DiceCoeff, Segloss
from iterativeFCN import IterativeFCN

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)


def train_single(model, device, img_patch, ins_patch, gt_patch, weight, c_label, optimizer):

    logging.debug('Start single training step')

    torch.cuda.empty_cache()

    model.train()
    correct = 0

    # convert data to float, just in case
    img_patch = img_patch.float()
    ins_patch = ins_patch.float()
    gt_patch = gt_patch.float()
    weight = weight.float()
    c_label = c_label.float()

    # pick a random scan
    optimizer.zero_grad()

    # concatenate the img_patch and ins_patch
    input_patch = torch.cat((img_patch, ins_patch), dim=1)
    input_patch, gt_patch, weight, c_label = input_patch.to(device), gt_patch.to(device), weight.to(device), c_label.to(
        device)

    S, C = model(input_patch.float())

    # calculate dice coefficient
    pred = torch.round(S).detach()
    train_dice_coef = DiceCoeff(pred, gt_patch.detach())

    # calculate total loss
    lamda = 0.1
    FP, FN = Segloss(S, gt_patch, weight)
    s_loss = lamda * FP + FN
    c_loss = F.binary_cross_entropy(torch.unsqueeze(C, dim=0), c_label)
    train_loss = s_loss + c_loss

    logging.debug("train_dice_coef: %s, S Loss: %s, C Loss: %s" % (train_dice_coef, s_loss.item(), c_loss.item()))

    if C.round() == c_label:
        correct = 1

    # optimize the parameters
    train_loss.backward()
    optimizer.step()

    return train_loss.item(), correct, train_dice_coef


def test_single(model, device, img_patch, ins_patch, gt_patch, weight, c_label):
    torch.cuda.empty_cache()

    model.eval()
    correct = 0

    img_patch = img_patch.float()
    ins_patch = ins_patch.float()
    gt_patch = gt_patch.float()
    weight = weight.float()
    c_label = c_label.float()

    input_patch = torch.cat((img_patch, ins_patch), dim=1)
    input_patch, gt_patch, weight, c_label = input_patch.to(device), gt_patch.to(device), weight.to(device), c_label.to(
        device)

    with torch.no_grad():
        S, C = model(input_patch.float())

    # calculate dice coefficient
    pred = torch.round(S).detach()
    test_dice_coef = DiceCoeff(pred, gt_patch.detach())

    # calculate total loss
    lamda = 0.1
    FP, FN = Segloss(S, gt_patch, weight)
    s_loss = lamda * FP + FN
    c_loss = F.binary_cross_entropy(torch.unsqueeze(C, dim=0), c_label)

    logging.debug("test_dice_coef: %s, S Loss: %s, C Loss: %s" % (test_dice_coef, s_loss.item(), c_loss.item()))

    if C.round() == c_label:
        correct = 1

    test_loss = s_loss + c_loss

    return test_loss.item(), correct, test_dice_coef


def get_patch_slices(patch):
    
    batch = np.zeros((3,1,128, 128))
    
    batch[0,0,:,:] = patch[0, 0, int(128/2), : ,:]
    batch[1,0,:,:] = patch[0, 0, :, int(128/2),:]
    batch[2,0,:,:] = patch[0, 0, :,:, int(128/2)]

    return batch


if __name__ == "__main__":
    # Version of Pytorch
    logging.info("Pytorch Version:%s" % torch.__version__)


    # Training args
    parser = argparse.ArgumentParser(description='Iterative Fully Convolutional Network')
    parser.add_argument('--dataset', type=str, default='./crop_isotropic_dataset',
                        help='path of processed dataset')
    parser.add_argument('--weight', type=str, default='./weights',
                        help='path of processed dataset')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help='path of training snapshot')
    parser.add_argument('--tensorboard', type=str, default='./experiment', 
                        help='foldername for the tensorboard logging information')
    parser.add_argument('--resume', type=bool, default=False,
                        help='resume training by loading last snapshot')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1)')
    parser.add_argument('--iterations', type=int, default=1000, metavar='N',
                        help='number of iterations to train (default: 80000)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='number of iterations to log (default: 1000)')
    parser.add_argument('--eval_iters', type=int, default=100, metavar='N',
                        help='number of iterations to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    # open de summarywriter
    writer = SummaryWriter(args.tensorboard)

    # set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Assure folders for storing weights are checkpoints are ready:
    weightdir = os.path.abspath(args.weight)
    weightfile = os.path.join(weightdir, 'IterativeFCN_best_valid.pth')
    checkpointdir = os.path.abspath(args.checkpoints)
    checkpointfile = os.path.join(checkpointdir, 'latest_checkpoints.pth')
    Path(weightdir).mkdir(parents=True, exist_ok=True)
    Path(checkpointdir).mkdir(parents=True, exist_ok=True)

    # Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug(f'device : {device}')

    # Create model and check if we want to resume training
    model = IterativeFCN(num_channels=4).to('cuda')

    batch_size = args.batch_size
    batch_size_valid = batch_size

    train_dataset = CSIDataset(args.dataset, subset='train')
    test_dataset = CSIDataset(args.dataset, subset='test')
    logging.info(f'train dataset : {train_dataset}')
    logging.info(f'test dataset : {test_dataset}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    train_dice, test_dice = [], []
    best_train_loss, best_test_dice = 0., 0.

    total_iteration = args.iterations
    train_interval = args.log_interval
    eval_interval = args.eval_iters

    iteration = 1

    if args.resume:
        logging.info("Resume Training: Load states from latest checkpoint.")
        checkpoint = torch.load(os.path.join(args.checkpoints, 'latest_checkpoints.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iteration = checkpoint['iteration']
        train_loss = checkpoint['train_loss']
        test_loss = checkpoint['test_loss']
        train_acc = checkpoint['train_acc']
        test_acc = checkpoint['test_acc']

    # Start Training
    logging.info('START TRAINING')
    start_time = time.time()
    epoch_train_dice = []
    epoch_test_dice = []
    epoch_train_loss = []
    epoch_test_loss = []
    epoch_train_accuracy = 0.
    epoch_test_accuracy = 0.
    correct_train_count = 0
    correct_test_count = 0

    while iteration <= args.iterations:
        img_patch, ins_patch, gt_patch, weight, c_label = next(iter(train_loader))
        t_loss, t_c, t_dice = train_single(model, device, img_patch, ins_patch, gt_patch, weight, c_label, optimizer)
        epoch_train_loss.append(t_loss)
        epoch_train_dice.append(t_dice)
        correct_train_count += t_c

        writer.add_scalar('train_loss', t_loss, global_step=iteration)
        writer.add_scalar('train_dice', t_dice, global_step=iteration)

        logging.debug(f'image patch shape {img_patch.shape}')
        logging.debug(f'ins patch shape {ins_patch.shape}')
        logging.debug(f'ground truth patch shape {gt_patch.shape}')

        img_batch = get_patch_slices(img_patch)
        ins_batch = get_patch_slices(ins_patch)
        gt_batch = get_patch_slices(gt_patch)

        logging.debug(f'tensorboard output {img_batch.shape}')

        writer.add_images('image_patches', img_batch, global_step=iteration, dataformats='NCHW')
        writer.add_images('ins_patches', ins_batch, global_step=iteration, dataformats='NCHW')
        writer.add_images('gt_patches', gt_batch, global_step=iteration, dataformats='NCHW')

        if iteration % args.log_interval == args.log_interval - 1:
            avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            avg_train_dice = (sum(epoch_train_dice) / len(epoch_train_dice)) * 100
            epoch_train_accuracy = (correct_train_count / train_interval) * 100

            writer.add_scalar('train_accuracy', epoch_train_accuracy, global_step=iteration)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                logging.info('--- Saving model at Avg Train Dice:{:.2f}%  ---'.format(avg_train_dice))
                torch.save(model.state_dict(), weightfile)

            # validation process
            for i in range(args.eval_iters):
                img_patch, ins_patch, gt_patch, weight, c_label = next(iter(test_loader))
                v_loss, v_c, v_dice = test_single(model, device, img_patch, ins_patch, gt_patch, weight, c_label)
                epoch_test_loss.append(v_loss)
                epoch_test_dice.append(v_dice)
                correct_test_count += v_c

            avg_test_loss = sum(epoch_test_loss) / len(epoch_test_loss)
            avg_test_dice = (sum(epoch_test_dice) / len(epoch_test_dice)) * 100
            epoch_test_accuracy = (correct_test_count / eval_interval) * 100

            writer.add_scalars('avg_evaluation_loss', {'train' : avg_train_loss, 'test' : avg_test_loss}, global_step=iteration)
            writer.add_scalars('avg_evaluation_dice', {'train' : avg_train_dice, 'test' : avg_test_dice}, global_step=iteration)
            writer.add_scalars('avg_evaluation_accuracy', {'train' : epoch_train_accuracy, 'test' : epoch_test_accuracy}, global_step=iteration)

            if avg_test_dice > best_test_dice:
                best_test_dice = avg_test_dice
                logging.info('--- Saving model at Avg Test Dice:{:.2f}%  ---'.format(avg_test_dice))
                torch.save(model.state_dict(), weightfile)

            train_loss.append(epoch_train_loss)
            test_loss.append(epoch_test_loss)
            train_acc.append(epoch_train_accuracy)
            test_acc.append(epoch_test_accuracy)

            # save snapshot for resume training
            logging.debug('--- Saving snapshot ---')
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'train_dice': train_dice,
                'test_dice': test_dice,
                'best_train_loss': best_train_loss,
                'best_test_dice': best_test_dice}, checkpointfile)

            logging.info(f' *** Time {int(time.time() - start_time)} seconds *** ')
            epoch_train_dice = []
            epoch_test_dice = []
            epoch_train_loss = []
            epoch_test_loss = []
            epoch_train_accuracy = 0.
            epoch_test_accuracy = 0.
            correct_train_count = 0
            correct_test_count = 0

        iteration += 1
        if not iteration % 25:
            logging.info(f'iteration {iteration}')

    print(epoch_test_dice)
    writer.close()