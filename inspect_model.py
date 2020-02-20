import numpy as np 
import getopt, sys
import torch
import torch.nn as nn
from option.default_option import TrainOptions
from model_search import Network
from genotypes import * 
from option.default_option import TrainOptions
from visualize import *

"""
Loads a model checkpoint from commandline arguments and displays with additional option to visualize
the checkpoint:
- epoch
- training loss
- training accuracy
- validation loss
- validation accuracy

Command Line Arguments:
- experiment name: --experiment=
- epoch number: --epoch=
- visualize the model as .png in /visualizations (bool): --visualize=
"""

opt = TrainOptions()

def initialize():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt = TrainOptions()

    criterion = nn.CrossEntropyLoss().to(device)
    model = Network(opt.init_channels, 10, opt.layers, criterion)
    model.to(device)

    optimizer_model = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer_arch = torch.optim.Adam(model.arch_parameters(), lr=opt.arch_learning_rate, betas=opt.arch_betas, weight_decay=opt.arch_weight_decay)

    return device, opt, criterion, model, optimizer_model, optimizer_arch

def load_checkpoint(LOAD_EPOCH, experiment):
    """
    Loads model checkpoint metadata saved in /experiments at a particular epoch
    """
    checkpoint = torch.load('experiments/' + experiment + '/weights_epoch_' + LOAD_EPOCH + '.pt', map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
    optimizer_arch.load_state_dict(checkpoint['optimizer_arch_state_dict'])

    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    train_acc_top1 = checkpoint['train_acc_top1']
    train_acc_top5 = checkpoint['train_acc_top5']
    valid_loss = checkpoint['valid_loss']
    valid_acc_top1 = checkpoint['valid_acc_top1']
    valid_acc_top5 = checkpoint['valid_acc_top5']
    model_state_dict = checkpoint['model_state_dict']
    arch_alphas = checkpoint['arch_alphas']

    return epoch, train_loss, train_acc_top1, train_acc_top5, valid_loss, valid_acc_top1, valid_acc_top5, model_state_dict, arch_alphas

if __name__ == '__main__':
    opt_list, _ = getopt.getopt(sys.argv[1:], 'x', ['experiment=', 'epoch=', 'visualize='])
    experiment, LOAD_EPOCH, visualize = opt_list[0][1], opt_list[1][1], opt_list[2][1] == 'True'

    device, opt, criterion, model, optimizer_model, optimizer_arch = initialize()
    epoch, train_loss, train_acc_top1, train_acc_top5, valid_loss, valid_acc_top1, valid_acc_top5, model_state_dict, arch_alphas = load_checkpoint(LOAD_EPOCH, experiment)

    if visualize:
        temperature = opt.initial_temp * np.exp(opt.anneal_rate * epoch)

        alpha_normal, alpha_reduce = arch_alphas[0], arch_alphas[1]

        m_normal = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            torch.tensor([temperature]), torch.tensor(alpha_normal)) 
        m_reduce = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            torch.tensor([temperature]) , torch.tensor(alpha_reduce))

        alpha_normal = m_normal.sample().cpu().numpy()
        alpha_reduce = m_reduce.sample().cpu().numpy()

        ex = genotype(alpha_normal, alpha_reduce)
        plot(ex.normal, './visualizations/' + experiment + '/normal_epoch_' + str(epoch))
        plot(ex.reduce, './visualizations/' + experiment + '/reduction_epoch_' + str(epoch))
        print("Saved visualization to normal.png and reduction.png")

    print('SNAS status')
    print('epoch:', epoch)
    print('train_loss:', train_loss)
    print('train_acc_top1:', train_acc_top1)
    print('train_acc_top5:', train_acc_top5)
    print('valid_loss:', valid_loss)
    print('valid_acc_top1:', valid_acc_top1)
    print('valid_acc_top5:', valid_acc_top5)