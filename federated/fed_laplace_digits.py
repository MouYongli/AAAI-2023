"""
federated learning with different aggregation strategy on benchmark exp.
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
from copy import deepcopy
from nets.models import DigitModel
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import data_utils

global eps
eps = 1e-5

def prepare_data(args):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # MNIST
    mnist_trainset     = data_utils.DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=True,  transform=transform_mnist)
    mnist_testset      = data_utils.DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=False, transform=transform_mnist)
    # SVHN
    svhn_trainset      = data_utils.DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent,  train=True,  transform=transform_svhn)
    svhn_testset       = data_utils.DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent,  train=False, transform=transform_svhn)
    # USPS
    usps_trainset      = data_utils.DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent,  train=True,  transform=transform_usps)
    usps_testset       = data_utils.DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent,  train=False, transform=transform_usps)
    # Synth Digits
    synth_trainset     = data_utils.DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,  train=True,  transform=transform_synth)
    synth_testset      = data_utils.DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,  train=False, transform=transform_synth)
    # MNIST-M
    mnistm_trainset     = data_utils.DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,  train=True,  transform=transform_mnistm)
    mnistm_testset      = data_utils.DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,  train=False, transform=transform_mnistm)
    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_test_loader  = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch,  shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch,  shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch,  shuffle=True)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch,  shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)
    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    test_loaders  = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]
    return train_loaders, test_loaders

def train(args, round_num, client_idx, client_alpha, server_model, client_model_set,
          server_omega, client_omega_set, train_loaders, loss_fun, device):
    log_ce_loss = 0
    log_csd_loss = 0
    num_data = 0
    correct = 0
    client_model_set[client_idx] = client_model_set[client_idx].to(device)
    optimizer = optim.SGD(client_model_set[client_idx].parameters(), lr=args.lr)
    new_omega = dict()
    new_mu = dict()
    data_alpha = client_alpha[client_idx]
    for name, param in client_model_set[client_idx].named_parameters():
        new_omega[name] = deepcopy(server_omega[name])
        new_mu[name] = deepcopy(server_model.state_dict()[name])
    client_model_set[client_idx].train()
    for batch_idx, (data, target) in enumerate(train_loaders[client_idx]):
        num_data += target.size(0)
        data, target = data.to(device).float(), target.to(device).long()
        optimizer.zero_grad()
        output = client_model_set[client_idx](data)
        ce_loss = loss_fun(output, target)
        csd_loss = get_csd_loss(client_idx, new_mu, new_omega, round_num) if args.csd_importance > 0 else 0
        ce_loss.backward(retain_graph=True)
        for name, param in client_model_set[client_idx].named_parameters():
            if param.grad is not None:
                client_omega_set[client_idx][name] += (len(target) / len(train_loaders[client_idx])) * param.grad.data.clone() ** 2
        optimizer.zero_grad()
        loss = ce_loss + args.csd_importance * csd_loss
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(client_model_set[client_idx].parameters(), args.clip)
        optimizer.step()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
        log_ce_loss += ce_loss.item()
        log_csd_loss += csd_loss.item() if args.csd_importance > 0 else 0
    client_model_set[client_idx] = client_model_set[client_idx].cpu()
    return log_ce_loss / len(train_loaders[client_idx]), log_csd_loss / len(train_loaders[client_idx]), correct / num_data

def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []
    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())
        output = model(data)
        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

################# Key Function ########################
def communication(args, client_alpha, server_model, client_model_set, server_omega, client_omega_set):
    new_param = {}
    new_omega = {}
    client_num = len(client_model_set)
    with torch.no_grad():
        for name, param in server_model.named_parameters():
            new_param[name] = param.data.zero_()
            new_omega[name] = server_omega[name].data.zero_()
            for client_idx in range(len(client_model_set)):
                new_param[name] += client_alpha[client_idx] * client_omega_set[client_idx][name] * \
                                   client_model_set[client_idx].state_dict()[name].to(device)
                new_omega[name] += client_alpha[client_idx] * client_omega_set[client_idx][name]
            new_param[name] /= (new_omega[name] + eps)

        for name, param in server_model.named_parameters():
            server_model.state_dict()[name].data.copy_(new_param[name])
            server_omega[name] = new_omega[name]
            for client_idx in range(client_num):
                client_model_set[client_idx].state_dict()[name].data.copy_(new_param[name].cpu())
                client_omega_set[client_idx][name].data.copy_(new_omega[name])
    return  server_model, client_model_set, server_omega, client_omega_set

def get_csd_loss(client_model, mu, omega, round_num):
    loss_set = []
    for name, param in client_model.named_parameters():
        theta = client_model.state_dict()[name]
        loss_set.append((0.5 / round_num) * (omega[name] * ((theta - mu[name]) ** 2)).sum())
    return sum(loss_set)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed= 1
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed)
    print('Device:', device)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help ='whether to make a log')
    parser.add_argument('--test', action='store_true', help ='test the pretrained model')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type = int, default= 32, help ='batch size')
    parser.add_argument('--iters', type = int, default=100, help = 'iterations for communication')
    parser.add_argument('--wk_iters', type = int, default=1, help = 'optimization iters in local worker between communication')
    parser.add_argument('--mode', type = str, default='fedlaplace', help='fedlaplace')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--csd_importance', type=float, default=0)
    parser.add_argument('--clip', type=float, default=10)
    parser.add_argument('--save_path', type = str, default='../checkpoint/digits', help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help ='resume training from the save path checkpoint')
    args = parser.parse_args()

    exp_folder = 'federated_digits'
    args.save_path = os.path.join(args.save_path, exp_folder)
    
    log = args.log
    if log:
        log_path = os.path.join('../logs/digits/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'{}.log'.format(args.mode)), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch: {}\n'.format(args.batch))
        logfile.write('    iters: {}\n'.format(args.iters))
        logfile.write('    wk_iters: {}\n'.format(args.wk_iters))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}'.format(args.mode))

    # prepare the data
    train_loaders, test_loaders = prepare_data(args)
    # name of each client dataset
    datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
    
    # federated setting
    client_num = len(datasets)
    client_alpha = [args.batch * len(tl) for tl in train_loaders]
    client_alpha = [alpha / sum(client_alpha) for alpha in client_alpha]

    loss_fun = nn.CrossEntropyLoss()
    server_model = DigitModel().to(device)
    server_omega = dict()
    client_model_set = [DigitModel() for _ in range(client_num)]
    client_omega_set = [dict() for _ in range(client_num)]

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('../snapshots/digits/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        for test_idx, test_loader in enumerate(test_loaders):
            _, test_acc = test(server_model, test_loader, loss_fun, device)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        for client_idx in range(client_num):
            client_model_set[client_idx].load_state_dict(checkpoint['server_model'])
        resume_iter = int(checkpoint['a_iter']) + 1
        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0

    # Inital client weights
    for client_idx in range(client_num):
        for key in server_model.state_dict().keys():
            client_model_set[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        for name, param in deepcopy(client_model_set[client_idx]).named_parameters():
            client_omega_set[client_idx][name] = torch.zeros_like(param.data).to(device)
    for name, param in deepcopy(server_model).named_parameters():
        server_omega[name] = torch.zeros_like(param.data).to(device)

    # start training
    for a_iter in range(resume_iter, args.iters):
        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))
            for client_idx in range(client_num):
                train(args, a_iter, client_idx, client_alpha, server_model, client_model_set, server_omega, client_omega_set, train_loaders, loss_fun, device)
        # aggregation
        server_model, client_model_set, server_omega, client_omega_set = communication(args, client_alpha, server_model, client_model_set, server_omega, client_omega_set)
        # report after aggregation
        for client_idx in range(client_num):
                train_loss, train_acc = test(client_model_set[client_idx].to(device), train_loaders[client_idx], loss_fun, device)
                print(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx] ,train_loss, train_acc))
                if args.log:
                    logfile.write(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx] ,train_loss, train_acc))\
        # start testing
        for test_idx, test_loader in enumerate(test_loaders):
            test_loss, test_acc = test(client_model_set[test_idx].to(device), test_loader, loss_fun, device)
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
            if args.log:
                logfile.write(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss, test_acc))

    # Save checkpoint
    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    torch.save({
        'server_model': server_model.state_dict(),
    }, SAVE_PATH)
    if log:
        logfile.flush()
        logfile.close()