import argparse
import math
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from helper import pmath
from helper.helper import get_optimizer, load_dataset
from helper.hyperbolicLoss import PeBusePenalty
from models.cifar import resnet as resnet_cifar
from models.cifar import densenet as densenet_cifar
from models.cub import resnet as resnet_cub

from sklearn.metrics import roc_auc_score

def main_train(model, trainloader, optimizer, initialized_loss, train_classes_map, c=1.0):
    # Set mode to training.
    model.train()
    avgloss, avglosscount, newloss, acc, newacc = 0., 0., 0., 0., 0.

    # Go over all batches.
    for bidx, (data, target) in enumerate(trainloader):
        # Data to device.
        target_tmp = target.cuda()

        target = model.polars[target.apply_(lambda x: train_classes_map[x])]
        data = torch.autograd.Variable(data).cuda()
        target = torch.autograd.Variable(target).cuda()
        # Compute outputs and losses.
        output = model(data)
        output_exp_map = pmath.expmap0(output, c=c)

        loss_function = initialized_loss(output_exp_map, target)

        # Backpropagation.
        optimizer.zero_grad()
        loss_function.backward()
        optimizer.step()

        avgloss += loss_function.item()
        avglosscount += 1.
        newloss = avgloss / avglosscount

        output = model.predict(output_exp_map).float()
        pred = output.max(1, keepdim=True)[1]       # todo shoud map to selected class?
        acc += pred.eq(target_tmp.view_as(pred)).sum().item()

    trainlen = len(trainloader.dataset)
    newacc = acc / float(trainlen)

    # I am returning new loss to show in the tensorboard!
    return newacc, newloss


def main_test(model, testloader, testloader_os, initialized_loss, train_classes, train_classes_map, c=1.0):
    # Set model to evaluation and initialize accuracy and cosine similarity.
    model.eval()
    acc = 0
    loss = 0

    osr_true_labels = []
    osr_scores = []
    # Go over all batches.
    with torch.no_grad():
        for data, target in testloader:
            # Data to device.
            data = torch.autograd.Variable(data).cuda()
            target = target.cuda(non_blocking=True)
            target = torch.autograd.Variable(target)
            target_loss = model.polars[target.cpu().apply_(lambda x: train_classes_map[x])]

            # Forward.
            output = model(data).float()
            output_exp_map = pmath.expmap0(output, c=c)
            scores = output_exp_map.norm(dim=-1, p=2, keepdim=True)
            for score in scores:
                osr_scores.append(score.item())
                osr_true_labels.append(1)

            output = model.predict(output_exp_map).float()
            pred = output.max(1, keepdim=True)[1]   # todo shoud map to selected class?
            acc += pred.eq(target.view_as(pred)).sum().item()

            loss += initialized_loss(output_exp_map, target_loss.cuda())

    # Print results.
    testlen = len(testloader.dataset)

    avg_acc = acc / float(testlen)
    avg_loss = loss / float(testlen)

    # open-set
    with torch.no_grad():
        for data, target in testloader_os:
            data = torch.autograd.Variable(data).cuda()
            # Forward.
            output = model(data).float()
            output_exp_map = pmath.expmap0(output, c=c)
            scores_os = output_exp_map.norm(dim=-1, p=2, keepdim=True)
            for score_os in scores_os:
                osr_scores.append(score_os.item())
                osr_true_labels.append(0)

    return avg_acc, avg_loss, roc_auc_score(osr_true_labels, osr_scores)


def parse_args():
    parser = argparse.ArgumentParser(description="classification")
    parser.add_argument("--data_name", dest="data_name", default="cifar100",
                        choices=["cifar100", "cifar10", "cub"], type=str)  # choose tha name of the dataset

    parser.add_argument("--datadir", dest="datadir", default="dat/", type=str)
    parser.add_argument("--resdir", dest="resdir", default="res/", type=str)
    parser.add_argument("--hpnfile", dest="hpnfile", default="", type=str)
    parser.add_argument("--logdir", dest="logdir", default="", type=str)
    parser.add_argument("--loss", dest="loss_name", default="PeBuseLoss", type=str)

    parser.add_argument("-n", dest="network", default="resnet32", type=str)
    parser.add_argument("-r", dest="optimizer", default="sgd", type=str)
    parser.add_argument("-l", dest="learning_rate", default=0.01, type=float)
    parser.add_argument("-m", dest="momentum", default=0.9, type=float)
    parser.add_argument("-c", dest="decay", default=0.0001, type=float)
    parser.add_argument("-s", dest="batch_size", default=128, type=int)
    parser.add_argument("-e", dest="epochs", default=250, type=int)
    parser.add_argument("-p", dest="penalty", default='dim', type=str)  # choose penalty in loss
    parser.add_argument("--mult", dest="mult", default=0.1, type=float)
    parser.add_argument("--curv", dest="curv", default=1.0, type=float)

    parser.add_argument("--seed", dest="seed", default=100, type=int)
    parser.add_argument("--drop1", dest="drop1", default=500, type=int)
    parser.add_argument("--drop2", dest="drop2", default=1000, type=int)
    parser.add_argument("--do_decay", dest="do_decay", default=False, type=bool)
    args = parser.parse_args()
    return args


#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user parameters and set device.
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    kwargs = {'num_workers': 2, 'pin_memory': True}
    cudnn.benchmark = True

    do_decay = args.do_decay
    curvature = args.curv

    # I want to use tensorboard to check the loss changes
    log_dir = os.path.join('./runs/' + args.data_name, args.logdir)
    writer = SummaryWriter(log_dir=log_dir)

    # Set the random seeds.
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    train_classes = [0, 1, 2, 3, 4, 5]  # todo parametrize
    train_classes_map = {train_classes[i]: i for i in range(len(train_classes))}

    # Load data.
    batch_size = args.batch_size
    trainloader, testloader, testloader_os = load_dataset(args.data_name, args.datadir, batch_size, kwargs, train_classes)

    if not os.path.exists(args.resdir):
        os.makedirs(args.resdir)

    # Load the polars and update the trainy labels.
    classpolars = torch.from_numpy(np.load(args.hpnfile)).float()
    # calculate radius of ball
    # This part is useful when curvature is not 1.
    radius = 1.0 / math.sqrt(curvature)
    classpolars = classpolars * radius

    # hpnfile name is like prototypes-xd-yc.npy : x : dimension of prototype, y: number of classes
    args.output_dims = int(args.hpnfile.split("/")[-1].split("-")[1][:-1])
    print(args.output_dims)

    # Load the model.
    if (args.data_name == "cifar100") or (args.data_name == "cifar10"):
        if args.network == "resnet32":
            model = resnet_cifar.ResNet(32, args.output_dims, 1, classpolars)
        elif args.network == "densenet121":
            model = densenet_cifar.DenseNet121(args.output_dims, classpolars)
        else:
            print('The model you have chosen is not available. I am choosing resnet for you.')
            model = resnet_cifar.ResNet(32, args.output_dims, 1, classpolars)
    elif args.data_name == "cub":
        if args.network == "resnet32":
            model = resnet_cub.ResNet34(args.output_dims, classpolars)
        else:
            print('The model you have chosen is not available. I am choosing resnet for you.')
            model = resnet_cub.ResNet34(args.output_dims, classpolars)
    else:
        raise Exception('Selected dataset is not available.')

    model = model.to(device)
    print('First time model initialization.')

    # Load the optimizer.
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.learning_rate, args.momentum, args.decay)

    # Initialize the loss functions.
    choose_penalty = args.penalty
    f_loss = PeBusePenalty(args.output_dims, penalty_option=choose_penalty, mult=args.mult).cuda()

    # Main loop.
    testscores = []
    learning_rate = args.learning_rate
    for i in range(args.epochs):
        print(i)

        # Learning rate decay.
        if i in [args.drop1, args.drop2] and do_decay:
            learning_rate *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        # Train and test.
        acc, loss = main_train(model, trainloader, optimizer, f_loss, train_classes_map, c=curvature)
        print("train_acc:", acc, "\ttrain_loss:", loss)

        # add the train loss to the tensorboard writer
        writer.add_scalar("Loss/train", loss, i)
        writer.add_scalar("Accuracy/train", acc, i)

        if i != 0 and (i % 1 == 0 or i == args.epochs - 1):
            test_acc, test_loss, auroc = main_test(model, testloader, testloader_os, f_loss, train_classes, train_classes_map, c=curvature)
            print("test_acc:", test_acc, "\ttest_loss:", test_loss, "\tAUROC:", auroc)

            testscores.append([i, test_acc])

            writer.add_scalar("Loss/test", test_loss, i)
            writer.add_scalar("Accuracy/test", test_acc, i)
            writer.add_scalar("AUROC/test", auroc, i)

    writer.flush()
    writer.close()
