import shutil
import time
import warnings
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

warnings.filterwarnings("ignore")
import torch.utils.data as data
import os
import argparse
from sklearn.metrics import f1_score, confusion_matrix
from data_preprocessing.sam import SAM
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import datetime
from models.model_7cls import *
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_ferplus import FerPlusSet
from wechat import wechat_inform
from util import LabelSmoothingCrossEntropy
from thop import profile
from torchsampler import ImbalancedDatasetSampler

warnings.filterwarnings("ignore", category=UserWarning)

now = datetime.datetime.now()
time_str = now.strftime("%Y%m%d%H%M-")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=r'./dataset/FERPlus')
parser.add_argument('--data_type', default='RAF-DB', choices=['RAF-DB', 'AffectNet-7', 'FER-Plus'],
                    type=str, help='dataset option')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model.pth')
parser.add_argument('--best_checkpoint_path', type=str, default='./checkpoint/' + time_str + 'model_best.pth')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=250, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=200, type=int, metavar='N')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

parser.add_argument('--lr', '--learning-rate', default=0.00004, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=60, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('-e', '--evaluate', default=None, type=str, help='evaluate model on test set')
parser.add_argument('--beta', type=float, default=0.6)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    best_acc = 0
    print('Training time: ' + now.strftime("%Y-%m-%d %H:%M"))
    print('batch_size:', args.batch_size)
    # create model
    model = BaseLine(img_size=224, num_classes=7).cuda()
    # img = torch.rand(size=(1, 3, 224, 224)).cuda()
    # img_lbp = torch.rand(size=(1, 3, 224, 224)).cuda()
    # flops, params = profile(model, inputs=(img, img_lbp))
    # print(f'flops:{flops / 1000 ** 3}G,params:{params / 1000 ** 2}M')

    CE_criterion = torch.nn.CrossEntropyLoss()
    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)

    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    recorder = RecorderMeter(args.epochs)
    recorder1 = RecorderMeter1(args.epochs)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            recorder1 = checkpoint['recorder1']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True

    # Data loading code
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    data_transforms_train = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((224, 224)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225]),
                                                transforms.RandomErasing(scale=(0.02, 0.1))
                                                ])
    if args.data_type == 'RAF-DB':
        data_path = './dataset/raf-db'
        feature_path = './dataset/raf-db-lbphog'
        train_dataset = RafDataSet(data_path, feature_path, train=True, transform=data_transforms_train, basic_aug=True)
        test_dataset = RafDataSet(data_path, feature_path, train=False, transform=data_transforms_val)
    if args.data_type == 'FER-Plus':
        data_path = './dataset/FERPlus'
        train_dataset = FerPlusSet(data_path, train=True, transform=data_transforms_train, basic_aug=True)
        test_dataset = FerPlusSet(data_path, train=False, transform=data_transforms_val)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               # sampler=ImbalancedDatasetSampler(train_dataset),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    if args.evaluate is not None:
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            best_acc = checkpoint['best_acc']
            best_acc = best_acc.to()
            print(f'best_acc:{best_acc}')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.evaluate))
        validate(val_loader, model, CE_criterion, args)
        return

    matrix = None
    for epoch in range(args.start_epoch, args.epochs):

        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
        print('Current learning rate:%.6f ' % current_learning_rate)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current learning rate:%.6f ' % current_learning_rate + '\n')
        # train for one epoch
        s = time.time()
        train_acc, train_los = train(train_loader, model, CE_criterion, lsce_criterion, optimizer, epoch, args)
        s1 = time.time()
        print("====train acc" + str(train_acc) + "====")
        print('train time', s1 - s)
        # evaluate on validation set
        val_acc, val_los, output, target, D = validate(val_loader, model, CE_criterion, lsce_criterion, args)
        scheduler.step()

        recorder.update(epoch, train_los, train_acc[0], val_los, val_acc[0])
        recorder1.update(output, target)

        curve_name = time_str + 'cnn.png'
        recorder.plot_curve(os.path.join('./log/', curve_name))

        # remember the best acc and save checkpoint
        is_best = val_acc[0] > best_acc
        best_acc = max(val_acc[0], best_acc)

        print('Current best accuracy: ', best_acc)

        if is_best:
            matrix = D
            wechat_inform("[Epoch %d] Validation accuracy:%.4f" % (
                epoch, best_acc))

        print('Current best matrix: ', matrix)

        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write('Current best accuracy: ' + str(best_acc) + '\n')

        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder1': recorder1,
                         'recorder': recorder}, is_best, args)


def train(train_loader, model, CE_criterion, lsce_criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top = AverageMeter('High Level Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [[losses], [top]],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    for i, (images, feature_images, target) in enumerate(train_loader):
        # print(images.shape)
        images = images.cuda()
        feature_images = feature_images.cuda()
        target = target.cuda()

        # compute output
        # output, output2, output3 = model(images, feature_images)
        output = model(images, feature_images)
        loss = CE_criterion(output, target.long())
        # loss = lsce_criterion(output, target.long())
        # loss = 2 * lsce_loss + CE_loss

        # measure accuracy and record loss
        acc = accuracy(output, target, topk=(1,))

        losses.update(loss.item(), images.size(0))
        top.update(acc[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        optimizer.first_step(zero_grad=True)

        # compute output
        output = model(images, feature_images)
        loss = CE_criterion(output, target.long())

        # measure accuracy and record loss
        acc = accuracy(output, target, topk=(1,))

        losses.update(loss.item(), images.size(0))
        top.update(acc[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        optimizer.second_step(zero_grad=True)

        # print loss and accuracy
        if i % args.print_freq == 0 or i == len(train_loader.batch_sampler) - 1:
            progress.display(i)

    return [top.avg], losses.avg


def validate(val_loader, model, CE_criterion,lsce_criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top = AverageMeter('High Level Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [[losses], [top]],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    D = [[0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]
    # D = [[0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 0, 0]]
    with torch.no_grad():
        for i, (images, feature_images, target) in enumerate(val_loader):
            images = images.cuda()
            feature_images = feature_images.cuda()
            target = target.cuda()
            output = model(images, feature_images)
            loss = CE_criterion(output, target.long())

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))

            top.update(acc1[0].item(), images.size(0))

            topk = (1,)
            # """Computes the accuracy over the k top predictions for the specified values of k"""
            with torch.no_grad():
                maxk = max(topk)
                # batch_size = target.size(0)
                _, pred = output.topk(maxk, 1, True, True)
                pred = pred.t()

            output = pred
            target = target.squeeze().cpu().numpy()
            output = output.squeeze().cpu().numpy()

            im_re_label = np.array(target)
            im_pre_label = np.array(output)
            y_ture = im_re_label.flatten()
            im_re_label.transpose()
            y_pred = im_pre_label.flatten()
            im_pre_label.transpose()

            C = metrics.confusion_matrix(y_ture, y_pred, labels=[0, 1, 2, 3, 4, 5, 6])
            D += C

            if i % args.print_freq == 0 or i == len(val_loader.batch_sampler) - 1:
                progress.display(i)

        print(' **** Accuracy {top.avg:.3f} *** '.format(top=top))
        with open('./log/' + time_str + 'log.txt', 'a') as f:
            f.write(' * Accuracy {top.avg:.3f}'.format(top=top) + '\n')
    print(D)
    return [top.avg], losses.avg, output, target, D


def save_checkpoint(state, is_best, args):
    torch.save(state, args.checkpoint_path)
    if is_best:
        best_state = state.pop('optimizer')
        torch.save(best_state, args.best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        for meter in self.meters:
            entries += [str(l) for l in meter]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(txt_name, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


labels = ['A', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']


class RecorderMeter1(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, output, target):
        self.y_pred = output
        self.y_true = target

    def plot_confusion_matrix(self, cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        y_true = self.y_true
        y_pred = self.y_pred

        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 8), dpi=120)

        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
        # offset the tick
        tick_marks = np.arange(len(7))
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
        # show confusion matrix
        plt.savefig('./log/confusion_matrix.png', format='png')
        # fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print('Saved figure')
        plt.show()

    def matrix(self):
        target = self.y_true
        output = self.y_pred
        im_re_label = np.array(target)
        im_pre_label = np.array(output)
        y_ture = im_re_label.flatten()
        # im_re_label.transpose()
        y_pred = im_pre_label.flatten()
        im_pre_label.transpose()


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):
        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)


if __name__ == '__main__':
    main()
