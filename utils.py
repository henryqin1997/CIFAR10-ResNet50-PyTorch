import torch
import torch.nn.functional as F
# import horovod.torch as hvd
import sys
import time
import os

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()

# def save_checkpoint(model, optimizer, checkpoint_format, epoch):
#     if hvd.rank() == 0:
#         filepath = checkpoint_format.format(epoch=epoch + 1)
#         state = {
#             'model': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#         }
#         torch.save(state, filepath)
#
# class LabelSmoothLoss(torch.nn.Module):
#
#     def __init__(self, smoothing=0.0):
#         super(LabelSmoothLoss, self).__init__()
#         self.smoothing = smoothing
#
#     def forward(self, input, target):
#         log_prob = F.log_softmax(input, dim=-1)
#         weight = input.new_ones(input.size()) * \
#             self.smoothing / (input.size(-1) - 1.)
#         weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
#         loss = (-weight * log_prob).sum(dim=-1).mean()
#         return loss
#
def metric_average(val_tensor):
    avg_tensor = hvd.allreduce(val_tensor)
    return avg_tensor.item()

# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val, n=1):
        self.sum += float(hvd.allreduce(val.detach().cpu(), name=self.name))
        self.n += n

    @property
    def avg(self):
        return self.sum / self.n

def create_lr_schedule(workers, warmup_epochs, decay_schedule, alpha=0.1):
    def lr_schedule(epoch):
        lr_adj = 1.
        if epoch < warmup_epochs:
            lr_adj = 1. / workers * (epoch * (workers - 1) / warmup_epochs + 1)
        else:
            decay_schedule.sort(reverse=True)
            for e in decay_schedule:
                if epoch >= e:
                    lr_adj *= alpha
        return lr_adj
    return lr_schedule

def create_lr_scheduler(warmup_epochs, decay_schedule, alpha=0.1):
    def lr_schedule(epoch):
        lr_adj = 1.
        if epoch < warmup_epochs:
            lr_adj = 1. * (epoch+1)/warmup_epochs
        else:
            decay_schedule.sort(reverse=True)
            for e in decay_schedule:
                if epoch >= e:
                    lr_adj *= alpha
        return lr_adj
    return lr_schedule

#no warmup
# def create_lr_scheduler(warmup_epochs, decay_schedule, alpha=0.1):
#     def lr_schedule(epoch):
#         lr_adj = 1.
#         decay_schedule.sort(reverse=True)
#         for e in decay_schedule:
#             if epoch >= e:
#                 lr_adj *= alpha
#         return lr_adj
#     return lr_schedule


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             init.kaiming_normal(m.weight, mode='fan_out')
#             if m.bias:
#                 init.constant(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.constant(m.weight, 1)
#             init.constant(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal(m.weight, std=1e-3)
#             if m.bias:
#                 init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
