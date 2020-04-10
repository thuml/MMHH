class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, iter_num):
        lr = self.init_lr * (1 + self.gamma * iter_num) ** (-self.decay_rate)
        for param_group, group_ratio in zip(optimizer.param_groups, group_ratios):
            param_group['lr'] = lr * group_ratio
        return optimizer


class StepScheduler(object):
    def __init__(self, gamma=0.5, step=2000, init_lr=0.0003):
        self.step = step
        self.gamma = gamma
        self.init_lr = init_lr
        self.last_step = -1

    def next_optimizer(self, group_ratios, optimizer, iter_num, logger=None):
        lr = self.init_lr * (self.gamma ** (iter_num // self.step))
        if logger and lr != self.last_step:
            self.last_step = lr
        for param_group, group_ratio in zip(optimizer.param_groups, group_ratios):
            param_group['lr'] = lr * group_ratio
        return optimizer
