import torch

from common.mmhh_config import BatchType, SemiInitType


class SemiBatch(object):
    def __init__(self, total_num, D_out, total_iter_num, init_momentum=0.5, semi_init_type=SemiInitType.MODEL,
                 model=None, loader=None, use_gpu=False):
        # init hash codes
        self.total_iter_num = total_iter_num
        self.init_momentum = init_momentum
        self.momentum = 0
        if semi_init_type == SemiInitType.RANDOM:
            self.aug_memory = torch.randn(total_num, D_out, requires_grad=False)
        elif semi_init_type == SemiInitType.MODEL:
            self.aug_memory = torch.zeros((total_num, D_out), dtype=torch.float, requires_grad=False)
        else:
            raise NotImplementedError
        # init labels
        self.labels = None
        if use_gpu:
            self.aug_memory = self.aug_memory.cuda()
        total_num = len(loader.dataset)
        calc_num = 0
        for inputs_batch, labels_batch, indices_batch in loader:
            if use_gpu:
                inputs_batch, labels_batch = inputs_batch.cuda(), labels_batch.cuda()
            if self.labels is None:
                self.labels = torch.zeros((total_num, labels_batch.shape[1]), dtype=torch.uint8, requires_grad=False)
                if use_gpu:
                    self.labels = self.labels.cuda()
            self.labels[indices_batch] = labels_batch.data
            if semi_init_type == "model":
                x_out = model.predict(inputs_batch)
                self.aug_memory[indices_batch] = x_out.data
            calc_num += len(indices_batch)
        if calc_num != total_num:
            raise Exception("predict num %d != total %d" % (calc_num, total_num))

    def update_memory(self, x_out: torch.Tensor, bidxs, norm_memory_batch=False):
        # update the non-parametric data
        m_x = self.aug_memory[bidxs]
        weighted_m_x = m_x * self.momentum + x_out.data * (1. - self.momentum)
        if norm_memory_batch:
            w_norm = (weighted_m_x ** 2).sum(axis=1, keepdim=True).pow(0.5)
            weighted_m_x = weighted_m_x.div(w_norm)
        self.aug_memory[bidxs] = weighted_m_x

    def update_momentum(self, iter_num, batch_type, batch_params) -> None:
        if batch_type == BatchType.BatchInitMem and iter_num < int(batch_params):
            self.momentum = 0
        else:
            self.momentum = self.init_momentum + (1 - self.init_momentum) * (
                        float(iter_num) / self.total_iter_num) ** 0.5
