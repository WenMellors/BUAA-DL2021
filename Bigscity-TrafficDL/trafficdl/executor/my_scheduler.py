import warnings

from torch.optim.lr_scheduler import _LRScheduler


class MyScheduler(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super(MyScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        epoch = self._step_count
        return [min(epoch * (4000 ** -1.5), 1 / (epoch ** 0.5))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        epoch = self._step_count
        return [min(epoch * (4000 ** -1.5), 1 / (epoch ** 0.5))
                for base_lr in self.base_lrs]
