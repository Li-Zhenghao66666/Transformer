import os
import torch

class Trainer(object):
    def __init__(self, model, optimizer, criterion, cfg, logger,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.logger = logger
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        self.save_dir = cfg['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        self.start_epoch = 1
        self.epochs = cfg['epochs']
        self.save_epochs = cfg['save_epochs']
        self.do_validation = valid_data_loader is not None
        self.log_step = cfg['log_step']
        self.monitor = 'min'     # monitor val_loss
        self.early_stop = cfg['early_stop']
        self.patience = cfg['patience']
        self.best = float('inf')
        self.counter = 0

        # 记录曲线
        self.history = {'train_loss': [], 'val_loss': [], 'bleu': [], 'acc': [], 'ppl': []}

        if cfg.get('resume', False) and os.path.isfile(cfg.get('resume_path', '')):
            self._resume_checkpoint(cfg['resume_path'])

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            val_loss, metrics = self._train_epoch(epoch)
            # 记录历史
            if isinstance(metrics, dict):
                for k in self.history:
                    if k in metrics and metrics[k] is not None:
                        self.history[k].append(float(metrics[k]))
            # 早停 & 最优
            score = -val_loss
            is_best = val_loss < self.best
            if is_best:
                self.best = val_loss
                self.counter = 0
                self._save_checkpoint(epoch, save_best=True)
            else:
                self.counter += 1

            if epoch % max(1, self.save_epochs) == 0:
                self._save_checkpoint(epoch, save_best=False)

            if self.early_stop and self.counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        # 训练结束绘图与落盘
        if hasattr(self, '_plot_and_dump'):
            try:
                self._plot_and_dump()
            except Exception as e:
                self.logger.debug(f'Plot/dump failed: {e}')

    # 子类需要实现
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.best,
            'config': self.cfg
        }
        if save_best:
            filename = os.path.join(self.save_dir, 'model_best.pt')
            torch.save(state, filename)
            self.logger.info(f"Saving current best: {filename}")
        else:
            filename = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(state, filename)
            self.logger.info(f"Saving checkpoint: {filename}")

    def _resume_checkpoint(self, resume_path):
        self.logger.info(f"Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        self.start_epoch = checkpoint['epoch'] + 1
        self.best = checkpoint.get('monitor_best', float('inf'))
        self.model.load_state_dict(checkpoint['state_dict'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
