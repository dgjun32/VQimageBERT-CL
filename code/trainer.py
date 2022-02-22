import torch
import torch.nn as nn

class Trainer:
    def __init__(self, train_dataset, val_dataset, model, cfg):
        self.model = model
        self.cfg = cfg.train
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=cfg.batch_size)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=cfg.batch_size*2)

    def train(self):
        flag = True
        step_count = 0
        
        # set optimizer & lr scheduler
        optimizer = eval(self.cfg.optimizer)(self.model.parameters(),
                                             lr=self.cfg.lr, 
                                             betas=self.cfg.betas)
        lr_sched = eval(self.cfg.lr_sched)(optimizer,
                                           epochs=self.cfg.n_epochs,
                                           steps_per_epoch=1,
                                           anneal_strategy='cos',
                                           max_lr=self.cfg.lr,
                                           pct_start=0.02,
                                           final_div_factor=1e+7)
        # start training
        total_step = 0
        for epoch in range(self.cfg.n_epochs):
            print('-'*30 + '{}th epoch'.format(epoch) + '-'*30)
            epoch_loss, epoch_step = 0, 0
            for batch in self.train_dataloader:
                optimizer.zero_grad()
                # forward propagation
                loss = self.model(batch)
                # backward propagation
                loss.backward()
                optimizer.step()
                total_step += 1
                epoch_step += 1
                epoch_loss += loss.item()
                # vervosity
                if (epoch_step) % 5000 == 0:
                    print('| epoch_step / {} | NLLloss : {}'.format(len(self.train_dataloader), epoch_loss/epoch_step))
            lr_sched.step()
            self.evaluate(lr_sched)

    def evaluate(self, lr_sched):
        print('-'*60)
        with torch.no_grad():
            val_loss = 0
            for step, batch in enumerate(self.val_dataloader):
                # negative log likelihood
                loss = self.model(batch)
                val_loss += loss.item()
            print('| validation loss : {} | lr : {} |'.format(val_loss / step, lr_sched.get_lr[0]))

                




    