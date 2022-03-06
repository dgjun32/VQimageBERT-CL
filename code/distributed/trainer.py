import torch
import torch.nn as nn

class Trainer:
    def __init__(self, train_dataset, val_dataset, model, cfg):
        self.model = model
        self.cfg = cfg.train
        self.train_dataset = train_dataset
        self.train_dataloader = iter(torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=self.cfg.batch_size,
                                                    shuffle = True)
                                                    )
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=self.cfg.batch_size*2)

    def train(self, start_step=0):
        if start_step != 0:
            state_dict = torch.load('../model/vqimagebert_{}epoch'.format(start_step-1))
        
        # set optimizer & lr scheduler
        optimizer = eval(self.cfg.optimizer)(self.model.parameters(),
                                            lr=self.cfg.lr, 
                                            betas=self.cfg.betas)
        lr_sched = eval(self.cfg.lr_sched)(optimizer,
                                        total_steps = self.cfg.total_steps,
                                        anneal_strategy='cos',
                                        max_lr=self.cfg.lr,
                                        pct_start=0.01,
                                        final_div_factor=1e+7)
        
        if start_step > 2000:
            optimizer.load_state_dict(state_dict['optim'])
            lr_sched.load_state_dict(state_dict['lr_sched'])
        
        # set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        
        # start training
        if start_step > 2000:
            self.model.load_state_dict(state_dict['model'])
            temp_step = start_step
            temp_mim_loss = state_dict['mim_loss']
            temp_nce_loss = state_dict['nce_loss']
        else:
            temp_step = 0
            temp_mim_loss = 0
            temp_nce_loss = 0
        while temp_step < self.cfg.total_steps:
            try:
                batch = next(self.train_dataloader)
                optimizer.zero_grad()
                # forward propagation
                mim_loss, nce_loss = self.model(batch.to(device))
                loss = mim_loss + self.cfg.gamma*nce_loss
                # backward propagation
                loss.backward()
                optimizer.step()
                # update verbosity
                temp_step += 1
                temp_mim_loss += mim_loss.item()
                temp_nce_loss += nce_loss.item()
                if (temp_step) % 20 == 0:
                    print("| {} / {} | loss : '{:.3f}' | MIMLoss : '{:.3f}' | NCELoss : '{:.3f}' |".format(
                                                                            temp_step,
                                                                            self.cfg.total_steps,
                                                                            loss.item(),
                                                                            temp_mim_loss/temp_step,
                                                                            temp_nce_loss/temp_step
                                                                            ))
                lr_sched.step()
                if temp_step % 2000 == 0:
                    torch.save({
                        'step' : temp_step,
                        'mim_loss': temp_mim_loss,
                        'nce_loss': temp_nce_loss,
                        'model' : self.model.state_dict(),
                        'optim' : optimizer.state_dict(),
                        'lr_sched' : lr_sched.state_dict
                        }
                        , '../model/vqimagebert_{}step.pt'.format(temp_step))
            except StopIteration: 
                self.train_dataloader = iter(torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_size=self.cfg.batch_size,
                                                    shuffle=True)
                                                    )

    def evaluate(self, lr_sched):
        print('-'*60)
        with torch.no_grad():
            val_loss = 0
            for step, batch in enumerate(self.val_dataloader):
                # negative log likelihood
                loss = self.model(batch)
                val_loss += loss.item()
            print('| validation loss : {} | lr : {} |'.format(val_loss / step, lr_sched.get_lr[0]))
