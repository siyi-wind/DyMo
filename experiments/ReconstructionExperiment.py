from typing import Tuple, List, Dict
from os.path import join
import torch
import torchmetrics
from utils.utils import AverageMeter
from omegaconf import DictConfig, open_dict, OmegaConf

import wandb
from tqdm import tqdm
import time

class ReconstructionExperiment:
    def __init__(self, hparams, wandb_run, save_dir):
        print('Pretraining Reconstruction Experiment')
        self.device =  f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        if hparams.checkpoint and hparams.resume_training:
            original_hparams = self.load_checkpoint(hparams.checkpoint)
            hparams = original_hparams
        else:
            self.set_model(hparams)
            self.set_criterion(hparams)
            self.set_optimizer_scheduler(hparams)
            self.set_metric(hparams)
            self.start_epoch = 0
            self.best_val_score = 0

        self.hparams = hparams
        self.save_dir = save_dir
        self.max_epochs = hparams.max_epochs_pretrain
        self.dataset_name = hparams.dataset_name
        self.num_classes = hparams.num_classes
        self.wandb_run= wandb_run
        self.model_name = hparams.model_name

        self.model.to(self.device)
        self.criterion.to(self.device)
        self.eval_metric = hparams.eval_metric
        self.patience = 20
        self.min_delta = 0.0001

        print(self.model)
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_trainable_params/1e6:.2f}M")
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_params/1e6:.2f}M")

    
    def set_model(self, args):
        if args.model_name == 'MultiAE':
            module_path = f"models.{args.dataset_name}.Comparison.MultimodalAutoEncoder"
            MODEL = getattr(__import__(module_path, fromlist=['MultiAE']), 'MultiAE')
            self.model = MODEL(args)

        print(module_path)

    def set_criterion(self, args):
        self.criterion = torch.nn.MSELoss()
    

    def set_optimizer_scheduler(self, args):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr_pretrain, weight_decay=args.weight_decay_pretrain)
        self.scheduler = None
    
    
    def set_metric(self, args):
        task = 'binary' if args.num_classes == 2 else 'multiclass'
        num_classes = args.num_classes
    

    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):
        for epoch in range(self.start_epoch, self.max_epochs):
            self.train_epoch(train_loader, epoch)
            val_results = self.validate(val_loader, epoch)

            # save model every 10 epochs
            if self.hparams.break_avoidance and epoch % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_last_epoch_{epoch}')
                print(f"Checkpoint saved for epoch {epoch}")
        
        # save the last epoch model
        self.save_checkpoint(epoch, f'checkpoint_last_epoch_{epoch}')

    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        device = self.device
        losses = AverageMeter('Pretrain Train Loss', ':.4f')
        if device == 'cuda:0' or device == 'cpu':
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
        
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            x, mask, y = batch
            # x is a dictionary
            for key in x:
                x[key] = x[key].to(device)
            y = y.to(device)
            mask = mask.to(device)
            self.optimizer.zero_grad()

            loss, x_recon = self.model.forward_recon(x, mask)

            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), y.size(0))
            if device == 'cuda:0' or device == 'cpu':
                pbar.update(1)

        if self.scheduler is not None:
            self.scheduler.step()
        if device == 'cuda:0' or device == 'cpu':
            pbar.close()
        end_time = time.time()

        # record
        self.wandb_run.log({"epoch": epoch, 'pretrain_train/lr': self.optimizer.param_groups[0]['lr']})
        self.wandb_run.log({"epoch": epoch, 'pretrain_train/loss': losses.avg})
        time_elapsed = end_time - start_time
        print(f"Epoch {epoch}: Train Loss: {losses.avg:.4f}, Time: {(time_elapsed%3600)/60:.2f}m")
        return {'train.loss':losses.avg}
        

    def validate(self, val_loader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        device = self.device
        losses = AverageMeter('Pretrain Val Loss')
        device = self.device
        if device == 'cuda:0' or device == 'cpu':
            pbar = tqdm(total=len(val_loader), desc=f"Epoch {epoch}")

        start_time = time.time()
        with torch.no_grad():
            for batch in val_loader:
                x, mask, y = batch
                for key in x:
                    x[key] = x[key].to(device)
                y = y.to(device)
                mask = mask.to(device)
                
                loss, x_recon = self.model.forward_recon(x, mask)

                losses.update(loss.item(), y.size(0))
                if device == 'cuda:0' or device == 'cpu':
                    pbar.update(1)
        
        if device == 'cuda:0' or device == 'cpu':
            pbar.close()
        end_time = time.time()

        # record
        self.wandb_run.log({"epoch": epoch, 'pretrain_val/loss': losses.avg})
        time_elapsed = end_time - start_time
        print(f"Epoch {epoch}: Val Loss: {losses.avg:.4f}, Time: {(time_elapsed%3600)/60:.2f}m")
        return {'val.loss':losses.avg}


    def test(self, test_loader: torch.utils.data.DataLoader, ckpt_path: str) -> Dict[str, float]:
        pass


    def save_checkpoint(self, epoch: int, name: str):
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'hyper_parameters': self.hparams,
            'best_val_score': self.best_val_score
        }
        torch.save(checkpoint, join(self.save_dir, name+'.ckpt'))
    

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        original_hparams = OmegaConf.create(checkpoint['hyper_parameters'])
        self.set_model(original_hparams)
        self.set_criterion(original_hparams)
        self.set_optimizer_scheduler(original_hparams)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        # resume training
        self.start_epoch = checkpoint['epoch']
        self.best_val_score = checkpoint['best_val_score']
        print(f"Checkpoint loaded from {checkpoint_path}")
        return original_hparams
        
        
