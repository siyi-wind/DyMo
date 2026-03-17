from typing import Tuple, List, Dict
from os.path import join, abspath, dirname
import torch
import torchmetrics
from utils.utils import AverageMeter
from omegaconf import DictConfig, open_dict, OmegaConf
import numpy as np

import wandb
from tqdm import tqdm
import time
import sys
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../'))
sys.path.append(project_path)
from models.utils.QMF_utils.utils import create_history

class DownstreamExperiment:
    def __init__(self, hparams, wandb_run, save_dir):
        print("Downstream Experiment")
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
        self.model_name = hparams.model_name
        self.save_dir = save_dir
        self.max_epochs = hparams.max_epochs
        self.dataset_name = hparams.dataset_name
        self.num_classes = hparams.num_classes
        self.wandb_run= wandb_run
        self.modality_names = hparams.modality_names

        self.model.to(self.device)
        self.criterion.to(self.device)
        self.eval_metric = hparams.eval_metric
        self.patience = 20
        self.min_delta = 0.0001
        self.gradient_accumulation_steps = hparams.gradient_accumulation_steps
        self.losses_list = []

        print(self.model)
        num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_trainable_params/1e6:.2f}M")
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {num_params/1e6:.2f}M")

    
    def set_model(self, args):
        module_path = None
        if args.dataset_name in set(['DVM', 'CAD', 'Infarction']):
            tmp_dataset_name = 'TIPData'
        else:
            tmp_dataset_name = args.dataset_name
            
        if args.model_name == 'DynamicTransformer':
            module_path = f"models.{tmp_dataset_name}.Dynamic.DynamicTransformer"
            MODEL = getattr(__import__(module_path, fromlist=['DynamicTransformer']), 'DynamicTransformer')
            self.model = MODEL(args)
        elif args.model_name == 'DyMo':
            module_path = f"models.{tmp_dataset_name}.Dynamic.DyMo"
            MODEL = getattr(__import__(module_path, fromlist=['DyMo']), 'DyMo')
            self.model = MODEL(args)
        
        #### Comparison Models ####
        elif args.model_name == 'MUSE':
            module_path = f"models.{tmp_dataset_name}.Comparison.MUSE"
            MODEL = getattr(__import__(module_path, fromlist=['MUSE']), 'MUSE')
            self.model = MODEL(args)
        elif args.model_name == 'MTL':
            module_path = f"models.{tmp_dataset_name}.Comparison.MTL"
            MODEL = getattr(__import__(module_path, fromlist=['MTL']), 'MTL')
            self.model = MODEL(args)
        elif args.model_name == 'M3Care':
            module_path = f"models.{tmp_dataset_name}.Comparison.M3Care"
            MODEL = getattr(__import__(module_path, fromlist=['M3Care']), 'M3Care')
            self.model = MODEL(args)
        elif args.model_name == 'MAP':
            module_path = f"models.{tmp_dataset_name}.Comparison.MAP"
            MODEL = getattr(__import__(module_path, fromlist=['MAP']), 'MAP')
            self.model = MODEL(args)
        elif args.model_name == 'ModDrop':
            module_path = f"models.{tmp_dataset_name}.Comparison.ModDrop"
            MODEL = getattr(__import__(module_path, fromlist=['ModDrop']), 'ModDrop')
            self.model = MODEL(args)
        elif args.model_name == 'MultiAE':
            module_path = f"models.{tmp_dataset_name}.Comparison.MultimodalAutoEncoder"
            MODEL = getattr(__import__(module_path, fromlist=['MultiAE']), 'MultiAE')
            self.model = MODEL(args) 
        elif args.model_name == 'MoPoE_Classification':
            module_path = f"models.{tmp_dataset_name}.Comparison.MoPoE_Classification"
            MODEL = getattr(__import__(module_path, fromlist=['MoPoE_Classification']), 'MoPoE_Classification')
            self.model = MODEL(args)
        elif args.model_name == 'CMVAE_Classification':
            module_path = f"models.{tmp_dataset_name}.Comparison.CMVAE_Classification"
            MODEL = getattr(__import__(module_path, fromlist=['CMVAE_Classification']), 'CMVAE_Classification')
            self.model = MODEL(args)
        elif args.model_name == 'OnlineMAE':
            module_path = f"models.{tmp_dataset_name}.Comparison.OnlineMAE"
            MODEL = getattr(__import__(module_path, fromlist=['OnlineMAE']), 'OnlineMAE')
            self.model = MODEL(args)
        elif args.model_name == 'PDF':
            module_path = f"models.{tmp_dataset_name}.Comparison.PDF"
            MODEL = getattr(__import__(module_path, fromlist=['PDF']), 'PDF')
            self.model = MODEL(args)
        elif args.model_name == 'QMF':
            module_path = f"models.{tmp_dataset_name}.Comparison.QMF"
            MODEL = getattr(__import__(module_path, fromlist=['QMF']), 'QMF')
            self.model = MODEL(args)
        elif args.model_name == 'Unimodal':
            module_path = f"models.{tmp_dataset_name}.Comparison.Unimodal"
            MODEL = getattr(__import__(module_path, fromlist=['Unimodal']), 'Unimodal')
            self.model = MODEL(args)
        elif args.model_name == 'Multimodal_CONCAT':
            module_path = f"models.{tmp_dataset_name}.Comparison.Multimodal_CONCAT"
            MODEL = getattr(__import__(module_path, fromlist=['Multimodal_CONCAT']), 'Multimodal_CONCAT')
            self.model = MODEL(args)
        elif args.model_name == 'DynMM':
            module_path = f"models.{tmp_dataset_name}.Comparison.DynMM"
            MODEL = getattr(__import__(module_path, fromlist=['DynMM']), 'DynMM')
            self.model = MODEL(args)
        

    def set_criterion(self, args):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_vis = torch.nn.CrossEntropyLoss(reduction='none') # For visualization per-sample loss
    

    def set_optimizer_scheduler(self, args):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr_downstream, weight_decay=args.weight_decay_downstream)
        self.scheduler = None
    
    
    def set_metric(self, args):
        task = 'binary' if args.num_classes == 2 else 'multiclass'
        num_classes = args.num_classes
        self.acc_train = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(self.device)
        self.auc_train = torchmetrics.AUROC(task=task, num_classes=num_classes).to(self.device)
        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(self.device)
        self.auc_val = torchmetrics.AUROC(task=task, num_classes=num_classes).to(self.device)
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=num_classes).to(self.device)
        self.auc_test = torchmetrics.AUROC(task=task, num_classes=num_classes).to(self.device)
    

    def train(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):
        ### QMF TODO
        if self.model_name == 'QMF':
            self.m_history = create_history(self.modality_names, len(train_loader.dataset))
            self.global_step = 0
            self.optimizer.zero_grad()
            print('QMF uses accumulated gradient steps:', self.gradient_accumulation_steps)

        counter = 0
        for epoch in range(self.start_epoch, self.max_epochs):
            self.train_epoch(train_loader, epoch)
            val_results = self.validate(val_loader, epoch)
            cur_val_score = val_results['val.'+self.eval_metric]
            if cur_val_score > self.best_val_score:
                improve = cur_val_score - self.best_val_score
                self.best_val_score = cur_val_score
                self.save_checkpoint(epoch, f'checkpoint_best_{self.eval_metric}')
                print(f"Best model saved at epoch {epoch}")
            else:
                improve = 0
            
            # early stopping
            if epoch > self.start_epoch and improve < self.min_delta:
                counter += 1
                if counter >= self.patience:
                    print("Early stopping")
                    break
            else:
                counter = 0

            # save model every 10 epochs
            if self.hparams.break_avoidance and epoch % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_last_epoch_{epoch}')
                print(f"Checkpoint saved for epoch {epoch}")

    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        device = self.device
        losses = AverageMeter('Downstream Train Loss', ':.4f')

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
            # TODO
            if self.model_name == 'QMF':
                pass
            else:
                self.optimizer.zero_grad()

            if self.model_name in set(['MissTransformer', 'MUSE', 'MTL', 'M3Care', 
                                       'MAP', 'ModDrop', 'MultiAE', 'MoPoE_Classification', 'CMVAE_Classification',
                                       'OnlineMAE', 'PDF', 'DynMM']):
                y_hat = self.model.forward_train(x, mask, y)
            elif self.model_name == 'QMF':
                y_hat = self.model.forward_train(x, mask, y, self.m_history)
            else:
                y_hat = self.model(x, mask)

            if self.model_name in set(['MUSE', 'MTL', 'M3Care', 'MAP',
                                       'MultiAE', 'MoPoE_Classification', 'CMVAE_Classification',
                                       'OnlineMAE', 'PDF', 'QMF', 'DynMM']):
                loss, y_hat = y_hat
            else:
                loss = self.criterion(y_hat, y)
            
            if self.model_name == 'QMF':
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                self.global_step += 1
                if self.global_step % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                self.optimizer.step()
            losses.update(loss.item(), y.size(0))
            # accuracy and auc calculation
            with torch.no_grad():
                y_hat = torch.softmax(y_hat.detach(), dim=1)
                if self.num_classes == 2:
                    y_hat = y_hat[:, 1]
                self.acc_train(y_hat, y)
                self.auc_train(y_hat, y)
            if device == 'cuda:0' or device == 'cpu':
                pbar.update(1)

        if self.scheduler is not None:
            self.scheduler.step()
        if device == 'cuda:0' or device == 'cpu':
            pbar.close()
        end_time = time.time()

        # record
        epoch_acc = self.acc_train.compute().item()
        epoch_auc = self.auc_train.compute().item()
        self.wandb_run.log({"epoch": epoch, 'downstream_train/lr': self.optimizer.param_groups[0]['lr']})
        self.wandb_run.log({"epoch": epoch, 'downstream_train/loss': losses.avg})
        self.wandb_run.log({"epoch": epoch, 'downstream_train/acc': epoch_acc, 'downstream_train/auc': epoch_auc})
        self.acc_train.reset()
        self.auc_train.reset()
        time_elapsed = end_time - start_time
        print(f"Epoch {epoch}: Train Loss: {losses.avg:.4f}, Train Acc: {epoch_acc:.4f}, Train AUC: {epoch_auc:.4f}, Time: {(time_elapsed%3600)/60:.2f}m")
        return {'train.acc':epoch_acc, 'train.auc':epoch_auc, 'train.loss':losses.avg}
        

    def validate(self, val_loader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        device = self.device
        losses = AverageMeter('Downstream Val Loss')
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
                y_hat = self.model(x, mask)
                loss = self.criterion(y_hat, y)
                losses.update(loss.item(), y.size(0))
                y_hat = torch.softmax(y_hat.detach(), dim=1)
                if self.num_classes == 2:
                    y_hat = y_hat[:, 1]
                self.acc_val(y_hat, y)
                self.auc_val(y_hat, y)
                if device == 'cuda:0' or device == 'cpu':
                    pbar.update(1)
        
        if device == 'cuda:0' or device == 'cpu':
            pbar.close()
        end_time = time.time()

        # record
        epoch_acc = self.acc_val.compute().item()
        epoch_auc = self.auc_val.compute().item()
        self.wandb_run.log({"epoch": epoch, 'downstream_val/loss': losses.avg})
        self.wandb_run.log({"epoch": epoch, 'downstream_val/acc': epoch_acc, 'downstream_val/auc': epoch_auc})
        self.acc_val.reset()
        self.auc_val.reset()
        time_elapsed = end_time - start_time
        print(f"Epoch {epoch}: Val Loss: {losses.avg:.4f}, Val Acc: {epoch_acc:.4f}, Val AUC: {epoch_auc:.4f}, Time: {(time_elapsed%3600)/60:.2f}m")
        return {'val.acc':epoch_acc, 'val.auc':epoch_auc, 'val.loss':losses.avg}


    def test(self, test_loader: torch.utils.data.DataLoader, ckpt_path: str) -> Dict[str, float]:
        device = self.device
        if self.hparams.model_name == 'DyMo':
            print('Test: Have already loaded all the checkpoints')
        else:
            print('Test: Load checkpoint from', ckpt_path)
            test_ckpt = torch.load(ckpt_path, map_location='cpu')
            load_info = self.model.load_state_dict(test_ckpt['state_dict'], strict=False)
            if self.hparams.model_name in set(['QMF', 'PDF', 'DynMM', 'Multimodal_CONCAT']):
                print(f" missing keys: {load_info.missing_keys}, unexpected keys: {load_info.unexpected_keys}")
            else:
                assert len(load_info.missing_keys) == 0, f"Missing keys in the checkpoint: {load_info.missing_keys}"
        self.model.eval()
        losses = AverageMeter('Downstream Test Loss')
        if device == 'cuda:0' or device == 'cpu':
            pbar = tqdm(total=len(test_loader))
        start_time = time.time()
        with torch.no_grad():
            for batch in test_loader:
                x, mask, y = batch
                for key in x:
                    x[key] = x[key].to(device)
                y = y.to(device)
                mask = mask.to(device)
                if self.hparams.model_name == 'DyMo':
                    y_hat = self.model(x, mask, y)
                else:
                    y_hat = self.model(x, mask)
                loss = self.criterion(y_hat, y)
                losses.update(loss.item(), y.size(0))

                per_sample_losses = self.criterion_vis(y_hat, y)
                self.losses_list.extend(per_sample_losses.detach().cpu().tolist())

                y_hat = torch.softmax(y_hat.detach(), dim=1)
                if self.num_classes == 2:
                    y_hat = y_hat[:, 1]
                self.acc_test(y_hat, y)
                self.auc_test(y_hat, y)
                if device == 'cuda:0' or device == 'cpu':
                    pbar.update(1)

        if device == 'cuda:0' or device == 'cpu':
            pbar.close()
        end_time = time.time()
        # record
        epoch_acc = self.acc_test.compute().item()
        epoch_auc = self.auc_test.compute().item()
        self.wandb_run.log({'downstream_test/loss': losses.avg})
        self.wandb_run.log({'downstream_test/acc': epoch_acc, 'downstream_test/auc': epoch_auc})
        self.acc_test.reset()
        self.auc_test.reset()
        time_elapsed = end_time - start_time
        print(f"Test Loss: {losses.avg:.4f}, Test Acc: {epoch_acc:.4f}, Test AUC: {epoch_auc:.4f}, Time: {(time_elapsed%3600)/60:.2f}m")
        # Save per-sample losses with numpy
        np.save(join(self.save_dir, 'test_per_sample_losses.npy'), np.array(self.losses_list))

        return {'test.acc': epoch_acc, 'test.auc': epoch_auc, 'test.loss':losses.avg}


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
        
        
