'''
DownstreamAugmentationExperiment.py
For DynamicTransformer. During training, randomly sample K kinds of subsets of modalities.
'''
from typing import Tuple, List, Dict
from os.path import join, abspath, dirname
import torch
import torchmetrics
import torch.nn.functional as F
from utils.utils import AverageMeter
from omegaconf import DictConfig, open_dict, OmegaConf

import wandb
from tqdm import tqdm
import time
import sys
current_path = abspath(__file__)
project_path = abspath(join(current_path, '../../'))
sys.path.append(project_path)
from models.utils.prototype_loss import PrototypeLoss

class DownstreamAugExperiment:
    def __init__(self, hparams, wandb_run, save_dir):
        print("Downstream Augmentation Experiment with Incomplete Modality Simulation")
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

        self.model.to(self.device)
        self.criterion.to(self.device)
        self.criterion_pt.to(self.device)
        self.eval_metric = hparams.eval_metric
        self.patience = 20
        self.min_delta = 0.0001
        self.K = hparams[hparams.dataset_name].augmentation_K
        self.pt_rate = hparams[hparams.dataset_name].pt_rate
        self.pt_epoch = 3
        print(f'Sample {self.K} kinds of subsets of modalities for augmentation, pt_rate: {self.pt_rate}, pt_epoch: {self.pt_epoch}')

        # prototypes
        self.projection_dim = self.hparams[self.dataset_name].projection_dim

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
        if args.model_name == 'Unimodal':
            if args.dataset_name == 'PolyMNIST':
                from models.PolyMNIST.UnimodalModel import UnimodalModel
                self.model = UnimodalModel(args)
        elif args.model_name == 'DynamicTransformer':
            module_path = f"models.{tmp_dataset_name}.Dynamic.DynamicTransformer"
            MODEL = getattr(__import__(module_path, fromlist=['DynamicTransformer']), 'DynamicTransformer')
            self.model = MODEL(args)
        print(module_path)

    
    def set_criterion(self, args):
        self.criterion = torch.nn.CrossEntropyLoss()
        temperature = args[args.dataset_name].temperature if 'temperature' in args[args.dataset_name] else 0.1
        print(f'Prototype Loss Temperature: {temperature}')
        self.criterion_pt = PrototypeLoss(temperature=temperature, metric_name=args[args.dataset_name].distance_metric)
    

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


    def cal_prototypes(self, label, feat):
        label = label.float()
        class_sum = label.t() @ feat
        class_count = torch.sum(label, dim=0, keepdim=True).t()
        return class_sum, class_count
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        device = self.device
        prototypes= self.model.prototypes.clone().detach()
        num_subsets = len(self.model.subset2id)
        prototypes_sum = torch.zeros(self.num_classes, self.projection_dim).to(device)
        prototypes_count_sum = torch.zeros(self.num_classes, 1).to(device)
        
        id2subset = self.model.id2subset
        subset2id = self.model.subset2id
        losses = AverageMeter('Downstream Train Loss', ':.4f')
        ce_losses = AverageMeter('CE Loss', ':.4f')
        pt_losses = AverageMeter('PT Loss', ':.4f')
        if device == 'cuda:0' or device == 'cpu':
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
        
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            x, origin_mask, y = batch
            # x is a dictionary
            for key in x:
                x[key] = x[key].to(device)
            y = y.to(device)

            self.optimizer.zero_grad()
            # augmentation to generate K kinds of mask
            if self.K is None or self.K == False:
                sample_idx = [subset2id[tuple(origin_mask[0].tolist())]]
            else:
                sample_idx = torch.randperm(len(id2subset), device=device)[:self.K]
            loss_ce_list = []
            loss_pt_list = []
            y_hat_list = []
            y_list = []
            for id in sample_idx:
                mask = torch.tensor(id2subset[int(id)], device=device)
                mask = mask.unsqueeze(0).expand(len(y), -1)

                # forward pass
                y_hat, feat = self.model.forward_train(x, mask)
                # record loss
                loss_ce_list.append(self.criterion(y_hat, y))
                label = F.one_hot(y, num_classes=self.num_classes).to(device)
                # prototypes = prototypes
                loss_pt_list.append(self.criterion_pt(label, prototypes, feat))
                # update prototypes_sum and prototypes_count_sum
                with torch.no_grad():
                    class_sum, class_count = self.cal_prototypes(label.detach(), feat.detach())
                    prototypes_sum = prototypes_sum + class_sum.detach()
                    prototypes_count_sum = prototypes_count_sum + class_count.detach()
                    y_hat_list.append(y_hat)
                    y_list.append(y)

            loss_pt = sum(loss_pt_list) / len(loss_pt_list)
            loss_ce = sum(loss_ce_list) / len(loss_ce_list)
            # start using pt after 5 epochs
            if epoch >= self.pt_epoch:
                loss = loss_ce + loss_pt * self.pt_rate
            else:
                loss = loss_ce

            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), y.size(0))
            ce_losses.update(loss_ce.item(), y.size(0))
            pt_losses.update(loss_pt.item(), y.size(0))

            with torch.no_grad():
                y_hat = torch.cat(y_hat_list, dim=0)
                y = torch.cat(y_list, dim=0)
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
        self.wandb_run.log({"epoch": epoch, 'downstream_train/ce_loss': ce_losses.avg})
        self.wandb_run.log({"epoch": epoch, 'downstream_train/pt_loss': pt_losses.avg})
        self.wandb_run.log({"epoch": epoch, 'downstream_train/acc': epoch_acc, 'downstream_train/auc': epoch_auc})
        self.acc_train.reset()
        self.auc_train.reset()

        # update prototypes
        zero_count = torch.where(prototypes_count_sum < 1)[0]
        assert len(zero_count) == 0, f"Prototypes count is zero for {zero_count} classes"
        self.model.prototypes.data.copy_(prototypes_sum / prototypes_count_sum)

        time_elapsed = end_time - start_time
        print(f"Epoch {epoch}: Train Loss: {losses.avg:.4f}, Train Acc: {epoch_acc:.4f}, Train AUC: {epoch_auc:.4f}, Time: {(time_elapsed%3600)/60:.2f}m")
        return {'train.acc':epoch_acc, 'train.auc':epoch_auc, 'train.loss':losses.avg, 'class_sum':class_sum, 'class_count':class_count}
        

    def validate(self, val_loader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        device = self.device
        losses = AverageMeter('Downstream Val Loss')
        losses_ce = AverageMeter('CE Loss')
        losses_pt = AverageMeter('PT Loss')
        subset_masks = list(self.model.subset2id.keys())
        K = len(subset_masks)
        prototypes = self.model.prototypes.clone().detach()
        device = self.device
        if device == 'cuda:0' or device == 'cpu':
            pbar = tqdm(total=len(val_loader), desc=f"Epoch {epoch}")

        start_time = time.time()
        with torch.no_grad():
            for batch in val_loader:
                x, origin_mask, y = batch
                for key in x:
                    x[key] = x[key].to(device)
                y = y.to(device)
                # augmentation to generate K kinds of mask
                if self.K is None or self.K == False:
                    sample_idx = [self.model.subset2id[tuple(origin_mask[0].tolist())]]
                else:
                    sample_idx = list(range(K))
                loss_ce_list = []
                loss_pt_list = []
                y_hat_list = []
                y_list = []
                for id in sample_idx:
                    mask = torch.tensor(subset_masks[id], device=device)
                    mask = mask.unsqueeze(0).expand(len(y), -1)
                    # forward pass
                    y_hat, feat = self.model.forward_train(x, mask)
                    # record loss
                    loss_ce_list.append(self.criterion(y_hat, y))
                    label = F.one_hot(y, num_classes=self.num_classes).to(device)
                    # prototypes = prototypes_all[id]
                    loss_pt_list.append(self.criterion_pt(label, prototypes, feat))
                    y_hat_list.append(y_hat)
                    y_list.append(y)

                # loss
                loss_pt = sum(loss_pt_list) / len(loss_pt_list)
                loss_ce = sum(loss_ce_list) / len(loss_ce_list)
                if epoch >= self.pt_epoch:
                    loss = loss_ce + loss_pt * self.pt_rate
                else:
                    loss = loss_ce
                losses.update(loss.item(), y.size(0))
                losses_ce.update(loss_ce.item(), y.size(0))
                losses_pt.update(loss_pt.item(), y.size(0))

                # metric
                y_hat = torch.cat(y_hat_list, dim=0)
                y = torch.cat(y_list, dim=0)
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
        self.wandb_run.log({"epoch": epoch, 'downstream_val/ce_loss': losses_ce.avg})
        self.wandb_run.log({"epoch": epoch, 'downstream_val/pt_loss': losses_pt.avg})
        self.wandb_run.log({"epoch": epoch, 'downstream_val/acc': epoch_acc, 'downstream_val/auc': epoch_auc})
        self.acc_val.reset()
        self.auc_val.reset()
        time_elapsed = end_time - start_time
        print(f"Epoch {epoch}: Val Loss: {losses.avg:.4f}, Val Acc: {epoch_acc:.4f}, Val AUC: {epoch_auc:.4f}, Time: {(time_elapsed%3600)/60:.2f}m")
        return {'val.acc':epoch_acc, 'val.auc':epoch_auc, 'val.loss':losses.avg}


    def test(self, test_loader: torch.utils.data.DataLoader, ckpt_path: str) -> Dict[str, float]:
        device = self.device
        if self.hparams.model_name == 'TwoStageTransformer':
            print('Test: Have already loaded all the checkpoints')
        else:
            print('Test: Load checkpoint from', ckpt_path)
            test_ckpt = torch.load(ckpt_path, map_location='cpu')
            self.model.load_state_dict(test_ckpt['state_dict'])
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
                y_hat = self.model(x, mask)
                loss = self.criterion(y_hat, y)
                losses.update(loss.item(), y.size(0))
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
        
        
