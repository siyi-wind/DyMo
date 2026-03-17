import os 
from os.path import join
import sys
import time
from datetime import datetime
import random
import numpy as np

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import wandb

from trainers.pretrain import pretrain
from trainers.downstream import downstream
from trainers.test import test
from utils.utils import create_logdir, prepend_paths, re_prepend_paths, fill_specific_params


torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
hydra.HYDRA_FULL_ERROR = 1

def run(args: DictConfig):
    now = datetime.now()
    start = time.time()
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.data_base = join(args.data_base, args.data_base_postfix)

    # fill model and dataset specific params
    args = fill_specific_params(args)

    args = prepend_paths(args)
    time.sleep(random.randint(1,5)) # Prevents multiple runs getting the same version when launching many jobs at once
    

    if args.resume_training:
        if args.wandb_id:
            wandb_id = args.wandb_id    
        tmp_data_base = args.data_base
        checkpoint = args.checkpoint
        ckpt = torch.load(args.checkpoint)
        args = ckpt['hyper_parameters']
        args = OmegaConf.create(args)
        #with open_dict(args):
        args.checkpoint = checkpoint
        args.resume_training = True
        if not 'wandb_id' in args or not args.wandb_id:
            args.wandb_id = wandb_id
        # Run prepend again in case we move to another server and need to redo the paths
        args.data_base = tmp_data_base
        args = re_prepend_paths(args)

    base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    base_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'result')

    exp_name = f'{args.exp_name}_{args.target}_{args.model_name}_{now.strftime("%m%d_%H%M%S")}'
    logdir = create_logdir(exp_name, args.resume_training, args)
    args.logdir = logdir

    wandb_mode = 'offline' if args.offline else 'online'
    if args.use_wandb:
       if args.resume_training and args.wandb_id:
            wandb_run = wandb.init(project=args.wandb_project, id=args.wandb_id, entity=args.wandb_entity, name=exp_name, 
                       dir=logdir, resume='allow', config=args, mode=wandb_mode)
       else:
            wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=exp_name, 
                       dir=logdir, config=args, mode=wandb_mode)
    else:
        wandb_run = wandb.init(name=exp_name, project='Debug', entity=args.wandb_entity, dir=logdir, mode=wandb_mode)
    args.wandb_id = wandb.run.id

    print(args)
    if (not args.test) and args.pretrain:
        print('=================================================================================\n')
        print('Start pretraining\n')  
        print('=================================================================================')
        torch.cuda.empty_cache()
        pretrain(args, wandb_run)
        args.checkpoint = os.path.join(args.logdir, 'pretrain', f'checkpoint_last_epoch_{args.max_epochs_pretrain-1}.ckpt')
    
    if args.test:
        print('=================================================================================\n')
        print('Start testing\n')  
        print('=================================================================================')
        torch.cuda.empty_cache()
        test(args, wandb_run)
    elif args.downstream:
        print('=================================================================================\n')
        print('Start downstream\n')  
        print('=================================================================================')
        torch.cuda.empty_cache()
        downstream(args, wandb_run)
    
    wandb.finish()
    end = time.time()
    time_elapsed = end-start
    print('Total running time: {:.0f}h {:.0f}m'.
        format(time_elapsed // 3600, (time_elapsed % 3600)//60))



@property
def exception(self):
  if self._pconn.poll():
    self._exception = self._pconn.recv()
  return self._exception

@hydra.main(config_path='./configs', config_name='config', version_base=None)
def control(args: DictConfig):
  run(args)

if __name__ == "__main__":
  control()