import torch
import wandb
from typing import Dict

wandb.login(key="aac9b8a16c68d4865d40c7c09bc9d86730ff4548")
class WandBLogger:

    def __init__(self, enabled=True, 
                 model: torch.nn.modules=None, 
                 run_name: str=None) -> None:
        
        self.enabled = enabled



        if self.enabled:
            wandb.init(entity="annalaczko",
                        project="dlfvc_assignment1",
                        group="try1")
            if run_name is None:
                wandb.run.name = wandb.run.id    
            else:
                wandb.run.name = run_name  

            if model is not None:
                self.watch(model)         
            
    def watch(self, model, log_freq: int=1):
        wandb.watch(model, log="all", log_freq=log_freq)
            

    def log(self, log_dict: dict, commit=True, step=None):
        if self.enabled:
            if step:
                wandb.log(log_dict, commit=commit, step=step)
            else:
                wandb.log(log_dict, commit=commit)
 

    def finish(self):
        if self.enabled:
            wandb.finish()
