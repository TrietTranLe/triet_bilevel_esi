### script vite fait mal fait pour entrainer un modèle
import os
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

os.environ['HYDRA_FULL_ERROR'] = "1"
@hydra.main(config_path="config", config_name="config_fsav994", version_base=None)
def main(cfg:DictConfig): 
    pl.seed_everything(333) # seed for reproducibility
    ## DATA ##
    dm = hydra.utils.call(cfg.datamodule)
    dm.setup("train")
    # train_dl, val_dl = dm.train_dataloader(), dm.val_dataloader()
    # print(f"{next(iter(val_dl))[0].shape=}")

    litmodel = hydra.utils.call(cfg.litmodel)
    for p in litmodel.solver.prior_cost.parameters(): 
        print(p.mean())
    print(litmodel)

    trainer = hydra.utils.call(cfg.trainer)
    trainer.fit(litmodel, dm)

if __name__ == "__main__": 
    main()