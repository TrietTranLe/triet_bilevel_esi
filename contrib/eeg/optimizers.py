import torch


def optim_adam( lit_model, lr ):
    return torch.optim.Adam(
        [
            {"params": lit_model.solver.parameters(), "lr": lr},
        ],
    )

def optim_adam_grad( lit_model, lr ):
    return torch.optim.Adam(
        [
            {"params": lit_model.solver.grad_mod.parameters(), "lr": lr},
        ],
    )

def cosanneal_lr_adam_bis(params, lr, T_max=100, weight_decay=0.):
    """
    optimizer with cosine annealing learning rate
    """
    opt = torch.optim.Adam(
        [
            {"params": params, "lr": lr},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }

def optim_adam_gradphi( lit_mod, lr ): 
    """
    optimizer for both the grad model and the prior cost
    """
    return torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr/2},
        ],
    )

def optim_adamw_gradphi( lit_mod, lr, weight_decay=0.01 ): 
    """
    optimizer for both the grad model and the prior cost, with weight decay
    """
    return torch.optim.AdamW(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr},
        ], weight_decay=weight_decay
    )
    
