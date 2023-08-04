import torch

def matching_loss_func(matching_preds, matching_gt):
    loss = torch.nn.L1Loss()

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(matching_gt ** 2, dim=1).sqrt()  # [B, H, W] #FIXME
    valid = (valid >= 0.5) & (mag < 400) #FIXME

    epe = torch.sum((matching_preds[-1] - matching_gt) ** 2, dim=1).sqrt() #FIXME

    epe = epe.view(-1)[valid.view(-1)] #FIXME
    metrics = { #FIXME
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    return loss(matching_preds, matching_gt), metrics