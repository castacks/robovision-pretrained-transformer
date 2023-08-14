import torch

def matching_loss_func(matching_preds, matching_gt):
    loss = torch.nn.L1Loss()

    epe = torch.sum((matching_preds[-1] - matching_gt) ** 2, dim=1).sqrt() #FIXME

    metrics = { #FIXME
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    training_loss = loss(matching_preds, matching_gt)

    print(training_loss)

    return training_loss, metrics