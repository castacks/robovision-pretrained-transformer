import torch
import cv2
import numpy as np

def matching_loss_func(matching_preds, matching_gt):
    loss = torch.nn.L1Loss(reduction = 'sum')

    cv2.imwrite('test_images/model_output.png', (matching_preds[0, 0, :, :] * 255).unsqueeze(dim=0).permute(1, 2, 0).to('cpu').detach().numpy().astype(np.uint8))
    cv2.imwrite('test_images/ground_truth.png', (matching_gt[0, 0, :, :] * 255).unsqueeze(dim=0).permute(1, 2, 0).to('cpu').detach().numpy().astype(np.uint8))

    epe = torch.sum((matching_preds[-1] - matching_gt) ** 2, dim=1).sqrt() #FIXME

    metrics = { #FIXME
        'epe': epe.mean().item(),
        '1px': (epe > 1).float().mean().item(),
        '3px': (epe > 3).float().mean().item(),
        '5px': (epe > 5).float().mean().item(),
    }

    training_loss = loss(matching_preds, matching_gt) / 10

    print(training_loss)

    return training_loss, metrics