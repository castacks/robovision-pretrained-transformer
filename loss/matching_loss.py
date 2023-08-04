import torch

def matching_loss_func(matching_prediction_and_information, matching_gt):
    loss = torch.nn.L1Loss()
    return loss(matching_prediction_and_information[0], matching_gt)