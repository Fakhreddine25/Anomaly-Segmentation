import torch
import torch.nn.functional as F


def get_MSP_score(outputs):
    probabilities = F.softmax(outputs, dim=1)
    score = 1.0 - probabilities.max(dim=1)[0]
    return score.squeeze(0)


def get_MaxL_score(outputs):
    score = -outputs.squeeze(0).max(dim=0).values
    return score


def get_MaxE_score(outputs):
    probabilities = F.softmax(outputs, dim=1)
    # entropy = torch.div(torch.sum(-probabilities * torch.log(probabilities), dim=1), torch.log(torch.tensor(probabilities.shape[1])))
    entropy = torch.sum(-probabilities * torch.log(probabilities), dim=1)
    return entropy.squeeze(0)


def get_MSP_T_score(outputs, tempScale):
    t = (torch.ones(1) * tempScale).cuda()
    probabilities = F.softmax(outputs / t, dim=1)
    score = 1.0 - probabilities.max(dim=1)[0]
    return score.squeeze(0)
