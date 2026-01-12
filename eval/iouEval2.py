# Code for evaluating IoU
# Nov 2017
# Eduardo Romera
# Adapted for robust anomaly detection 2025
#######################

import torch


class iouEval:
    def __init__(self, nClasses, ignoreIndex=255):
        self.nClasses = nClasses
        # If ignoreIndex is out of class range, set to -1 (no ignore)
        self.ignoreIndex = (
            ignoreIndex if (ignoreIndex >= 0 and ignoreIndex < nClasses) else -1
        )
        self.reset()

    def reset(self):
        # Determine number of valid classes to evaluate
        classes_to_score = (
            self.nClasses if self.ignoreIndex == -1 else self.nClasses - 1
        )
        self.tp = torch.zeros(classes_to_score).double()
        self.fp = torch.zeros(classes_to_score).double()
        self.fn = torch.zeros(classes_to_score).double()

    def addBatch(self, x, y):  # x=preds, y=targets
        # 1. Ensure inputs are 4D [B, C, H, W]
        if x.dim() == 2:  # [H, W]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:  # [B, H, W]
            x = x.unsqueeze(1)

        if y.dim() == 2:
            y = y.unsqueeze(0).unsqueeze(0)
        elif y.dim() == 3:
            y = y.unsqueeze(1)

        # 2. Device Consistency
        if x.device != y.device:
            x = x.to(y.device)

        # 3. Convert Predictions to One-Hot if they are indices [B, 1, H, W]
        if x.size(1) == 1:
            x_onehot = torch.zeros(
                x.size(0), self.nClasses, x.size(2), x.size(3), device=x.device
            )
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()

        # 4. Convert Targets to One-Hot if they are indices [B, 1, H, W]
        if y.size(1) == 1:
            y_onehot = torch.zeros(
                y.size(0), self.nClasses, y.size(2), y.size(3), device=y.device
            )
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        # 5. Handle Ignore Index (Proper Masking)
        if self.ignoreIndex != -1:
            # Extract the 'ignore' channel mask
            ignores = y_onehot[:, self.ignoreIndex, :, :].unsqueeze(1)
            # Keep only valid class channels
            valid_indices = [i for i in range(self.nClasses) if i != self.ignoreIndex]
            x_onehot = x_onehot[:, valid_indices, :, :]
            y_onehot = y_onehot[:, valid_indices, :, :]
        else:
            ignores = 0

        # 6. Calculate TP, FP, FN
        # Intersection
        tpmult = x_onehot * y_onehot
        tp = torch.sum(tpmult, dim=(0, 2, 3))

        # False Positives: predicted as class, but GT is not that class AND not ignore
        fpmult = x_onehot * (1 - y_onehot - ignores)
        fp = torch.sum(fpmult, dim=(0, 2, 3))

        # False Negatives: GT is class, but predicted as something else
        fnmult = (1 - x_onehot) * y_onehot
        fn = torch.sum(fnmult, dim=(0, 2, 3))

        # Accumulate
        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou  # returns mean_iou, per_class_iou


# Class for colors
class colors:
    RED = "\033[31;1m"
    GREEN = "\033[32;1m"
    YELLOW = "\033[33;1m"
    BLUE = "\033[34;1m"
    MAGENTA = "\033[35;1m"
    CYAN = "\033[36;1m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ENDC = "\033[0m"


def getColorEntry(val):
    """
    Helper to colorize terminal output.
    Accepts floats, numpy scalars, or tensor items.
    """
    try:
        f_val = float(val)
    except (TypeError, ValueError):
        return colors.ENDC

    if f_val < 0.20:
        return colors.RED
    elif f_val < 0.40:
        return colors.YELLOW
    elif f_val < 0.60:
        return colors.BLUE
    elif f_val < 0.80:
        return colors.CYAN
    else:
        return colors.GREEN
