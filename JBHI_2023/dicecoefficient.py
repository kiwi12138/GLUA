import torch
import torch.nn as nn
import numpy as np
import monai
from medpy.metric.binary import assd,dc,hd

class DiceCoefficient(nn.Module):
    def __init__(self):
        super(DiceCoefficient, self).__init__()

    def forward(self, y_true, y_pred, epsilon=1e-6):
        y_true_flatten = np.asarray(y_true).astype(np.bool)
        y_pred_flatten = np.asarray(y_pred).astype(np.bool)

        if not np.sum(y_true_flatten) + np.sum(y_pred_flatten):
            return 1.0

        return (2. * np.sum(y_true_flatten * y_pred_flatten)) / \
               (np.sum(y_true_flatten) + np.sum(y_pred_flatten) + epsilon)


class MulticlassDiceCoefficient(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """
    def __init__(self):
        super(MulticlassDiceCoefficient, self).__init__()


    def forward(self, input, target, weights=None):
        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes
        coefficient = DiceCoefficient()
        input = np.array(input.cpu())
        input = input.argmax(1)
        # input = np.transpose(input,(1,2,0))
        input = one_hot(input)
        input = np.transpose(input, (1, 0, 2, 3))

        target = np.array(target.cpu())
        # target = np.transpose(target,(1,2,0))
        target = one_hot(target)
        target = np.transpose(target,(1,0,2,3))

        dice_score=[]
        assd_score=[]
        hd_score=[]
        for c in range(1,input.shape[1],1):

                dice_score.append(dc(input[:,c],target[:,c]))
                assd_score.append(assd(input[:,c],target[:,c]))

        return np.array(dice_score)


def one_hot(seg):
    vals = np.array([0, 1, 2, 3, 4])
    res = np.zeros([len(vals)] + list(seg.shape), seg.dtype)
    for i, c in enumerate(vals):
        res[i][seg == c] = 1
    return res