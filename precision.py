import torch

eps = 0.0001

def acc_calc(input, target):
    target = torch.split(target, 1, dim=0)[0]
    correct = torch.sum((input==target).float())
    total_preds = target.numel()+eps
    t = (correct + eps) / total_preds
    return t

def recall_calc(input, target):
    target = torch.split(target, 1, dim=0)[0]
    correct = torch.dot(input.view(-1), target.view(-1)) # TP
    T = torch.sum(target)+eps # TP + FN
    t = (correct + eps) / T
    return t

def precision_calc(input, target):
    target = torch.split(target, 1, dim=0)[0]
    correct = torch.dot(input.view(-1), target.view(-1))
    T = torch.sum(input)+eps # TP + FN
    t = (correct + eps) / T
    return t

def PixelAcc(input, target):
    """Pixel Accuracy for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + acc_calc(c[0], c[1])

    return s / (i + 1)

def Recall(input, target):
    """Recall for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + recall_calc(c[0], c[1])

    return s / (i + 1)

def Precision(input, target):
    """Recall for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + precision_calc(c[0], c[1])

    return s / (i + 1)