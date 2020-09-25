import torch

# SR : Segmentation Result
# GT : Ground Truth
eps = 1e-6

def get_accuracy(SR, GT, threshold = 0.5):
    """
    ACC: Accuracy

    """
    SR = (SR > threshold)
    GT = (GT == torch.max(GT))

    cor = torch.sum(SR == GT)
    acc = float(cor) / float(SR.numel())

    return acc

def get_sensitivity(SR, GT, threshold = 0.5):
    """
    Sensitivity == Recall

    """
    SR = (SR > threshold)
    GT = (GT == torch.max(GT))

    # TP : True Positive
    # FN : False Negative
    TP =  SR & GT
    FN = ~SR & GT

    SE = float(torch.sum(TP)) / (float(torch.sum(TP)) + float(torch.sum(FN)) + eps)
    
    return SE

def get_specificity(SR, GT, threshold = 0.5):
    """
    SP: Specificity

    """
    SR = (SR > threshold)
    GT = (GT == torch.max(GT))

    # TN : True Negative
    # FP : False Positive
    TN = ~SR & ~GT
    FP =  SR & ~GT

    SP = float(torch.sum(TN)) / (float(torch.sum(TN) + torch.sum(FP)) + eps)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    """
    PC: percision

    """
    SR = (SR > threshold)
    GT = (GT == torch.max(GT))

    # TP : True Positive
    # FP : False Positive
    TP =  SR &  GT
    FP =  SR & ~GT

    PC = float(torch.sum(TP)) / (float(torch.sum(TP)) + float(torch.sum(FP)) + eps)

    return PC

def get_F1(SR, GT, threshold = 0.5):
    """
    F1: F1 score
    """
    SE = get_sensitivity(SR, GT, threshold = threshold)
    PC = get_precision(  SR, GT, threshold = threshold)

    F1 = 2 * SE * PC / (SE + PC + eps)

    return F1

def get_JS(SR, GT, threshold = 0.5):
    """
    JS : Jaccard similarity

    """
    SR = (SR > threshold)
    GT = (GT == torch.max(GT))
    
    Inter = SR & GT
    Union = SR | GT
    
    JS = float(torch.sum(Inter)) / (float(torch.sum(Union)) + eps)
    
    return JS

def get_DC(SR, GT, threshold = 0.5):
    """
    DC : Dice Coefficient

    """
    SR = (SR > threshold)
    GT = (GT == torch.max(GT))

    Inter = SR & GT
    DC = float(2 * torch.sum(Inter)) / (float(torch.sum(SR)) + float(torch.sum(GT)) + eps)

    return DC