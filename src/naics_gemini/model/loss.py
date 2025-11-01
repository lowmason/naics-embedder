import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    Prevents the model from becoming over-confident.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing >= 0.0
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor): Logits. Shape: (batch_size, num_classes)
            target (torch.Tensor): Ground truth labels. Shape: (batch_size,)
        """
        num_classes = output.size(1)
        log_probs = F.log_softmax(output, dim=-1)
        
        # Create the smooth label tensor
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        # Calculate the KL divergence loss
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        return torch.mean(loss)

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced datasets.
    Down-weights the loss for well-classified examples, focusing on hard ones.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (torch.Tensor, optional): Weighting factor for each class. 
                                            Shape: (num_classes,).
            gamma (float): Focusing parameter.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, output, target):
        """
        Args:
            output (torch.Tensor): Logits. Shape: (batch_size, num_classes)
            target (torch.Tensor): Ground truth labels. Shape: (batch_size,)
        """
        num_classes = output.size(1)
        
        # Calculate Cross-Entropy Loss
        ce_loss = F.cross_entropy(output, target, reduction='none')
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Calculate Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != output.device:
                self.alpha = self.alpha.to(output.device)
            
            # Gather alpha weights for each sample
            alpha_t = self.alpha.gather(0, target)
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_loss_function(loss_name: str = 'cross_entropy', **kwargs):
    """
    Factory function for loss functions.
    
    Args:
        loss_name (str): 'cross_entropy', 'label_smoothing', or 'focal'.
        **kwargs: Arguments to pass to the loss constructor.
                  For 'label_smoothing': {'smoothing': 0.1}
                  For 'focal': {'alpha': None, 'gamma': 2.0}
    
    Returns:
        nn.Module: The instantiated loss function.
    """
    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

