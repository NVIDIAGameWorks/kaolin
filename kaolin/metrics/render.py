import torch

def mask_iou(lhs_mask, rhs_mask):
    """Compute the Intersection over Union of two segmentation masks.

    Args:
        lhs_mask (torch.FloatTensor): A segmentation mask of shape (B, H, W)
        rhs_mask (torch.FloatTensor): A segmentation mask of shape (B, H, W)

    Returns:
        (torch.FloatTensor): The IoU loss, as a torch scalar.
    """
    batch_size, height, width = lhs_mask.shape
    assert rhs_mask.shape == lhs_mask.shape
    sil_mul = lhs_mask * rhs_mask
    sil_add = lhs_mask + rhs_mask
    iou_up = torch.sum(sil_mul.reshape(batch_size, -1), dim=1)
    iou_down = torch.sum((sil_add - sil_mul).reshape(batch_size, -1), dim=1)
    iou_neg = iou_up / (iou_down + 1e-10)
    mask_loss = 1.0 - torch.mean(iou_neg)
    return mask_loss
