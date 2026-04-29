import numpy as np
import torch

try:
    from transformers import Sam2Processor, Sam2Model
    _sam2_available = True
except ImportError:
    _sam2_available = False
    Sam2Processor = None
    Sam2Model = None
    print('Failed to import Sam2Processor/Sam2Model from transformers. '
          'SAM2 functionality will not be available. '
          'Ensure transformers>=4.47.0 is installed.')


class GlobalSegmentAnything:
    """
    Keeps track of a global SAM2 model loaded via HuggingFace transformers.
    Call set_default_model_id() before get_global_sam2() to configure which model to load.
    """
    _processor = None
    _model = None
    _model_id = None  # e.g. "facebook/sam2-hiera-large"

    @staticmethod
    def set_default_model_id(model_id):
        GlobalSegmentAnything._model_id = model_id

    @staticmethod
    def get_global_sam2(device):
        if GlobalSegmentAnything._model is not None:
            return GlobalSegmentAnything._processor, GlobalSegmentAnything._model
        if GlobalSegmentAnything._model_id is None:
            return None, None
        processor, model = load_sam2(GlobalSegmentAnything._model_id, device=device)
        GlobalSegmentAnything._processor = processor
        GlobalSegmentAnything._model = model
        return processor, model


def load_sam2(model_id, device="cuda"):
    """Load SAM2 processor and model from HuggingFace hub."""
    if not _sam2_available:
        raise ImportError('transformers with Sam2 support not available; '
                          'install transformers>=4.47.0')
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id).to(device).eval()
    return processor, model


def default_sam2_components(device):
    """Returns (processor, model) for the globally registered SAM2 model, or (None, None)."""
    return GlobalSegmentAnything.get_global_sam2(device)

class SimplerSAMSegmenter:
    def __init__(self, sam2_processor, sam2_model):
        self.sam2_processor = sam2_processor
        self.sam2_model = sam2_model

    def compute_mask(self, view, points, labels):
        """

        Args:
            view: H x W x 3 torch uint8 0...255
            points: nested to 4 levels [image level, object level, point level, coord level]
            labels: nested to 3 levels [image level, object level, point level]

        Returns:

        """
        if len(points) == 0 or view is None:
            return None
        assert len(view.shape) == 3, f'Batched input not supported'

        device = view.device
        inputs = self.sam2_processor(
            images=view, # [H, W, 3] uint8 RGB for SAM2
            input_points=points,   # [[[[x0,y0], ...]]] — 4 levels: image, object, point, coord
            input_labels=labels,    # [[[l0, ...]]] — 3 levels: image, object, labels
            return_tensors="pt"
        )
        # Only move tensor values to device; sizes may be lists in transformers 5.x
        model_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.sam2_model(**model_inputs)

        # list of [1 x 3 x H x W]
        masks = self.sam2_processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])

        best_idx = torch.argmax(outputs['iou_scores'])
        return masks[0][0, best_idx, ...]