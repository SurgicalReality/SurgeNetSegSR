import torch
import numpy as np
from typing import Optional
import os


class SAM2ClickController:
    """
    Wrapper around SAM2ImagePredictor to match the ClickController interface.
    Maintains a history of clicks and re-predicts on each update.
    """

    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'cuda', 
                 max_size: int = 1024):
        """
        Args:
            config_path: SAM2 config name or path (e.g., 'sam2.1_hiera_l.yaml')
            checkpoint_path: Path to SAM2 checkpoint
            device: Device to run on ('cuda' or 'cpu')
            max_size: Maximum size for input images (longer edge)
        """
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError("SAM2 not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")

        self.device = device
        self.max_size = max_size
        
        print(f"[SAM2] Initializing with config: {config_path}, checkpoint: {checkpoint_path}, device: {device}")
        
        # If config_path is an absolute path or points to a custom file outside the sam2 package,
        # register its parent directory as an additional Hydra config search path.
        # Otherwise treat it as a Hydra-relative path (e.g. 'configs/sam2.1/sam2.1_hiera_s.yaml').
        if os.path.isabs(config_path):
            config_dir = os.path.dirname(config_path)
            config_name = os.path.basename(config_path)
            # Register extra search path so Hydra can find the custom config
            from hydra.core.global_hydra import GlobalHydra
            from hydra import initialize_config_dir, compose as hydra_compose
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            initialize_config_dir(config_dir=config_dir, version_base="1.2")
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            cfg = hydra_compose(config_name=config_name)
            OmegaConf.resolve(cfg)
            sam2_model = instantiate(cfg.model, _recursive_=True)
            from sam2.build_sam import _load_checkpoint
            _load_checkpoint(sam2_model, checkpoint_path)
            sam2_model = sam2_model.to(device).eval()
        else:
            # Standard Hydra-relative path (e.g. 'configs/sam2.1/sam2.1_hiera_s.yaml')
            config_name = config_path
            sam2_model = build_sam2(config_name, checkpoint_path, device=device, apply_postprocessing=False)
        
        self.predictor = SAM2ImagePredictor(sam2_model)
        print(f"[SAM2] Predictor initialized successfully")
        
        # State tracking
        self.anchored = False
        self.current_image = None
        self.current_image_shape = None
        self.pos_clicks = []  # List of (x, y) tuples
        self.neg_clicks = []  # List of (x, y) tuples
        self.last_prob = None
        
    def unanchor(self):
        """Reset the image anchor; next interact() will re-set the image."""
        self.anchored = False
        self.current_image = None
        self.pos_clicks = []
        self.neg_clicks = []
        self.last_prob = None
    
    def _normalize_image(self, image: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy uint8 RGB image in (H, W, C) format."""
        # Ensure on CPU
        image = image.cpu()
        
        # Handle batch dimension - remove if present
        while image.ndim > 3:
            if image.shape[0] == 1:
                image = image.squeeze(0)
            else:
                break
        
        # Now image should be (C, H, W) or already (H, W, C)
        # Check if it's (C, H, W) format (channels first) - typical for torch
        if image.ndim == 3:
            if image.shape[0] <= 4 and image.shape[1] > image.shape[0] and image.shape[2] > image.shape[0]:
                # Looks like (C, H, W) - permute to (H, W, C)
                image = image.permute(1, 2, 0)
        
        # Convert to uint8
        if image.dtype == torch.float32 or image.dtype == torch.float64:
            # Assume range [0, 1]
            if image.max() <= 1.0:
                image = (image * 255).to(torch.uint8)
            else:
                image = image.to(torch.uint8)
        elif image.dtype != torch.uint8:
            image = image.to(torch.uint8)
        
        # Make contiguous and convert to numpy
        image_np = image.contiguous().numpy()
        
        # Ensure RGB (3 channels) - drop alpha if present
        if image_np.ndim == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]
        
        # Debug output
        print(f"[SAM2] Image shape: {image_np.shape}, dtype: {image_np.dtype}, C-contiguous: {image_np.flags['C_CONTIGUOUS']}")
        print(f"[SAM2] Image range: [{image_np.min()}, {image_np.max()}]")
        
        # Ensure C-contiguous for SAM2
        return np.ascontiguousarray(image_np, dtype=np.uint8)
    
    def interact(self, image: torch.Tensor, x: int, y: int, is_positive: bool,
                 prev_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process a click and return the predicted mask probability.
        
        Args:
            image: Input image tensor (1, 3, H, W) or (3, H, W)
            x: Click x coordinate
            y: Click y coordinate
            is_positive: Whether this is a positive (True) or negative (False) click
            prev_mask: Previous mask (unused for SAM2 but kept for API compatibility)
        
        Returns:
            Probability tensor of shape (H, W) with values in [0, 1]
        """
        print(f"[SAM2] interact() called - input shape: {image.shape}, device: {image.device}, dtype: {image.dtype}")
        
        # Convert image to numpy if needed
        image_np = self._normalize_image(image)
        
        # Set image if not anchored or if image changed
        if not self.anchored or self.current_image_shape != image_np.shape:
            print(f"[SAM2] Setting new image - anchored: {self.anchored}, shape changed: {self.current_image_shape != image_np.shape if self.current_image_shape else True}")
            try:
                self.predictor.set_image(image_np)
                self.anchored = True
                self.current_image = image_np
                self.current_image_shape = image_np.shape
                self.pos_clicks = []
                self.neg_clicks = []
                print(f"[SAM2] Image set successfully")
            except Exception as e:
                print(f"[SAM2] ERROR setting image: {e}")
                raise
        
        # Add the click to history
        if is_positive:
            self.pos_clicks.append((x, y))
        else:
            self.neg_clicks.append((x, y))
        
        print(f"[SAM2] Click added - pos_clicks: {len(self.pos_clicks)}, neg_clicks: {len(self.neg_clicks)}")
        
        # Build point coordinates and labels
        point_coords = []
        point_labels = []
        
        for px, py in self.pos_clicks:
            point_coords.append([px, py])
            point_labels.append(1)
        
        for px, py in self.neg_clicks:
            point_coords.append([px, py])
            point_labels.append(0)
        
        if not point_coords:
            # No clicks yet; return empty probability
            return torch.zeros(image_np.shape[:2], dtype=torch.float32, device='cpu')
        
        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.int32)
        
        print(f"[SAM2] Predicting with coords: {point_coords.tolist()}, labels: {point_labels.tolist()}")
        
        # Predict
        try:
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,  # Get single best mask
            )
            print(f"[SAM2] Prediction successful - masks shape: {masks.shape}, scores: {scores}")
        except Exception as e:
            print(f"[SAM2] ERROR during prediction: {e}")
            raise
        
        # masks shape: (num_masks, H, W)
        # scores shape: (num_masks,)
        # Take the best mask (highest score)
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]  # (H, W) bool
        
        # Convert to probability: use confidence score
        prob = torch.from_numpy(best_mask.astype(np.float32))
        
        self.last_prob = prob
        return prob
    
    def undo(self) -> Optional[torch.Tensor]:
        """
        Remove the last click and re-predict.
        
        Returns:
            Updated probability tensor or None if no clicks remain
        """
        if not self.pos_clicks and not self.neg_clicks:
            return None
        
        # Remove last click
        if self.neg_clicks:
            self.neg_clicks.pop()
        elif self.pos_clicks:
            self.pos_clicks.pop()
        
        if not self.pos_clicks and not self.neg_clicks:
            # No more clicks
            return None
        
        # Re-predict with remaining clicks
        point_coords = []
        point_labels = []
        
        for px, py in self.pos_clicks:
            point_coords.append([px, py])
            point_labels.append(1)
        
        for px, py in self.neg_clicks:
            point_coords.append([px, py])
            point_labels.append(0)
        
        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.int32)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        prob = torch.from_numpy(best_mask.astype(np.float32))
        
        self.last_prob = prob
        return prob
