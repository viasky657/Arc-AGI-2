import torch
import torch.nn.functional as F
import re
import os
import numpy as np # For entropy calculation

def compute_decay(T, params, clamp_lims=(0, 15)):
    """
    This function computes exponential decays for learnable synchronisation 
    interactions between pairs of neurons. 
    """
    assert len(clamp_lims), 'Clamp lims should be length 2'
    assert type(clamp_lims) == tuple, 'Clamp lims should be tuple'
    
    indices = torch.arange(T-1, -1, -1, device=params.device).reshape(T, 1).expand(T, params.shape[0])
    out = torch.exp(-indices * torch.clamp(params, clamp_lims[0], clamp_lims[1]).unsqueeze(0))
    return out

def add_coord_dim(x, scaled=True):
    """
    Adds a final dimension to the tensor representing 2D coordinates.

    Args:
        tensor: A PyTorch tensor of shape (B, D, H, W).

    Returns:
        A PyTorch tensor of shape (B, D, H, W, 2) with the last dimension
        representing the 2D coordinates within the HW dimensions.
    """
    B, H, W = x.shape
    # Create coordinate grids
    x_coords = torch.arange(W, device=x.device, dtype=x.dtype).repeat(H, 1)  # Shape (H, W)
    y_coords = torch.arange(H, device=x.device, dtype=x.dtype).unsqueeze(-1).repeat(1, W)  # Shape (H, W)
    if scaled:
        x_coords /= (W-1)
        y_coords /= (H-1)
    # Stack coordinates and expand dimensions
    coords = torch.stack((x_coords, y_coords), dim=-1)  # Shape (H, W, 2)
    coords = coords.unsqueeze(0)  # Shape (1, 1, H, W, 2)
    coords = coords.repeat(B, 1, 1, 1)  # Shape (B, D, H, W, 2)
    return coords

def compute_normalized_entropy(logits, reduction='mean'):
    """
    Calculates the normalized entropy of a PyTorch tensor of logits along the 
    final dimension.

    Args:
      logits: A PyTorch tensor of logits. 

    Returns:
      A PyTorch tensor containing the normalized entropy values.
    """

    # Apply softmax to get probabilities
    preds = F.softmax(logits, dim=-1)

    # Calculate the log probabilities
    log_preds = torch.log_softmax(logits, dim=-1)

    # Calculate the entropy
    entropy = -torch.sum(preds * log_preds, dim=-1)

    # Calculate the maximum possible entropy
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

    # Normalize the entropy
    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)

    return normalized_entropy
    
def reshape_predictions(predictions, prediction_reshaper):
    B, T = predictions.size(0), predictions.size(-1)
    new_shape = [B] + prediction_reshaper + [T]
    rehaped_predictions = predictions.reshape(new_shape)
    return rehaped_predictions

def get_all_log_dirs(root_dir):
    folders = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.endswith(".pt") for f in filenames):
            folders.append(dirpath)
    return folders

def get_latest_checkpoint(log_dir):
    files = [f for f in os.listdir(log_dir) if re.match(r'checkpoint_\d+\.pt', f)]
    return os.path.join(log_dir, max(files, key=lambda f: int(re.search(r'\d+', f).group()))) if files else None

def get_latest_checkpoint_file(filepath, limit=300000):
    checkpoint_files = get_checkpoint_files(filepath)
    checkpoint_files = [
        f for f in checkpoint_files if int(re.search(r'checkpoint_(\d+)\.pt', f).group(1)) <= limit
    ]
    if not checkpoint_files:
        return None
    return checkpoint_files[-1]

def get_checkpoint_files(filepath):
    regex = r'checkpoint_(\d+)\.pt'
    files = [f for f in os.listdir(filepath) if re.match(regex, f)]
    files = sorted(files, key=lambda f: int(re.search(regex, f).group(1)))
    return [os.path.join(filepath, f) for f in files]

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint

def get_model_args_from_checkpoint(checkpoint):
    if "args" in checkpoint:
        return(checkpoint["args"])
    else:
        raise ValueError("Checkpoint does not contain saved args.")

def get_accuracy_and_loss_from_checkpoint(checkpoint, device="cpu"):
    training_iteration = checkpoint.get('training_iteration', 0)
    train_losses = checkpoint.get('train_losses', [])
    test_losses = checkpoint.get('test_losses', [])
    train_accuracies = checkpoint.get('train_accuracies_most_certain', [])
    test_accuracies = checkpoint.get('test_accuracies_most_certain', [])
    return training_iteration, train_losses, test_losses, train_accuracies, test_accuracies

class TaskAnalyzer:
    """
    Analyzes input data or context to determine task characteristics,
    such as modality, to guide HIPA and other adaptive components.
    """
    def __init__(self, config=None):
        """
        Initializes the TaskAnalyzer.
        Args:
            config: Optional configuration object that might contain parameters
                    for modality detection heuristics.
        """
        self.config = config
        self.modality_heuristics = {
            "audio": {"fft_dims": [-1], "freq_threshold": 0.2, "enhancement_strength": 0.5, "use_hipa": True},
            "image": {"fft_dims": [-2, -1], "freq_threshold": 0.1, "enhancement_strength": 0.3, "use_hipa": True},
            "text": {"use_hipa": False},
            "Point Cloud": {"use_hipa": False},
            "Textures": {"fft_dims": [-2, -1], "freq_threshold": 0.1, "enhancement_strength": 0.3, "use_hipa": True},
            # For unknown/default, allow learned signal to decide. Provide generic continuous data params.
            "unknown": {"use_hipa": True, "fft_dims": [-1], "freq_threshold": 0.15, "enhancement_strength": 0.25, "modality_type": "continuous_generic"},
            "default": {"use_hipa": True, "fft_dims": [-1], "freq_threshold": 0.15, "enhancement_strength": 0.25, "modality_type": "continuous_generic"}
        }

    def _calculate_shannon_entropy(self, data_tensor: torch.Tensor) -> float:
        """Helper to calculate Shannon entropy for a byte-like tensor."""
        if data_tensor.numel() == 0:
            return 0.0
        # Ensure tensor is on CPU and is integer type for byte-like analysis
        data_bytes = data_tensor.cpu().to(torch.uint8).numpy().tobytes()
        if not data_bytes:
            return 0.0
        
        counts = np.bincount(np.frombuffer(data_bytes, dtype=np.uint8))
        probabilities = counts / len(data_bytes)
        probabilities = probabilities[probabilities > 0] # Avoid log(0)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def detect_modality(self, x: torch.Tensor, task_id: int = None, context_hints: dict = None) -> dict:
        """
        Detects the modality of the input data and returns a configuration for HIPA.

        Args:
            x (torch.Tensor): The input tensor. Shape can vary (e.g., B, S, D or B, C, H, W).
            task_id (int, optional): An optional task identifier.
            context_hints (dict, optional): Additional context hints, e.g., {'ctm_data': ctm_data_dict}.

        Returns:
            dict: A configuration dictionary for HIPA, including:
                  'modality': Detected modality (e.g., 'audio', 'image', 'text', 'unknown').
                  'use_hipa': Boolean indicating if HIPA should be applied.
                  'fft_dims': Dimensions for FFT (e.g., [-1] for 1D, [-2, -1] for 2D).
                  'freq_threshold': Threshold for high-frequency component detection.
                  'enhancement_strength': Strength of frequency enhancement.
                  'detection_source': String indicating how modality was determined.
        """
        # 1. Prioritize explicit hints
        if context_hints:
            if "expected_modality" in context_hints:
                hinted_modality = context_hints["expected_modality"]
                if hinted_modality in self.modality_heuristics:
                    config = self.modality_heuristics[hinted_modality].copy()
                    config['modality'] = hinted_modality
                    config['detection_source'] = 'explicit_hint'
                    # Allow further overrides from hints
                    if "force_hipa_on" in context_hints and context_hints["force_hipa_on"]: config['use_hipa'] = True
                    if "force_hipa_off" in context_hints and context_hints["force_hipa_off"]: config['use_hipa'] = False
                    return config
            if "filename" in context_hints: # Basic file extension check
                filename = context_hints["filename"].lower()
                if filename.endswith((".wav", ".mp3", ".flac")):
                    config = self.modality_heuristics["audio"].copy(); config['modality'] = "audio"; config['detection_source'] = 'hint_filename_audio'; return config
                if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    config = self.modality_heuristics["image"].copy(); config['modality'] = "image"; config['detection_source'] = 'hint_filename_image'; return config
                if filename.endswith((".txt", ".md", ".json", ".csv")):
                    config = self.modality_heuristics["text"].copy(); config['modality'] = "text"; config['detection_source'] = 'hint_filename_text'; return config
                if filename.endswith((".xyz", ".ply", ".obj", ".glb", ".pts")): # Common point cloud extensions
                    config = self.modality_heuristics["Point Cloud"].copy(); config['modality'] = "Point Cloud"; config['detection_source'] = 'hint_filename_pointcloud'; return config

# 2. Tensor Shape and dtype Analysis
        b = x.shape[0]
        dims = x.shape[1:]
        dtype = x.dtype

        # Image-like (B, C, H, W)
        if x.ndim == 4 and dims[0] in [1, 3, 4] and dims[1] > 16 and dims[2] > 16: # C, H, W
            config = self.modality_heuristics["image"].copy()
            config['modality'] = "image"
            config['detection_source'] = 'shape_image_4d'
            return config

        # Point Cloud-like (B, N, 3 or 6)
        if x.ndim == 3 and dims[1] in [3, 6] and dims[0] > 10: # N, D (D=3 for XYZ, 6 for XYZRGB/Normals)
            config = self.modality_heuristics["Point Cloud"].copy()
            config['modality'] = "Point Cloud"
            config['detection_source'] = 'shape_point_cloud_3d'
            return config

        # Sequence-like (B, S, D)
        if x.ndim == 3:
            seq_len, feat_dim = dims[0], dims[1]
            if feat_dim < 128 and seq_len > 256: 
                config = self.modality_heuristics["audio"].copy() 
                config['modality'] = "audio_or_long_sequence"
                config['detection_source'] = 'shape_sequence_3d_long_narrow'
                return config
            else: 
                config = self.modality_heuristics["text"].copy() 
                config['modality'] = "generic_sequence_embeddings"
                config['detection_source'] = 'shape_sequence_3d_generic'
                return config

        # Flat sequence / Raw bytes / Tokenized text (B, L)
        if x.ndim == 2:
            seq_len = dims[0]
            if dtype == torch.long and seq_len > 1: 
                config = self.modality_heuristics["text"].copy()
                config['modality'] = "text_tokenized"
                config['detection_source'] = 'shape_dtype_tokenized_text_2d'
                return config
            elif dtype == torch.uint8 and seq_len > 64: 
                # entropy = self._calculate_shannon_entropy(x) # Example: could use entropy
                if seq_len > 1024: 
                    config = self.modality_heuristics["audio"].copy()
                    config['modality'] = "audio_byte_stream"
                    config['detection_source'] = 'shape_dtype_byte_stream_long_2d'
                else:
                    config = self.modality_heuristics["default"].copy()
                    config['modality'] = "generic_byte_stream"
                    config['detection_source'] = 'shape_dtype_byte_stream_short_2d'
                return config
            elif dtype in [torch.float32, torch.float16] and seq_len > 256: 
                config = self.modality_heuristics["audio"].copy()
                config['modality'] = "audio_signal_1d"
                config['detection_source'] = 'shape_dtype_audio_signal_1d_2d'
                return config

        # 3. Fallback
        config = self.modality_heuristics["unknown"].copy()
        config['modality'] = "unknown"
        config['detection_source'] = 'fallback_unknown'
        return config
