import sys
import os

# Set up paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from utils.config import load_cfg_from_cfg_file

cfg = load_cfg_from_cfg_file("config/refcoco/segvlm_r50.yaml")

print("=" * 70)
print("SEGVLM MODEL PARAMETER COUNTER")
print("=" * 70)

# Import actual model components
from model.layers import FPN, Projector, TransformerDecoder
from model.clip import CLIP

# Manually instantiate components to count parameters
try:
    # Build CLIP with proper R50 dimensions
    # CLIP RN50 standard architecture
    clip = CLIP(
        embed_dim=1024,           # text and image embedding dimension
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),  # ResNet50 layers
        vision_width=64,
        vision_patch_size=None,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        txt_length=17
    )
    backbone_params = sum(p.numel() for p in clip.parameters())
    
    # Build FPN
    fpn_in = cfg.get('fpn_in', [512, 1024, 1024])
    fpn_out = cfg.get('fpn_out', [256, 512, 1024])
    fpn = FPN(in_channels=fpn_in, out_channels=fpn_out, dropout=cfg.get('dropout', 0.1))
    fpn_params = sum(p.numel() for p in fpn.parameters())
    
    # Build Transformer Decoder
    decoder = TransformerDecoder(
        num_layers=cfg.get('num_layers', 3),
        d_model=cfg.get('vis_dim', 512),
        nhead=cfg.get('num_head', 8),
        dim_ffn=cfg.get('dim_ffn', 2048),
        dropout=cfg.get('dropout', 0.1),
        return_intermediate=cfg.get('intermediate', False)
    )
    decoder_params = sum(p.numel() for p in decoder.parameters())
    
    # Build Projector
    projector = Projector(
        cfg.get('word_dim', 1024),
        cfg.get('vis_dim', 512) // 2,
        3,
        dropout=cfg.get('dropout', 0.1)
    )
    projector_params = sum(p.numel() for p in projector.parameters())
    
    # Print breakdown
    print(f"\nPARAMETER COUNT BREAKDOWN:")
    print("-" * 70)
    print(f"Backbone (CLIP):            {backbone_params:>18,} ({backbone_params/1e6:>7.2f}M)")
    print(f"FPN Neck:                   {fpn_params:>18,} ({fpn_params/1e6:>7.2f}M)")
    print(f"Decoder (Transformer):      {decoder_params:>18,} ({decoder_params/1e6:>7.2f}M)")
    print(f"Projector Head:             {projector_params:>18,} ({projector_params/1e6:>7.2f}M)")
    print("-" * 70)
    
    total_params = backbone_params + fpn_params + decoder_params + projector_params
    print(f"TOTAL PARAMETERS:           {total_params:>18,} ({total_params/1e6:>7.2f}M)")
    print("=" * 70)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()