#!/usr/bin/env python3
"""
Test script to verify that the EnhancedCTMConfig can be created successfully
with all the required QAT parameters.
"""

import sys
import os

# Add the workspace to Python path
sys.path.insert(0, '/workspaces/Arc-AGI-2')

try:
    # Import the actual EnhancedCTMConfig from the correct module
    from contineous_thought_machines.models.ctm_components import EnhancedCTMConfig
    
    print("✅ Successfully imported EnhancedCTMConfig")
    
    # Test creating the config with QAT parameters
    config = EnhancedCTMConfig(
        d_model=512,
        ctm_use_qat=True,
        ctm_adaptive_quantization=True,
        ctm_quant_min_bits=2,
        ctm_quant_max_bits=8,
        ctm_quant_policy_search=True,
        ctm_selective_quantization=True,
        output_audio_bytes=True
    )
    
    print("✅ Configuration created successfully!")
    print(f"   QAT enabled: {config.ctm_use_qat}")
    print(f"   Adaptive quantization: {config.ctm_adaptive_quantization}")
    print(f"   Quant min bits: {config.ctm_quant_min_bits}")
    print(f"   Quant max bits: {config.ctm_quant_max_bits}")
    print(f"   Policy search: {config.ctm_quant_policy_search}")
    print(f"   Selective quantization: {config.ctm_selective_quantization}")
    print(f"   Output audio bytes: {config.output_audio_bytes}")
    
    # Test that all required attributes exist
    required_attrs = [
        'd_model', 'n_heads', 'n_layers', 'max_sequence_length',
        'ctm_use_qat', 'ctm_adaptive_quantization', 'ctm_quant_min_bits',
        'ctm_quant_max_bits', 'ctm_quant_policy_search', 'ctm_selective_quantization',
        'output_audio_bytes', 'inferred_task_latent_dim'
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(config, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        print(f"❌ Missing attributes: {missing_attrs}")
    else:
        print("✅ All required attributes present")
    
    print("\n🎉 Configuration test completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except TypeError as e:
    print(f"❌ TypeError when creating config: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)