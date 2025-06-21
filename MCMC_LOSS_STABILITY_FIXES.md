# MCMC Loss Stability Fixes

## Problem Summary
The CTM training was experiencing loss explosion where the MCMC loss component would:
1. Start with positive high values (585-3500+)
2. Suddenly transition to deeply negative values (-400+) around epoch 2, batch 100
3. Never recover to near-zero convergence after the explosion
4. Cause total loss to become negative and unstable

## Root Causes Identified

### 1. Numerical Precision Issues
- `MCMC_LOSS_GAMMA = 0.00001` was too small, causing numerical precision problems
- `torch.log1p()` on large values could cause numerical instability

### 2. Dynamic Hebbian Scaling Explosion
- Division by very small `abs_hebbian_mean` values caused `dyn_lambda` to explode
- No bounds on the dynamic scaling factor

### 3. Unstable Learning Signal Calculation
- Hard clamping with `torch.clamp(-plasticity_loss, -1.0, 1.0)` created discontinuities
- No bounds on individual loss components before combination

### 4. Insufficient Gradient Monitoring
- No alerts for large gradient norms
- No specific monitoring for MCMC loss magnitude

## Fixes Implemented

### 1. Enhanced MCMC Loss Normalization (`training.py` lines 183-184)
```python
# BEFORE
MCMC_LOSS_GAMMA = 0.00001  # Too small, numerical precision issues

# AFTER  
MCMC_LOSS_GAMMA = 0.001  # Increased for better numerical stability
MAX_MCMC_LOSS_FOR_PLASTICITY = 10.0  # Cap for numerical stability
```

### 2. Improved MCMC Loss Processing (`training.py` lines 285-297)
```python
# BEFORE
mcmc_for_plasticity = torch.relu(mcmc_loss_val.detach()) 
norm_mcmc_loss_for_plasticity = MCMC_LOSS_GAMMA * torch.log1p(mcmc_for_plasticity)

# AFTER
# Clamp MCMC loss to prevent explosion
mcmc_loss_val = torch.clamp(mcmc_loss_val, -MAX_MCMC_LOSS_FOR_PLASTICITY, MAX_MCMC_LOSS_FOR_PLASTICITY)

# Use tanh for bounded output instead of log1p which can explode
mcmc_for_plasticity = mcmc_loss_val.detach()
norm_mcmc_loss_for_plasticity = MCMC_LOSS_GAMMA * torch.tanh(mcmc_for_plasticity / MAX_MCMC_LOSS_FOR_PLASTICITY)
```

### 3. Bounded Dynamic Hebbian Scaling (`training.py` lines 298-303)
```python
# BEFORE
dyn_lambda = orig_local_selector_loss_weight / (abs_hebbian_mean + 1e-8)

# AFTER
# Clamp the denominator to prevent division by very small numbers
abs_hebbian_mean_clamped = torch.clamp(abs_hebbian_mean, min=1e-4, max=10.0)
dyn_lambda = torch.clamp(orig_local_selector_loss_weight / abs_hebbian_mean_clamped, min=0.01, max=10.0)
dynamic_hebbian_loss = dyn_lambda * abs_hebbian_mean_clamped
```

### 4. Enhanced Loss Monitoring (`training.py` lines 305-315)
```python
# Added comprehensive loss monitoring
if (batch_idx + 1) % 10 == 0:  # Print every 10 batches instead of every batch
    print(f"[Losses] Diff: {diffusion_loss.item():.4f}, CE: {ce_loss.item():.4f}, MCMC: {mcmc_loss_val.item():.4f}, Dyn_Heb: {dynamic_hebbian_loss.item():.4f}, Total: {total_loss.item():.4f}")
    
    # MCMC Loss Monitoring
    if abs(mcmc_loss_val.item()) > 5.0:  # Alert if MCMC loss is getting large
        print(f"[WARNING] Large MCMC loss detected: {mcmc_loss_val.item():.4f}")
```

### 5. Enhanced Gradient Clipping (`training.py` lines 320-325)
```python
# Added gradient norm monitoring
total_norm = torch.nn.utils.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
if total_norm > MAX_GRAD_NORM * 2:  # Alert if gradients are very large
    print(f"[WARNING] Large gradient norm detected: {total_norm:.4f}")
```

### 6. Stabilized Plasticity Updates (`training.py` lines 326-329)
```python
# Clamp individual losses before plasticity update
clamped_diffusion_loss = torch.clamp(diffusion_loss, -10.0, 10.0)
clamped_ce_loss = torch.clamp(ce_loss, -10.0, 10.0)
unwrapped_model.ctm_core.apply_activity_plasticity(clamped_diffusion_loss, clamped_ce_loss, norm_mcmc_loss_for_plasticity)
```

### 7. Improved Learning Signal Calculation (`ctm_Diffusion_NEWNEW.py` lines 2743-2748)
```python
# BEFORE
plasticity_loss = diffusion_loss - ce_loss - mcmc_loss.detach()
learning_signal = torch.clamp(-plasticity_loss, -1.0, 1.0)

# AFTER
# Clamp individual losses to prevent extreme values
clamped_diffusion_loss = torch.clamp(diffusion_loss, -5.0, 5.0)
clamped_ce_loss = torch.clamp(ce_loss, -5.0, 5.0)
clamped_mcmc_loss = torch.clamp(mcmc_loss.detach(), -5.0, 5.0)

plasticity_loss = clamped_diffusion_loss - clamped_ce_loss - clamped_mcmc_loss
# Use tanh for smoother, bounded learning signal instead of hard clamp
learning_signal = torch.tanh(plasticity_loss / 2.0)  # Normalize by 2.0 for gentler scaling
```

## Expected Results

### 1. Numerical Stability
- MCMC loss should remain bounded within [-10.0, 10.0] range
- No more sudden transitions to deeply negative values
- Smoother loss curves with gradual convergence

### 2. Better Convergence
- Loss should approach zero more consistently
- No more permanent divergence after initial convergence
- More stable training dynamics

### 3. Enhanced Monitoring
- Early detection of gradient explosion
- Real-time alerts for large MCMC losses
- Better debugging information for loss components

### 4. Improved Plasticity
- Smoother learning signals using tanh instead of hard clamps
- Bounded dynamic scaling prevents explosion
- More stable Hebbian updates

## Monitoring Recommendations

1. **Watch for Warnings**: Pay attention to gradient norm and MCMC loss warnings
2. **Loss Trends**: Monitor that MCMC loss stays within reasonable bounds (-5 to +5)
3. **Convergence**: Look for consistent approach to zero loss without sudden explosions
4. **Plasticity Signals**: Ensure learning signals remain in [-1, 1] range

## Next Steps

1. Run training with these fixes
2. Monitor loss curves for stability
3. Verify convergence behavior
4. Adjust bounds if needed based on observed behavior

The fixes address the core numerical instability issues while maintaining the model's learning capabilities.