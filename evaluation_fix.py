# ## ARC-AGI-2 Evaluation 
import traceback
from safetensors.torch import load_file
print("\n" + "="*60)
print(f"🔬 STARTING ARC-AGI-2 Evaluation")
print("="*60 + "\n")
if not all([ctm_model_arc is not None, arc_output_head is not None, arc_eval_loader is not None]):
    print("⚠️ Skipping ARC-AGI-2 evaluation due to missing components.")
else:
    latest_epoch = NUM_EPOCHS_ARC
    ctm_checkpoint_path_eval = os.path.join(CHECKPOINT_DIR_ARC, f"ctm_model_arc_epoch_{latest_epoch}.safetensors")
    head_checkpoint_path_eval = os.path.join(CHECKPOINT_DIR_ARC, f"arc_output_head_epoch_{latest_epoch}.safetensors")

    try:
        # Load CTM Model
        if os.path.exists(ctm_checkpoint_path_eval):
            print(f"  > Loading CTM checkpoint from {ctm_checkpoint_path_eval}...")
            unwrapped_ctm_model = accelerator_arc.unwrap_model(ctm_model_arc) if accelerator_arc else ctm_model_arc
            
            # Load the state dict from the safetensors file
            state_dict_ctm = load_file(ctm_checkpoint_path_eval, device=device if not accelerator_arc else accelerator_arc.device)
            
            # Load the state dict into the model
            unwrapped_ctm_model.load_state_dict(state_dict_ctm)
            
            print(f"✓ Loaded CTM checkpoint from epoch {latest_epoch} using safetensors and deepspeed strategy.")
        else:
            print(f"⚠️ CTM Checkpoint not found at {ctm_checkpoint_path_eval}. Evaluating with current model state.")

        # Load ARC Output Head Model
        if os.path.exists(head_checkpoint_path_eval):
            print(f"  > Loading ARC Output Head checkpoint from {head_checkpoint_path_eval}...")
            unwrapped_head_model = accelerator_arc.unwrap_model(arc_output_head) if accelerator_arc else arc_output_head
            
            # Load the state dict from the safetensors file
            state_dict_head = load_file(head_checkpoint_path_eval, device=device if not accelerator_arc else accelerator_arc.device)
            
            # Load the state dict into the model
            unwrapped_head_model.load_state_dict(state_dict_head)
            
            print(f"✓ Loaded ARC Output Head checkpoint from epoch {latest_epoch} using safetensors.")
        else:
            print(f"⚠️ ARC Output Head Checkpoint not found at {head_checkpoint_path_eval}. Evaluating with current model state.")

        ctm_model_arc.eval()
        arc_output_head.eval()
        if ctm_mcmc_integration_arc: ctm_mcmc_integration_arc.eval()
        total_tasks = 0
        solved_tasks = 0
        with torch.inference_mode():
            for task_idx, task_batch in enumerate(arc_eval_loader):
                if not task_batch: continue
                
                current_task_data = task_batch # Dataloader batch_size=1, so task_batch is the task dict
                    
                total_tasks += 1
                task_solved_overall = True

                if 'test' not in current_task_data or not current_task_data['test']:
                    print(f"Task {task_idx + 1} ({current_task_data.get('id', 'N/A')}): No test cases found. Skipping.")
                    task_solved_overall = False
                    continue

                for test_pair_idx, test_pair in enumerate(current_task_data['test']):
                    # Input for evaluation is a single grid, needs to be converted to byte sequence
                    input_grid_np_eval = test_pair['input'].numpy() # Get numpy array from tensor
                    input_bytes_eval_single = serialize_and_pad_grid(input_grid_np_eval, max_len=MAX_SEQUENCE_LENGTH, pad_value=PADDING_BYTE_VALUE)
                    input_bytes_eval = torch.tensor(list(input_bytes_eval_single), dtype=torch.uint8).unsqueeze(0).to(device if not accelerator_arc else accelerator_arc.device)

                    target_grid_np = test_pair['output'].cpu().numpy()
                    original_dims = test_pair['original_output_dims']

                    test_input_solved = False
                    for trial in range(3): # ARC rules allow 3 trials
                        # Forward pass with EnhancedCTMDiffusion using CTM-controlled diffusion for generation
                        # Assuming timestep 0 is appropriate for one-step or final-step generation
                        current_batch_size_eval = input_bytes_eval.size(0) # Should be 1 for evaluation
                        eval_timestep = torch.zeros(current_batch_size_eval, device=input_bytes_eval.device).long()

                        eval_model_output_dict = ctm_model_arc(
                            byte_sequence=input_bytes_eval,
                            mode='ctm_controlled_diffusion', # Use CTM-controlled diffusion
                            target_diffusion_output=None,   # No target during generation
                            timestep=eval_timestep,
                            task_name="ARC_AGI_2_EVAL_DIFFUSION"
                        )
                        
                        # ASSUMPTION: The generated output is a byte sequence under the key 'diffusion_output_pred'
                        # The shape is expected to be (batch_size, MAX_SEQUENCE_LENGTH)
                        predicted_byte_sequence = eval_model_output_dict.get('diffusion_output_pred') 
                        
                        if predicted_byte_sequence is None:
                            print("Warning: Key 'diffusion_output_pred' not found in model output. Trying 'generated_output'.")
                            predicted_byte_sequence = eval_model_output_dict.get('generated_output') # Common alternative
                        
                        if predicted_byte_sequence is None:
                            print("Warning: Generated output key not found. Using zeros as prediction.")
                            # Fallback: create a zero tensor of the expected grid size if generation fails to be found
                            preds_grid = np.zeros(MAX_GRID_SIZE, dtype=int)
                        else:
                            # Ensure the sequence has the correct batch dimension (should be 1)
                            if predicted_byte_sequence.ndim == 1 and current_batch_size_eval == 1:
                                predicted_byte_sequence = predicted_byte_sequence.unsqueeze(0)

                            # Extract the part of the sequence corresponding to the flattened grid
                            # ARC_INPUT_FLAT_DIM = MAX_GRID_SIZE[0] * MAX_GRID_SIZE[1]
                            if predicted_byte_sequence.shape[1] >= ARC_INPUT_FLAT_DIM:
                                preds_flat_bytes = predicted_byte_sequence[0, :ARC_INPUT_FLAT_DIM] # Get first item in batch, first ARC_INPUT_FLAT_DIM bytes
                                # Convert byte values (0-9 for ARC symbols) to long tensor and reshape
                                preds_grid = preds_flat_bytes.view(MAX_GRID_SIZE).long().cpu().numpy()
                            else:
                                print(f"Warning: Generated byte sequence too short ({predicted_byte_sequence.shape[1]} vs {ARC_INPUT_FLAT_DIM}). Using zeros.")
                                preds_grid = np.zeros(MAX_GRID_SIZE, dtype=int)
                        
                        # Unpad to original dimensions
                        h, w = original_dims
                        final_pred = preds_grid[:h, :w]
                        final_target = target_grid_np[:h, :w]

                        if np.array_equal(final_pred, final_target):
                            test_input_solved = True
                            break

                    if not test_input_solved:
                        task_solved_overall = False
                        break
                
                if task_solved_overall:
                    solved_tasks += 1
                    print(f"  Task {task_idx + 1}/{len(arc_eval_loader)} ({current_task_data.get('id', 'N/A')}): SOLVED")
                else:
                    print(f"  Task {task_idx + 1}/{len(arc_eval_loader)} ({current_task_data.get('id', 'N/A')}): FAILED")
        
        if total_tasks > 0:
            accuracy = (solved_tasks / total_tasks) * 100
            summary = f"ARC-AGI-2 Evaluation Summary:\n  Total tasks evaluated: {total_tasks}\n  Tasks solved: {solved_tasks}\n  Accuracy: {accuracy:.2f}%"
            print(f"\n{summary}")
            with open('arc_agi_2_evaluation_summary.txt', 'w') as f:
                f.write(summary)
        else:
            print("\nARC-AGI-2 Evaluation: No tasks were evaluated.")
            
    except Exception as e:
        print(f"❌ Error during ARC-AGI-2 evaluation: {e}")
        traceback.print_exc()
        
    print("\n🔬 ARC-AGI-2 Evaluation Phase Completed.")

#Load the most recent checkpoint-saved model from here and prepare it for evaluation: # --- ARC-AGI-2 Training Loop ---
print("\n" + "="*60)
print(f"🚀 STARTING PHASE 4: ARC-AGI-2 Training")
print(f"   Epochs: {NUM_EPOCHS_ARC}, Batch Size: {ARC_BATCH_SIZE}, Task ID: {ARC_TASK_ID}")
print(f"   Device: {device if not accelerator_arc else accelerator_arc.device}")
print("="*60 + "\n")
if not all([ctm_model_arc, arc_output_head, optimizer_arc, arc_train_loader, arc_criterion]):
    print("⚠️ Skipping ARC-AGI-2 training due to missing components.")
else:
    for epoch in range(NUM_EPOCHS_ARC):
        ctm_model_arc.train()
        arc_output_head.train()
        if ctm_mcmc_integration_arc: ctm_mcmc_integration_arc.train()
        
        total_arc_loss = 0
        processed_batches = 0
        
        for batch_idx, batch_data in enumerate(arc_train_loader):
            if not batch_data or batch_data['input_byte_sequences'].numel() == 0:
                print(f"Skipping empty batch {batch_idx}")
                continue
            
            # Get data from the updated collate_fn
            input_bytes = batch_data['input_byte_sequences'].to(device if not accelerator_arc else accelerator_arc.device)
            target_bytes_for_diffusion = batch_data['target_byte_sequences_for_diffusion'].to(device if not accelerator_arc else accelerator_arc.device)
            original_target_grids_for_ce = batch_data['original_target_grids_for_ce_loss'].to(device if not accelerator_arc else accelerator_arc.device)

            current_batch_size = input_bytes.size(0)

            optimizer_arc.zero_grad()

            with autocast(enabled=USE_MIXED_PRECISION, dtype=autocast_dtype) if not accelerator_arc else accelerator_arc.autocast():
                # Forward pass through EnhancedCTMDiffusion
                # The model internally handles patching, CTM core, diffusion (if target provided), and entropy aux loss.
                model_output_dict = ctm_model_arc(
                    byte_sequence=input_bytes,
                    target_diffusion_output=target_bytes_for_diffusion, # Provide target for diffusion loss component
                    mode='ctm_controlled_diffusion', # Ensure diffusion part is active for loss calculation
                    timestep=torch.randint(0, config_arc_diffusion.diffusion_steps, (current_batch_size,), device=input_bytes.device).long(), # Random timesteps for diffusion training
                    target_mcmc_output=None, # Internal MCMC is disabled in config_arc_diffusion
                    task_name="ARC_AGI_2", # Optional task name
                    current_epoch=epoch # Pass current epoch
                )

                # Loss from EnhancedCTMDiffusion (includes entropy aux loss, diffusion loss, etc.)
                enhanced_ctm_loss = model_output_dict.get('total_loss', torch.tensor(0.0, device=input_bytes.device))
                loss = enhanced_ctm_loss

                # Get CTM core output for the external ARC head
                ctm_core_output_data = model_output_dict.get('ctm_core_data')
                ctm_backbone_output = None
                if ctm_core_output_data and 'final_sync_out' in ctm_core_output_data:
                    ctm_backbone_output = ctm_core_output_data['final_sync_out']
                elif ctm_core_output_data and 'ctm_latent_representation' in ctm_core_output_data: # Fallback key
                    ctm_backbone_output = ctm_core_output_data['ctm_latent_representation']
                else:
                    print("Warning: CTM core output ('final_sync_out' or 'ctm_latent_representation') not found. Using zeros for ARC head input.")
                    ctm_backbone_output = torch.zeros(current_batch_size, config_arc_diffusion.ctm_out_dims, device=input_bytes.device)
                
                # External ARC Output Head for CrossEntropy loss on original grid prediction
                if arc_output_head and ctm_backbone_output is not None:
                    if ctm_backbone_output.ndim > 2 and ctm_backbone_output.shape[1] > 0:
                         ctm_features_for_head = ctm_backbone_output.mean(dim=1)
                    else:
                         ctm_features_for_head = ctm_backbone_output
                    
                    predicted_logits = arc_output_head(ctm_features_for_head)
                    predicted_logits_reshaped = predicted_logits.view(current_batch_size * ARC_INPUT_FLAT_DIM, NUM_ARC_SYMBOLS)
                    target_grids_reshaped = original_target_grids_for_ce.view(current_batch_size * ARC_INPUT_FLAT_DIM)
                    ce_loss = arc_criterion(predicted_logits_reshaped, target_grids_reshaped)
                    loss += ce_loss # Add CE loss to the total loss

                # External MCMC Integration (if enabled)
                if ctm_mcmc_integration_arc and ctm_backbone_output is not None:
                    target_grids_for_mcmc = (original_target_grids_for_ce > 0).float()
                    mcmc_input_features = ctm_backbone_output.detach()
                    if mcmc_input_features.ndim > 2 and mcmc_input_features.shape[1] > 0:
                        mcmc_input_features = mcmc_input_features.mean(dim=1)

                    mcmc_loss_val, _, _ = ctm_mcmc_integration_arc(
                        x=mcmc_input_features,
                        target_y=target_grids_for_mcmc 
                    )
                    loss += mcmc_loss_val

            if scaler: # Mixed precision (manual, without Accelerate)
                scaler.scale(loss).backward()
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer_arc)
                    torch.nn.utils.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
                    scaler.step(optimizer_arc)
                    scaler.update()
                    optimizer_arc.zero_grad()
            elif accelerator_arc: # Using Hugging Face Accelerate
                 accelerator_arc.backward(loss)
                 if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer_arc.step()
                    optimizer_arc.zero_grad()
            else: # Standard training
                loss.backward()
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(ctm_model_arc.parameters(), MAX_GRAD_NORM)
                    optimizer_arc.step()
                    optimizer_arc.zero_grad()

            total_arc_loss += loss.item()
            processed_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS_ARC}], Batch [{batch_idx+1}/{len(arc_train_loader)}], Loss: {loss.item():.4f}")
        
        avg_epoch_loss = total_arc_loss / processed_batches if processed_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS_ARC}] completed. Average Loss: {avg_epoch_loss:.4f}")
        
        from safetensors.torch import save_file
        import os
        import torch.distributed as dist
        
        # === DEBUG + RANK CHECK ===
        def get_rank_debug():
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1
            
            print(f"[DEBUG] Rank {rank} out of {world_size} total ranks")
            return rank, world_size
        
        rank, world_size = get_rank_debug()
        
        # === SAVE ONLY ON RANK 0 ===
        if rank == 0 and CHECKPOINT_DIR_ARC:
            model_to_save_ctm = accelerator_arc.unwrap_model(ctm_model_arc) if accelerator_arc else ctm_model_arc
            model_to_save_head = accelerator_arc.unwrap_model(arc_output_head) if accelerator_arc else arc_output_head
            
            # Check if DeepSpeed is used and the model is wrapped
            if hasattr(model_to_save_ctm, 'zero_optimization') and hasattr(model_to_save_ctm, 'module'):
                print("✓ Using DeepSpeed consolidated state_dict for CTM model")
                ctm_state_dict = model_to_save_ctm._zero3_consolidated_16bit_state_dict()
            else:
                ctm_state_dict = model_to_save_ctm.state_dict()

            if hasattr(model_to_save_head, 'zero_optimization') and hasattr(model_to_save_head, 'module'):
                print("✓ Using DeepSpeed consolidated state_dict for ARC head")
                head_state_dict = model_to_save_head._zero3_consolidated_16bit_state_dict()
            else:
                head_state_dict = model_to_save_head.state_dict()

            # Save model weights with safetensors
            save_file(ctm_state_dict, os.path.join(CHECKPOINT_DIR_ARC, f"ctm_model_arc_epoch_{epoch+1}.safetensors"))
            save_file(head_state_dict, os.path.join(CHECKPOINT_DIR_ARC, f"arc_output_head_epoch_{epoch+1}.safetensors"))

            # Save optimizer (use torch.save, not supported by safetensors)
            torch.save(optimizer_arc.state_dict(), os.path.join(CHECKPOINT_DIR_ARC, f"optimizer_arc_epoch_{epoch+1}.pt"))

            print(f"✓ Checkpoint saved for epoch {epoch+1} on rank {rank} to {CHECKPOINT_DIR_ARC}")