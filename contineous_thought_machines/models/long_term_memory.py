import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import random
from enum import Enum
from .mamba_block import MambaBlock, Mamba2Block

class MemoryReplayPolicy(Enum):
    SIMPLE_REPLAY = "simple_replay"
    SURPRISE_WEIGHTED_REPLAY = "surprise_weighted_replay"
    USEFULNESS_REPLAY = "usefulness_replay"

class LongTermMemory(nn.Module):
    """
    A hippocampus-like long-term memory module for storing and retrieving
    high-surprise states. This version uses a Mamba block to process retrieved
    memories for more efficient and context-aware synthesis.
    """
    def __init__(self, d_model: int, memory_size: int = 1024, top_k: int = 5, replay_policy: MemoryReplayPolicy = MemoryReplayPolicy.SIMPLE_REPLAY):
        super().__init__()
        self.d_model = d_model
        self.memory_size = memory_size
        self.top_k = top_k
        self.replay_policy = replay_policy

        # Initialize active memory buffers
        self.register_buffer('memory', torch.zeros(self.memory_size, self.d_model))
        self.register_buffer('memory_surprise', torch.zeros(self.memory_size))
        self.register_buffer('memory_timestamp', torch.zeros(self.memory_size, dtype=torch.long))
        self.register_buffer('memory_usage', torch.zeros(self.memory_size, dtype=torch.long))
                
        self.register_buffer('is_initialized', torch.zeros(self.memory_size, dtype=torch.bool))
        self.memory_pointer = 0
        self.global_time = 0
        
        # External memory for overflow
        self.external_memory = []  # List of dicts: {'state': tensor, 'surprise': float, 'timestamp': int, 'usage': int}
        
        # Mamba block for processing retrieved memories
        self.retrieval_processor = Mamba2Block(
            d_model=d_model, d_state=64, d_head=64, expand=2, chunk_size=256
        )
        self.retrieval_norm = nn.LayerNorm(d_model)

    def add_to_memory(self, state: torch.Tensor, surprise: torch.Tensor):
        """
        Adds a state to active memory if space available, otherwise to external memory.
        """
        state = state.to(self.memory.device)
        surprise = surprise.to(self.memory.device)
        self.global_time += 1
        
        if self.memory_pointer < self.memory_size:
            idx = self.memory_pointer
            self.memory_pointer += 1
            self.memory[idx] = state
            self.memory_surprise[idx] = surprise
            self.memory_timestamp[idx] = self.global_time
            self.memory_usage[idx] = 1
            self.is_initialized[idx] = True
        else:
            # Add to external memory
            self.external_memory.append({
                'state': state.clone(),
                'surprise': surprise.item(),
                'timestamp': self.global_time,
                'usage': 1
            })

    def retrieve_from_memory(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieves from both active and external memory, processes top-k with Mamba.
        """
        num_valid_active = self.is_initialized.sum().item()
        num_external = len(self.external_memory)
        if num_valid_active + num_external == 0:
            return torch.zeros(1, self.d_model, device=query.device)

        # Active memory
        if num_valid_active > 0:
            valid_active = self.memory[self.is_initialized]
            sim_active = F.cosine_similarity(query.unsqueeze(0), valid_active, dim=1)
        else:
            sim_active = torch.tensor([], device=query.device)

        # External memory
        if num_external > 0:
            external_states = torch.stack([mem['state'] for mem in self.external_memory])
            sim_external = F.cosine_similarity(query.unsqueeze(0), external_states, dim=1)
        else:
            sim_external = torch.tensor([], device=query.device)

        # Combine similarities
        all_sim = torch.cat([sim_active, sim_external])
        
        k = min(self.top_k, len(all_sim))
        if k == 0:
            return torch.zeros(1, self.d_model, device=query.device)
            
        top_k_values, top_k_indices = torch.topk(all_sim, k=k)
        
        # Retrieve states
        retrieved = []
        for idx in top_k_indices:
            if idx < num_valid_active:
                orig_idx = torch.where(self.is_initialized)[0][idx]
                retrieved.append(self.memory[orig_idx])
                self.memory_usage[orig_idx] += 1
            else:
                ext_idx = idx - num_valid_active
                retrieved.append(self.external_memory[ext_idx]['state'])
                self.external_memory[ext_idx]['usage'] += 1
        
        retrieved_memories = torch.stack(retrieved) # (1, k, d_model)
        
        # Process the sequence with the Mamba block
        retrieved_sequence = retrieved_memories.unsqueeze(0)
        processed_memory = self.retrieval_processor(retrieved_sequence)  # (1, k, d_model)
        contextualized_memory = processed_memory[:, -1, :] # (1, d_model)
        
        return self.retrieval_norm(contextualized_memory)

    def replay_memory(self, batch_size: int) -> Optional[torch.Tensor]:
        """
        Replays memories from both active and external buffers according to the replay policy.
        """
        num_valid_active = self.is_initialized.sum().item()
        num_external = len(self.external_memory)
        total_memories = num_valid_active + num_external
        if total_memories == 0:
            return None

        actual_batch_size = min(batch_size, total_memories)
        
        if self.replay_policy == MemoryReplayPolicy.SIMPLE_REPLAY:
            # Simple random selection from all
            all_indices = torch.randperm(total_memories)[:actual_batch_size]
            return self._select_memories(all_indices, num_valid_active)

        elif self.replay_policy == MemoryReplayPolicy.SURPRISE_WEIGHTED_REPLAY:
            # Collect all surprises
            active_surprise = self.memory_surprise[self.is_initialized]
            external_surprise = torch.tensor([mem['surprise'] for mem in self.external_memory], device=active_surprise.device)
            all_surprise = torch.cat([active_surprise, external_surprise])
            probs = F.softmax(all_surprise, dim=0)
            if probs.sum() > 0:
                indices = torch.multinomial(probs, num_samples=actual_batch_size, replacement=True)
                return self._select_memories(indices, num_valid_active)
            else:
                # Fallback to simple
                all_indices = torch.randperm(total_memories)[:actual_batch_size]
                return self._select_memories(all_indices, num_valid_active)

        elif self.replay_policy == MemoryReplayPolicy.USEFULNESS_REPLAY:
            # Calculate utility for active
            active_age = self.global_time - self.memory_timestamp[self.is_initialized].float()
            active_utility = (
                self.memory_surprise[self.is_initialized]
                + 0.1 * self.memory_usage[self.is_initialized].float()
                - 0.01 * active_age
            )
            
            # Calculate for external
            external_ages = torch.tensor([self.global_time - mem['timestamp'] for mem in self.external_memory], device=active_utility.device)
            external_surprises = torch.tensor([mem['surprise'] for mem in self.external_memory], device=active_utility.device)
            external_usages = torch.tensor([mem['usage'] for mem in self.external_memory], device=active_utility.device)
            external_utility = external_surprises + 0.1 * external_usages.float() - 0.01 * external_ages.float()
            
            all_utility = torch.cat([active_utility, external_utility])
            probs = F.softmax(all_utility, dim=0)
            if probs.sum() > 0:
                indices = torch.multinomial(probs, num_samples=actual_batch_size, replacement=True)
                return self._select_memories(indices, num_valid_active)
            else:
                # Fallback to simple
                all_indices = torch.randperm(total_memories)[:actual_batch_size]
                return self._select_memories(all_indices, num_valid_active)

        return None
        
    def _select_memories(self, indices: torch.Tensor, num_valid_active: int) -> torch.Tensor:
        """Helper to select states from indices spanning active and external."""
        selected = []
        for idx in indices:
            if idx < num_valid_active:
                orig_idx = torch.where(self.is_initialized)[0][idx]
                selected.append(self.memory[orig_idx])
            else:
                ext_idx = idx - num_valid_active
                selected.append(self.external_memory[ext_idx]['state'])
        return torch.stack(selected)