import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import random
from enum import Enum
from .mamba_block import MambaBlock

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

        # Initialize memory buffers
        self.register_buffer('memory', torch.zeros(self.memory_size, self.d_model))
        self.register_buffer('memory_surprise', torch.zeros(self.memory_size))
        self.register_buffer('memory_age', torch.zeros(self.memory_size, dtype=torch.long))
        self.register_buffer('memory_usage', torch.zeros(self.memory_size, dtype=torch.long))
        
        self.register_buffer('is_initialized', torch.zeros(self.memory_size, dtype=torch.bool))
        self.memory_pointer = 0
        
        # Mamba block for processing retrieved memories
        self.retrieval_processor = MambaBlock(
            d_model=d_model, d_state=16, d_conv=4, expand=2
        )
        self.retrieval_norm = nn.LayerNorm(d_model)

    def add_to_memory(self, state: torch.Tensor, surprise: torch.Tensor):
        """
        Adds a state to the memory. If the memory is full, it replaces an existing
        entry based on a combination of low surprise, high age, and low usage.
        """
        # Ensure state is on the correct device
        state = state.to(self.memory.device)
        surprise = surprise.to(self.memory.device)
        
        if self.memory_pointer < self.memory_size:
            # Fill memory sequentially until full
            idx = self.memory_pointer
            self.memory_pointer += 1
        else:
            # Eviction strategy: Replace the least valuable memory entry
            # Score = surprise + 0.1 * usage - 0.01 * age
            # We want to replace the one with the lowest score.
            utility_scores = (
                self.memory_surprise 
                + 0.1 * self.memory_usage.float() 
                - 0.01 * self.memory_age.float()
            )
            idx = torch.argmin(utility_scores)

        # Add the new state to memory at the determined index
        self.memory[idx] = state
        self.memory_surprise[idx] = surprise
        self.memory_age[idx] = 0
        self.memory_usage[idx] = 1 # Start with a usage count of 1
        self.is_initialized[idx] = True

        # Increment age of all other valid memories
        age_mask = torch.ones_like(self.memory_age, dtype=torch.bool)
        age_mask[idx] = False
        valid_entries = self.is_initialized & age_mask
        self.memory_age[valid_entries] += 1

    def retrieve_from_memory(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the top_k most similar states from memory, combines them
        using a Mamba block, and returns a contextualized memory vector.
        Increments the usage count of retrieved memories.
        """
        num_valid_memories = self.is_initialized.sum()
        if num_valid_memories == 0:
            return torch.zeros(1, self.d_model, device=query.device)

        # Get only the valid, initialized memories for retrieval
        valid_memory = self.memory[self.is_initialized]
        
        # Cosine similarity between the query and all valid memories
        similarities = F.cosine_similarity(query.unsqueeze(0), valid_memory, dim=1)
        
        # Determine how many memories to retrieve
        k = min(self.top_k, num_valid_memories.item())
        
        # Get the indices of the top_k most similar memories within the valid subset
        top_k_indices_in_valid = torch.topk(similarities, k=k).indices
        
        # Get the actual top-k memories
        retrieved_memories = valid_memory[top_k_indices_in_valid] # Shape (k, d_model)

        # Update usage statistics for the retrieved memories
        original_indices = torch.where(self.is_initialized)[0][top_k_indices_in_valid]
        self.memory_usage[original_indices] += 1

        # Mamba requires a sequence input: (batch, seq_len, d_model)
        # We treat the retrieved memories as a sequence for a single batch item.
        retrieved_sequence = retrieved_memories.unsqueeze(0) # (1, k, d_model)
        
        # Process the sequence with the Mamba block
        processed_memory = self.retrieval_processor(retrieved_sequence) # (1, k, d_model)
        
        # We take the output corresponding to the last memory in the sequence
        # as the final contextualized representation.
        contextualized_memory = processed_memory[:, -1, :] # (1, d_model)
        
        return self.retrieval_norm(contextualized_memory)

    def replay_memory(self, batch_size: int) -> Optional[torch.Tensor]:
        """
        Replays memories from the buffer according to the replay policy.
        """
        num_valid_memories = self.is_initialized.sum().item()
        if num_valid_memories == 0:
            return None

        actual_batch_size = min(batch_size, num_valid_memories)
        valid_indices = torch.where(self.is_initialized)[0]

        if self.replay_policy == MemoryReplayPolicy.SIMPLE_REPLAY:
            indices = torch.randperm(num_valid_memories)[:actual_batch_size]
            selected_indices = valid_indices[indices]
            return self.memory[selected_indices]

        elif self.replay_policy == MemoryReplayPolicy.SURPRISE_WEIGHTED_REPLAY:
            valid_surprise = self.memory_surprise[self.is_initialized]
            probs = F.softmax(valid_surprise, dim=0)
            if probs.sum() > 0:
                indices = torch.multinomial(probs, num_samples=actual_batch_size, replacement=True)
                selected_indices = valid_indices[indices]
                return self.memory[selected_indices]
            else:
                # Fallback to simple replay if probabilities are all zero
                indices = torch.randperm(num_valid_memories)[:actual_batch_size]
                selected_indices = valid_indices[indices]
                return self.memory[selected_indices]

        elif self.replay_policy == MemoryReplayPolicy.USEFULNESS_REPLAY:
            # Score = surprise + 0.1 * usage - 0.01 * age
            utility_scores = (
                self.memory_surprise[self.is_initialized]
                + 0.1 * self.memory_usage[self.is_initialized].float() 
                - 0.01 * self.memory_age[self.is_initialized].float()
            )
            probs = F.softmax(utility_scores, dim=0)
            if probs.sum() > 0:
                indices = torch.multinomial(probs, num_samples=actual_batch_size, replacement=True)
                selected_indices = valid_indices[indices]
                return self.memory[selected_indices]
            else:
                # Fallback to simple replay
                indices = torch.randperm(num_valid_memories)[:actual_batch_size]
                selected_indices = valid_indices[indices]
                return self.memory[selected_indices]

        return None