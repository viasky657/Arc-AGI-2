import math
import random
import re
from dataclasses import dataclass
from typing import List, Callable, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dgm.llm import extract_json_between_markers
from dgm.self_improve_step import client


# --- Constants ---
VALUE_THRESHOLD = 0.75 # Threshold for triggering the final, expensive evaluation


# --- External Judge and Co-piloted Surrogate ---

class GodelMachineJudge:
    """
    An external judge that uses an LLM to evaluate a self-edit's quality,
    inspired by the DGM's model quality estimator.
    """
    def evaluate(self, state: str) -> float:
        """
        Calls an LLM to evaluate the quality of the given text state.
        Returns a score between 0.0 and 1.0.
        """
        print("--- (Calling Godel-style LLM Judge) ---")
        
        # The prompt is simplified to evaluate a single piece of text.
        prompt = f"""
        Please act as a senior software engineer and AI architect.
        You are evaluating the quality of a piece of text, which is a proposed "self-edit" for an AI model.

        Text to Evaluate:
        ---
        {state[:4000]}
        ---

        Your task is to **predict the quality of this text.**
        Consider its clarity, coherence, and potential usefulness as an instruction or an edit for an AI.

        Provide a predicted quality score between 0.0 and 1.0.
        Return your response as a JSON object with a single key "predicted_quality_score".
        """
        
        try:
            completion = client.chat.completions.create(
                model="google/gemini-2.5-pro-preview",
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            response = completion.choices[0].message.content
            response_json = extract_json_between_markers(response)
            quality = float(response_json.get("predicted_quality_score", 0.0))
        except Exception as e:
            print(f"Error getting quality prediction from LLM: {e}")
            quality = random.uniform(0.1, 0.4) # Fallback to a lower random score

        print(f"--- LLM Judge returned score: {quality:.3f} ---")
        return quality

# --- Configuration and Scheduling ---

@dataclass
class MCMCConfig:
    """Configuration for MCMC sampling parameters."""
    num_chains: int = 200
    chain_length: int = 100
    burn_in: int = 10
    temperature_schedule: str = "geometric"
    initial_temp: float = 5.0
    final_temp: float = 1.0
    decay_rate: float = 0.99

class TemperatureScheduler:
    """Generates a temperature schedule for simulated annealing."""
    @staticmethod
    def get_schedule(config: MCMCConfig) -> Callable[[int], float]:
        if config.temperature_schedule == "geometric":
            return lambda step: max(config.initial_temp * (config.decay_rate ** step), config.final_temp)
        elif config.temperature_schedule == "linear":
            return lambda step: config.initial_temp - (config.initial_temp - config.final_temp) * min(step / config.chain_length, 1.0)
        else: # constant
            return lambda step: config.final_temp

# --- Output Space for Self-Edits ---

class SelfEditSpace:
    """
    Defines the combinatorial space of self-edits and their neighborhoods.
    A "state" in this space is the text of a self-edit.
    """
    def get_neighbors(self, state: str) -> List[str]:
        """
        Generates neighbors of a state using a symmetric proposal strategy.
        Strategy: Swap any two distinct sentences.
        """
        # Use regex to split into sentences, keeping delimiters.
        sentences = [s for s in re.split(r'([.!?]\s*)', state) if s]
        if len(sentences) < 4: # Need at least 2 sentences with delimiters to swap
            return []

        # Group sentences with their delimiters
        grouped_sentences = ["".join(sentences[i:i+2]) for i in range(0, len(sentences) -1, 2)]
        if len(sentences) % 2 == 1:
            grouped_sentences.append(sentences[-1])

        if len(grouped_sentences) < 2:
            return []

        neighbors = []
        from itertools import combinations
        for i, j in combinations(range(len(grouped_sentences)), 2):
            new_list = grouped_sentences[:]
            new_list[i], new_list[j] = new_list[j], new_list[i]
            neighbors.append("".join(new_list))
        
        return neighbors

    def get_proposal_prob(self, from_state: str, to_state: str) -> float:
        """
        Computes the proposal probability q(to_state | from_state).
        Since our 'swap' neighborhood is symmetric, this is uniform over neighbors.
        """
        neighbors = self.get_neighbors(from_state)
        if not neighbors:
            return 0.0
        
        # Check if to_state is in the list of neighbors
        if to_state in neighbors:
            return 1.0 / len(neighbors)
        
        return 0.0

# --- Surrogate Model ---

class SurrogateValueFunction:
    """Base class for a model that predicts the quality of a self-edit."""
    def predict(self, state: str) -> float:
        """Returns a score for a given self-edit string. Higher is better."""
        raise NotImplementedError

    def update(self, state: str, true_reward: float):
        """Updates the surrogate model with a new true evaluation."""
        pass # Base implementation does nothing

class JudgedNeuralSurrogate(SurrogateValueFunction):
    """
    A surrogate that uses an internal neural network but can appeal to an
    external judge when its confidence is low.
    """
    def __init__(self, judge: GodelMachineJudge):
        self.judge = judge
        self.internal_surrogate = NeuralSurrogate()
        self.error_history = []
        self.call_judge_prob = 1.0  # Start by always calling the judge

    def predict(self, state: str) -> float:
        """
        Dynamically decides whether to use the internal model or call the
        expensive external judge.
        """
        if random.random() < self.call_judge_prob:
            # Make the expensive call to the external judge
            true_score = self.judge.evaluate(state)
            # Use this opportunity to train our internal model
            self.internal_surrogate.update(state, true_score)
            self._update_judge_probability()
            return true_score
        else:
            # Use the cheap internal model's prediction
            return self.internal_surrogate.predict(state)

    def update(self, state: str, true_reward: float):
        """
        Updates the internal model with a true reward obtained from the
        main evaluation loop (not the judge).
        """
        # Record the error before updating
        prediction = self.internal_surrogate.predict(state)
        error = (prediction - true_reward) ** 2
        self.error_history.append(error)
        
        # Update the internal model
        self.internal_surrogate.update(state, true_reward)
        self._update_judge_probability()

    def _update_judge_probability(self):
        """
        Adjusts the probability of calling the judge based on the recent
        performance of the internal surrogate model.
        """
        if len(self.error_history) < 10:
            self.call_judge_prob = 1.0 # Stay in high-supervision mode
            return

        # Use a moving average of the mean squared error
        recent_mse = np.mean(self.error_history[-20:])
        
        # Sigmoid-like function to map error to probability.
        # High error -> high probability, low error -> low probability.
        self.call_judge_prob = 1 / (1 + np.exp(-(recent_mse - 0.1) * 10))
        print(f"--- Surrogate confidence updated. Recent MSE: {recent_mse:.4f}. Judge call probability: {self.call_judge_prob:.2f} ---")


class NeuralSurrogate(SurrogateValueFunction):
    """A learned surrogate model using a small neural network."""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.model = nn.Sequential(
            nn.Linear(1000, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        self.history_X = []
        self.history_y = []
        self.is_fitted = False

    def predict(self, state: str) -> float:
        if not self.is_fitted:
            return 0.5 # Default score before the model is trained
        
        self.model.eval()
        with torch.no_grad():
            try:
                features = self.vectorizer.transform([state]).toarray()
                tensor_features = torch.FloatTensor(features)
                prediction = self.model(tensor_features)
                return prediction.item()
            except Exception:
                # If a term is not in the vocabulary, predict a neutral score
                return 0.5

    def update(self, state: str, true_reward: float):
        """Adds a new data point and retrains the model."""
        self.history_X.append(state)
        self.history_y.append(true_reward)
        
        # Retrain the model with all accumulated data
        if len(self.history_X) > 1:
            X_features = self.vectorizer.fit_transform(self.history_X).toarray()
            y_true = torch.FloatTensor(self.history_y).unsqueeze(1)
            
            self.model.train()
            for epoch in range(10): # Simple training loop
                self.optimizer.zero_grad()
                predictions = self.model(torch.FloatTensor(X_features))
                loss = self.loss_fn(predictions, y_true)
                loss.backward()
                self.optimizer.step()
            self.is_fitted = True

# --- MCMC Sampler ---

def metropolis_hastings_sampler(
    initial_state: str,
    surrogate_func: SurrogateValueFunction,
    output_space: SelfEditSpace,
    config: MCMCConfig
) -> str:
    """
    Performs a Metropolis-Hastings search to find a high-quality self-edit.
    """
    best_state = initial_state
    best_energy = -surrogate_func.predict(initial_state)
    
    current_state = initial_state
    current_energy = best_energy

    temp_schedule = TemperatureScheduler.get_schedule(config)
    total_steps = config.chain_length + config.burn_in

    for step in range(total_steps):
        temperature = temp_schedule(step)
        
        # Propose a new state from the neighborhood
        neighbors = output_space.get_neighbors(current_state)
        if not neighbors:
            continue
        
        proposal_state = random.choice(neighbors)
        
        # Calculate energy of the proposal
        proposal_energy = -surrogate_func.predict(proposal_state)
        
        # Metropolis-Hastings acceptance criterion
        energy_diff = proposal_energy - current_energy
        
        # For a symmetric proposal (like swapping), the proposal ratio is 1.
        # q(current|proposal) / q(proposal|current) = 1
        acceptance_prob = min(1.0, math.exp(-energy_diff / temperature))
        
        if random.random() < acceptance_prob:
            current_state = proposal_state
            current_energy = proposal_energy
            
            if current_energy < best_energy:
                best_state = current_state
                best_energy = current_energy
                
    return best_state
    
if __name__ == '__main__':
    # --- Example Execution ---
    initial_text = "The quick brown fox jumps over the lazy dog. This is a test sentence. We need to see if this works."
    
    # 1. Initialize the external judge
    judge = GodelMachineJudge()
    
    # 2. Initialize the surrogate model with the judge
    surrogate = JudgedNeuralSurrogate(judge)
    
    # 3. Define the output space
    space = SelfEditSpace()
    
    # 4. Configure MCMC
    config = MCMCConfig(
        num_chains=1, # For a single run demonstration
        chain_length=50,
        burn_in=5,
        initial_temp=10.0,
        final_temp=1.0
    )
    
    print("--- Starting MCMC Search for Best Self-Edit ---")
    print(f"Initial Text: '{initial_text}'")
    
    # 5. Run the sampler to get a candidate edit
    candidate_edit = metropolis_hastings_sampler(
        initial_state=initial_text,
        surrogate_func=surrogate,
        output_space=space,
        config=config
    )
    
    print("\n--- MCMC Search Finished ---")
    print(f"Candidate Edit Found: '{candidate_edit}'")

    # 6. Temporary Isolation: Decide whether to perform final evaluation
    candidate_score = surrogate.predict(candidate_edit)
    print(f"Surrogate score for candidate: {candidate_score:.3f}")

    final_edit = initial_text
    if candidate_score > VALUE_THRESHOLD:
        print(f"Candidate score {candidate_score:.3f} > threshold {VALUE_THRESHOLD}. Performing final, isolated evaluation.")
        # The "true" evaluation is a definitive call to the judge
        true_reward = judge.evaluate(candidate_edit)
        print(f"Final 'True' Reward for candidate: {true_reward:.3f}")
        
        # In a real system, we might have another threshold on the true_reward
        if true_reward > 0.7: # Example of a final acceptance threshold
            print("Final evaluation passed. Accepting the new edit.")
            final_edit = candidate_edit
        else:
            print("Final evaluation did not pass. Discarding the edit.")
    else:
        print(f"Candidate score {candidate_score:.3f} did not exceed threshold {VALUE_THRESHOLD}. Discarding the edit.")

    print("\n--- Process Finished ---")
    print(f"Final Accepted Edit: '{final_edit}'")