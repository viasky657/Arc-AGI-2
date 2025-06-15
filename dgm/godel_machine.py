import random
import time
import os
import json
import tempfile
import shutil
from dgm.alpha_mcmc_evolution import (
    ModelState,
    SurrogateValueFunction,
    MCTS_AlphaZero_Style,
    VALUE_THRESHOLD,
    # Stubs that might be overridden or made more concrete here
    apply_mutation as placeholder_apply_mutation,
    evaluate_model_performance as placeholder_evaluate_model_performance,
    model_quality_estimator as placeholder_model_quality_estimator
)
from dgm.self_improve_step import self_improve, run_harness_polyglot, get_all_performance
from llm import extract_json_between_markers
from dgm.self_improve_step import client

# --- DGM Specific Implementations ---

def dgm_apply_mutation(code_representation: str, mutation: str) -> str:
    """
    **Cheap, Hypothetical Mutation Application**
    Instead of running the expensive self-improvement process, this function
    now creates a simple, hypothetical representation of the code *if* the
    mutation were applied. This is used for cheap exploration by the MCTS.
    """
    # The new "code" is just the old code with a comment indicating the change.
    # This is a lightweight way to represent the hypothetical state.
    return f"{code_representation}\n# HYPOTHETICAL_MUTATION_ATTEMPT: {mutation}\n"


def dgm_execute_mutation(code_representation: str, mutation: str) -> str:
    """
    **Expensive, Real Mutation Execution**
    This function runs the actual, expensive self-improvement process.
    It should only be called once per generation, on the most promising
    mutation found by the MCTS search.
    """
    print(f"DGM_GODEL_MACHINE: EXECUTING mutation (task) '{mutation}'...")

    # Create a temporary directory for this mutation
    with tempfile.TemporaryDirectory() as temp_dir:
        agent_code_path = os.path.join(temp_dir, "coding_agent.py")
        with open(agent_code_path, "w") as f:
            f.write(code_representation)

        output_dir = os.path.join(temp_dir, "output_godel_machine")
        os.makedirs(output_dir, exist_ok=True)

        metadata = self_improve(
            entry=mutation,
            output_dir=output_dir,
            num_evals=0,
            post_improve_diagnose=False,
            polyglot=True,
            # Pass the agent path to self_improve
            agent_file_path=agent_code_path
        )

        new_code = code_representation
        # The patch is applied by self_improve, so we just read the new code
        if os.path.exists(agent_code_path):
            with open(agent_code_path, "r") as f:
                new_code = f.read()
        
        print(f"DGM_GODEL_MACHINE: Mutation EXECUTED. New code length: {len(new_code)}")
        return new_code


def dgm_evaluate_model_performance(code_representation: str) -> float:
    """
    Evaluates the DGM's performance by running the Polyglot harness.
    """
    print(f"DGM_GODEL_MACHINE: Evaluating true performance of DGM code...")

    with tempfile.TemporaryDirectory() as temp_dir:
        agent_code_path = os.path.join(temp_dir, "coding_agent.py")
        with open(agent_code_path, "w") as f:
            f.write(code_representation)

        # We need a task to evaluate on. We'll use a fixed one for now.
        entry_task = "psf__requests-1066" # A polyglot task
        output_dir = os.path.join(temp_dir, f"eval_{int(time.time())}")
        os.makedirs(output_dir, exist_ok=True)
        
        run_id = f"eval_{int(time.time())}"
        
        # Run the harness
        run_harness_polyglot(
            test_task_list=[entry_task],
            model_name_or_path=run_id,
            model_patch_paths=[], # The code is written directly, no patch needed
            num_evals=1,
            pred_dname=os.path.join(output_dir, "predictions"),
            output_dir=output_dir,
            metadata={},
            # Pass the agent path to the harness
            agent_file_path=agent_code_path
        )

        # Get performance
        _, overall_performance = get_all_performance(run_id, results_dir=output_dir)
        
        score = 0.0
        if overall_performance and 'total_resolved_instances' in overall_performance:
            score = overall_performance['total_resolved_instances'] / overall_performance.get('total_instances', 1)
        
        print(f"DGM_GODEL_MACHINE: True performance score: {score:.3f}")
        return score


def dgm_model_quality_estimator(code_representation: str, mutation: str = None) -> float:
    """
    **Enhanced Surrogate Model for Prediction**
    This function now acts as a predictive model. It takes the current code and
    a *proposed mutation* (task) and estimates what the quality of the code
    *would be* if the self-improvement process were run on that task.
    This is the core of the cost-saving refactoring.
    """
    print(f"DGM_GODEL_MACHINE: Predicting quality for mutation '{mutation}'...")
    
    
    # The prompt is updated to ask for a prediction about a future state.
    prompt = f"""
    Please act as a senior software engineer and AI architect.
    You are evaluating a proposed change to a self-improving AI's codebase.

    Current Codebase Summary:
    ---
    {code_representation[:2000]}
    ---
    
    Proposed Task for Self-Improvement:
    ---
    {mutation}
    ---

    Your task is to **predict the final quality of the codebase *after* the AI attempts to solve the proposed task.**
    Consider the difficulty of the task, the capabilities suggested by the current code, and the likelihood of a successful, high-quality patch.

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

    print(f"DGM_GODEL_MACHINE: Predicted quality estimate: {quality:.3f}")
    return quality



# --- Override the placeholder functions in alpha_mcmc_evolution context ---
# This is a way to inject DGM-specific logic into the generic MCTS framework
# by replacing the stub functions it calls.
# Note: This direct override works if alpha_mcmc_evolution directly calls these
# global-like functions. If they are methods of classes that are instantiated
# with these functions, that's cleaner. The current alpha_mcmc_evolution.py
# uses global functions for these helpers.

# It's cleaner if ModelState and other classes are initialized with these functions,
# or if the functions in alpha_mcmc_evolution are designed to be easily swapped.
# For now, we rely on the fact that the `alpha_mcmc_evolution` module's functions
# like `apply_mutation` are global in its scope. When `ModelState.apply_mutation` calls
# `apply_mutation(self.code, mutation)`, it will call the one defined in its module.
# To truly override, we would need to modify how `alpha_mcmc_evolution` accesses these.

# A better approach: The `ModelState` and `MCTS_AlphaZero_Style` should accept these
# functions as arguments, or `SurrogateValueFunction` should encapsulate them.
# Given the current structure of `alpha_mcmc_evolution.py`:
# - `apply_mutation` is global.
# - `evaluate_model_performance` is global.
# - `model_quality_estimator` is global (used by `SurrogateValueFunction.predict`).
# - `generate_possible_rewrites` is global (used by `ModelState.get_possible_mutations`).

# So, we can't directly monkeypatch them from here easily without altering alpha_mcmc_evolution.py
# to accept them as parameters or re-importing them after defining them here (which is tricky).

# The `alpha_mcmc_evolution.py` has been updated so that `ModelState` takes `surrogate_func_ref`
# and `generate_possible_rewrites` can use it.
# The `SurrogateValueFunction` class uses `model_quality_estimator`.
# The `ModelState` methods `apply_mutation` and `evaluate_true_performance` call global functions.

# For this orchestrator, let's define DGM-specific versions and ensure they are used.
# We will pass these to the MCTS and ModelState where possible, or use them to configure
# the objects from `alpha_mcmc_evolution`.

# The `SurrogateValueFunction` in `alpha_mcmc_evolution.py` calls `model_quality_estimator`.
# We can create a DGM-specific surrogate that uses `dgm_model_quality_estimator`.
class DGMSurrogateValueFunction(SurrogateValueFunction):
    def predict(self, model_state: 'DGMModelState') -> float:
        """
        Predicts the quality of a state. If the state is a hypothetical
        mutation, it calls the enhanced estimator with both the base code
        and the proposed mutation.
        """
        if model_state.mutation_attempted:
            # This is a hypothetical state, so we predict its quality.
            # We need the code of the parent state to make the prediction.
            base_code = model_state.parent_code
            return dgm_model_quality_estimator(base_code, model_state.mutation_attempted)
        else:
            # This is a real, instantiated state. Estimate its quality directly.
            return dgm_model_quality_estimator(model_state.code)

# We need a DGMModelState that can distinguish between real and hypothetical states.
class DGMModelState(ModelState):
    def __init__(self, code_representation, surrogate_func_ref=None, mutation_attempted=None, parent_code=None):
        super().__init__(code_representation, surrogate_func_ref)
        self.mutation_attempted = mutation_attempted # The task proposed to reach this state
        self.parent_code = parent_code # The code of the state before the mutation

    def get_possible_mutations(self, use_mcmc_guidance=True) -> list[str]:
        # This now calls the improved generate_possible_rewrites from alpha_mcmc_evolution
        return generate_possible_rewrites(self.code, self.surrogate_func_ref, use_mcmc_guidance)

    def apply_mutation(self, mutation: str) -> 'DGMModelState':
        """
        Creates a new, *hypothetical* state representing the outcome of
        applying the mutation.
        """
        print(f"DGM_GODEL_MACHINE: Creating hypothetical state for mutation: {mutation}")
        # The new code is a lightweight representation of the change.
        hypothetical_code = dgm_apply_mutation(self.code, mutation)
        return DGMModelState(
            hypothetical_code,
            surrogate_func_ref=self.surrogate_func_ref,
            mutation_attempted=mutation,
            parent_code=self.code # Store the parent's code
        )

    def evaluate_true_performance(self) -> float:
        print("DGM_GODEL_MACHINE: DGMModelState.evaluate_true_performance called.")
        # True evaluation should only be called on real, non-hypothetical states.
        if self.mutation_attempted:
            print("Warning: Evaluating true performance on a hypothetical state.")
        return dgm_evaluate_model_performance(self.code)


# --- Main Godel Machine Loop ---
def run_godel_machine_evolution(initial_code: str, num_generations: int, mcts_iterations: int):
    print("\n🚀 Starting DGM Evolution Process (Cost-Effective) 🚀")
    print("=======================================")

    dgm_surrogate_func = DGMSurrogateValueFunction()
    current_best_state = DGMModelState(initial_code, surrogate_func_ref=dgm_surrogate_func)

    print("Performing initial evaluation of the base code...")
    initial_true_score = current_best_state.evaluate_true_performance()
    print(f"Initial True Score: {initial_true_score:.3f}")

    history = [{
        "generation": 0,
        "code": current_best_state.code,
        "true_score": initial_true_score,
        "action": "initial_evaluation"
    }]

    for gen in range(1, num_generations + 1):
        print(f"\n--- Generation {gen}/{num_generations} ---")
        print(f"Current best DGM state has true score: {history[-1]['true_score']:.3f}")

        # MCTS explores hypothetical states cheaply
        hypothetical_state, predicted_score = MCTS_AlphaZero_Style(
            root_state=current_best_state,
            surrogate_value_function=dgm_surrogate_func,
            iterations=mcts_iterations,
            use_mcmc_for_expansion_proposals=True
        )

        generation_summary = {
            "generation": gen,
            "code": current_best_state.code,
            "true_score": history[-1].get("true_score")
        }

        if hypothetical_state and hypothetical_state.mutation_attempted:
            print(f"MCTS proposes mutation '{hypothetical_state.mutation_attempted}' with predicted score {predicted_score:.3f}")
            
            # Now, we execute the single best mutation found by the search
            new_code = dgm_execute_mutation(current_best_state.code, hypothetical_state.mutation_attempted)
            
            # Create a new, real state with the updated code
            new_real_state = DGMModelState(new_code, surrogate_func_ref=dgm_surrogate_func)
            
            # Evaluate the true performance of the new code
            true_score_candidate = new_real_state.evaluate_true_performance()
            print(f"True score of new version: {true_score_candidate:.3f}")

            # Decision logic: Accept if the true score has improved
            previous_true_score = history[-1].get("true_score", 0.0)
            if true_score_candidate > previous_true_score:
                print(f"New DGM version accepted. Score improved from {previous_true_score:.3f} to {true_score_candidate:.3f}.")
                current_best_state = new_real_state
                generation_summary["action"] = "accepted_new_model"
                generation_summary["true_score"] = true_score_candidate
                generation_summary["code"] = new_code
            else:
                print(f"MCTS proposal rejected. Score {true_score_candidate:.3f} is not better than {previous_true_score:.3f}.")
                generation_summary["action"] = "rejected_mcts_proposal"
        else:
            print(f"MCTS did not find a promising new mutation in Generation {gen}.")
            generation_summary["action"] = "kept_existing_model_mcts_found_nothing"
        
        history.append(generation_summary)
        print(f"End of Generation {gen}. Best True Score so far: {history[-1]['true_score']:.3f}")

    print("\n=======================================")
    print("🏁 DGM Evolution Process Finished 🏁")
    final_true_score = history[-1].get("true_score")
    
    print("Final DGM State:")
    print(f"Code: '{current_best_state.code}'")
    print(f"Final True Evaluated Score: {final_true_score:.3f}")

    print("\nEvolution History:")
    for item in history:
        print(f"  Gen {item['generation']}: Action: {item.get('action', 'initial')}, True Score: {item['true_score']:.3f}, Code Preview: '{item['code'][:50]}...'")

    return current_best_state.code


if __name__ == "__main__":
    # The initial code should be the content of the agent we want to improve.
    # Let's read it from the actual file.
    try:
        with open("dgm/coding_agent.py", "r") as f:
            initial_dgm_code = f.read()
    except FileNotFoundError:
        print("Warning: dgm/coding_agent.py not found. Using a placeholder.")
        initial_dgm_code = "# Placeholder for the coding agent"

    # Number of generations determines the DEPTH of the evolution.
    # Each generation builds upon the last. 3 is a reasonable starting point.
    num_generations_to_run = 3
    
    # MCTS iterations determine the BREADTH of the search in each generation.
    # With the new cost-effective design, we can afford to make this search
    # much more thorough to ensure the single chosen mutation is the best one.
    # A value of 10 is too low; 200 provides a much more robust search.
    mcts_iterations_per_gen = 200

    print(f"DGM Orchestrator: Starting evolution with {num_generations_to_run} generations, {mcts_iterations_per_gen} MCTS iterations per gen.")
    
    final_evolved_code = run_godel_machine_evolution(
        initial_code=initial_dgm_code,
        num_generations=num_generations_to_run,
        mcts_iterations=mcts_iterations_per_gen
    )

    print("\n--- Final Evolved DGM Code ---")
    print(final_evolved_code)
    print("-----------------------------")

    # Save the final evolved code
    with open("evolved_dgm_final.py", "w") as f:
        f.write(final_evolved_code)
    print("\nSaved final evolved code to evolved_dgm_final.py")