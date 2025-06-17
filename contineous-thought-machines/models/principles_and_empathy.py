# contineous-thought-machines/models/principles_and_empathy.py

"""
This module implements the logic for a combined empathy and principles-based learning system.
It includes mechanisms for:
- Representing principles in a knowledge graph.
- A dual-reward system for both empathic actions and principle adherence.
- An interaction loop for training a student model with a teacher agent.
"""

import os
# Assume these classes are defined elsewhere and imported.
# from .knowledge_store import UniversalKnowledgeStore as UKS
# from .student_model import StudentModel
# from .teacher_api import TeacherAPI

class PrinciplesAndEmpathyTrainer:
    """
    A class to manage the training process combining principles and empathy.
    """
    def __init__(self, student_model, teacher_api, uks, principle_weight=0.5):
        self.student_model = student_model
        self.teacher_api = teacher_api
        self.uks = uks
        self.principle_weight = principle_weight

    def add_principles_to_graph(self):
        """
        Adds a set of moral/social principles to the knowledge graph.
        """
        # Correctly locate the principles.txt file relative to the current script.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        principles_path = os.path.join(dir_path, 'Principles', 'principles.txt')

        with open(principles_path, 'r') as f:
            # Read lines, strip whitespace, and filter out empty lines.
            principles = [line.strip() for line in f if line.strip()]
        
        for p_label in principles:
            # Using MERGE to avoid duplicates
            self.uks.graph.run("MERGE (p:Thing:Principle {label: $label})", label=p_label)

        # Example of connecting actions to principles
        self.uks.add_statement("Comfort", "FOLLOWS_PRINCIPLE", "BeKind")
        self.uks.add_statement("RespectSpace", "FOLLOWS_PRINCIPLE", "RespectAutonomy")

    def assign_dual_rewards(self):
        """
        Encodes dual rewards for actions in the knowledge graph.
        """
        # Example: 'Comfort' action
        # +1.0 for helping Alex emotionally
        self.uks.add_action_outcome("Comfort", "AlexRelieved", valence="positive", weight=1.0)
        # +0.5 for following the kindness rule
        self.uks.add_action_outcome("Comfort", "Follows_BeKind", valence="positive", weight=self.principle_weight)


    def interaction_loop(self, max_steps=10):
        """
        Runs the main interaction loop between the student and teacher agent.
        """
        for step in range(max_steps):
            teacher_state = self.teacher_api.get_state()

            # Update graph with teacher's current state
            self.update_teacher_state_in_graph(teacher_state)

            # Student proposes actions
            candidate_actions = self.student_model.propose_actions(teacher_state)

            # Evaluate via simulated or real teacher feedback
            # The evaluation should consider the dual reward system.
            best_action = self.student_model.evaluate_actions(
                candidate_actions,
                teacher_state,
                principle_weight=self.principle_weight
            )

            # Apply best action and observe new teacher state
            result = self.teacher_api.respond_to_action(best_action)

            # Update knowledge graph and reinforce
            self.uks.update_with_action_result(best_action, result)
            self.student_model.reinforce(best_action, result)

    def update_teacher_state_in_graph(self, teacher_state):
        """
        Updates the knowledge graph with the teacher's current state.
        """
        agent_name = teacher_state.get("agent_name")
        emotion = teacher_state.get("emotion")
        goal = teacher_state.get("goal")

        if agent_name and emotion:
            self.uks.add_statement(agent_name, "HAS_EMOTION", emotion)
        if agent_name and goal:
            self.uks.add_statement(agent_name, "HAS_GOAL", goal)

def calculate_total_reward(empathy_reward, principle_reward, principle_weight=0.5):
    """
    Calculates the total reward by combining empathy and principle rewards.
    """
    return empathy_reward * 1.0 + principle_reward * principle_weight

# Example of how the evaluation logic might work inside the student model
def example_conflict_resolution():
    """
    Demonstrates how the model would choose between two actions
    based on the combined reward.
    """
    # Action 1: Hug the agent
    hug_empathy_score = 1.0  # calms
    hug_principle_score = 0.0  # violates autonomy
    hug_total_reward = calculate_total_reward(hug_empathy_score, hug_principle_score) # 1.0

    # Action 2: Ask for consent
    ask_empathy_score = 0.7  # slower, but still positive
    ask_principle_score = 1.0  # respects autonomy (assuming principle reward is binary 0 or 1)
    ask_total_reward = calculate_total_reward(ask_empathy_score, ask_principle_score) # 0.7 + 0.5 = 1.2

    print(f"Hug reward: {hug_total_reward}")
    print(f"Ask for consent reward: {ask_total_reward}")

    if ask_total_reward > hug_total_reward:
        print("Model chooses to ask for consent. ✅")
    else:
        print("Model chooses to hug.")

if __name__ == '__main__':
    # This is for demonstration purposes.
    # In a real scenario, these objects would be properly initialized.
    class MockUKS:
        def add_statement(self, *args): print(f"UKS: add_statement({args})")
        def add_action_outcome(self, *args, **kwargs): print(f"UKS: add_action_outcome({args}, {kwargs})")
        def update_with_action_result(self, *args): print(f"UKS: update_with_action_result({args})")
        def run(self, *args, **kwargs): print(f"UKS DB: run({args}, {kwargs})")
        graph = property(lambda self: self) # self.graph.run -> self.run

    class MockStudent:
        def propose_actions(self, *args): print("Student: proposing actions."); return ["Hug", "Ask for consent"]
        def evaluate_actions(self, *args, **kwargs): print(f"Student: evaluating actions with {kwargs}."); return "Ask for consent"
        def reinforce(self, *args): print(f"Student: reinforcing with {args}")

    class MockTeacher:
        def get_state(self): return {"agent_name": "Alex", "emotion": "Anxious", "goal": "Find shelter"}
        def respond_to_action(self, action): return {"emotion": "Relieved", "goal": None, "feedback": "Thank you."}

    uks = MockUKS()
    student = MockStudent()
    teacher = MockTeacher()

    trainer = PrinciplesAndEmpathyTrainer(student, teacher, uks)
    print("--- Initializing Principles ---")
    trainer.add_principles_to_graph()
    print("\n--- Running Interaction Loop ---")
    trainer.interaction_loop(max_steps=1)
    print("\n--- Example Conflict Resolution ---")
    example_conflict_resolution()