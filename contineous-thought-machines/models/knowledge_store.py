from __future__ import annotations
from neo4j import GraphDatabase, Driver

# --- Data Classes ---
# These classes are simple containers for data returned from Neo4j.
# A "Relationship" is now a node itself, allowing for a more flexible graph.

class Thing:
    """Represents a 'Thing' node in the knowledge graph."""
    def __init__(self, label: str, value: str | None = None, element_id: str | None = None):
        self.label = label
        self.value = value
        self.element_id = element_id

    def __str__(self):
        return f"Thing(label='{self.label}')"

class Relationship:
    """Represents a 'Relationship' node in the knowledge graph (Reification)."""
    def __init__(self, rel_type: Thing, source: Thing, target: Thing, weight: float = 1.0, element_id: str | None = None):
        self.rel_type = rel_type
        self.source = source
        self.target = target
        self.weight = weight
        self.element_id = element_id

    def __str__(self):
        return f"({self.source.label})-[{self.rel_type.label}]->({self.target.label})"

# --- UKS: The Neo4j-backed Knowledge Store ---

class UKS:
    """
    The Universal Knowledge Store, using Neo4j as the backend.
    This implementation uses a reified model for relationships, treating them
    as nodes to allow for relationships between relationships (clauses).
    """
    def __init__(self, uri, user, password):
        self._driver: Driver = GraphDatabase.driver(uri, auth=(user, password))
        self.create_constraints()
        self.create_initial_structure()

    def close(self):
        """Closes the database connection."""
        self._driver.close()

    def create_constraints(self):
        """Ensures that Thing labels are unique for data integrity and performance."""
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT unique_thing_label IF NOT EXISTS FOR (t:Thing) REQUIRE t.label IS UNIQUE")

    def create_initial_structure(self):
        """Creates the foundational 'Thing' node if it doesn't exist."""
        self.get_or_add_thing("Thing")

    def get_or_add_thing(self, label: str, value: str | None = None) -> Thing:
        """Retrieves a Thing by its label, creating it if it doesn't exist."""
        with self._driver.session() as session:
            result = session.execute_write(self._get_or_add_thing_tx, label, value)
            node = result['t']
            return Thing(label=node['label'], value=node.get('value'), element_id=node.element_id)

    @staticmethod
    def _get_or_add_thing_tx(tx, label, value):
        query = (
            "MERGE (t:Thing {label: $label}) "
            "ON CREATE SET t.value = $value, t.created = timestamp() "
            "RETURN t"
        )
        result = tx.run(query, label=label, value=value)
        return result.single()

    def add_statement(self, source_label: str, rel_type_label: str, target_label: str, weight: float = 1.0) -> Relationship:
        """
        Creates a reified relationship: (Source)-[:SOURCE]->(Relationship)-[:TARGET]->(Target).
        This makes the relationship a first-class citizen in the graph.
        """
        with self._driver.session() as session:
            result = session.execute_write(
                self._add_statement_tx, source_label, rel_type_label, target_label, weight
            )
            s_node, r_node, t_node = result['s'], result['r'], result['t']
            
            source_thing = Thing(label=s_node['label'], element_id=s_node.element_id)
            target_thing = Thing(label=t_node['label'], element_id=t_node.element_id)
            # The relationship's type is itself a Thing
            rel_type_thing = self.get_or_add_thing(r_node['type'])

            return Relationship(
                rel_type=rel_type_thing,
                source=source_thing,
                target=target_thing,
                weight=r_node['weight'],
                element_id=r_node.element_id
            )

    @staticmethod
    def _add_statement_tx(tx, source_label, rel_type_label, target_label, weight):
        # Ensure source, target, and relationship type 'Things' exist
        tx.run("MERGE (:Thing {label: $source_label})", source_label=source_label)
        tx.run("MERGE (:Thing {label: $target_label})", target_label=target_label)
        tx.run("MERGE (:Thing {label: $rel_type_label})", rel_type_label=rel_type_label)

        query = (
            "MATCH (s:Thing {label: $source_label}) "
            "MATCH (t:Thing {label: $target_label}) "
            # Merge the relationship node itself, uniquely identified by its components
            "MERGE (s)-[:SOURCE]->(r:Relationship {type: $rel_type_label, source_label: $source_label, target_label: $target_label})-[:TARGET]->(t) "
            "ON CREATE SET r.weight = $weight, r.created = timestamp(), r.last_updated = timestamp() "
            "ON MATCH SET r.weight = $weight, r.last_updated = timestamp() "
            "RETURN s, r, t"
        )
        result = tx.run(query, source_label=source_label, target_label=target_label, rel_type_label=rel_type_label, weight=weight)
        return result.single()

    def add_clause(self, base_relationship: Relationship, clause_rel_type_label: str, clause_relationship: Relationship):
        """
        Creates a conditional link between two Relationship nodes.
        This is now a simple, standard relationship creation because relationships are nodes.
        """
        with self._driver.session() as session:
            session.execute_write(
                self._add_clause_tx,
                base_relationship.element_id,
                clause_rel_type_label,
                clause_relationship.element_id
            )

    @staticmethod
    def _add_clause_tx(tx, base_rel_element_id, clause_rel_type_label, clause_rel_element_id):
        # Ensure the clause relationship type exists as a Thing
        tx.run("MERGE (:Thing {label: $clause_rel_type_label})", clause_rel_type_label=clause_rel_type_label)

        query = (
            "MATCH (base_r:Relationship) WHERE elementId(base_r) = $base_rel_element_id "
            "MATCH (clause_r:Relationship) WHERE elementId(clause_r) = $clause_rel_element_id "
            "MERGE (base_r)-[c:HAS_CLAUSE {type: $clause_rel_type_label}]->(clause_r) "
            "ON CREATE SET c.created = timestamp(), c.last_updated = timestamp() "
            "ON MATCH SET c.last_updated = timestamp()"
        )
        tx.run(query, base_rel_element_id=base_rel_element_id, clause_rel_type_label=clause_rel_type_label, clause_rel_element_id=clause_rel_element_id)

    def add_action_outcome(self, action_label: str, outcome_label: str, valence: str, weight: float = 1.0) -> Relationship:
        """
        A specialized method to create a relationship between an action and an outcome.
        This is a key component for learning from behavior.
        """
        with self._driver.session() as session:
            result = session.execute_write(
                self._add_action_outcome_tx, action_label, outcome_label, valence, weight
            )
            s_node, r_node, t_node = result['s'], result['r'], result['t']

            source_thing = Thing(label=s_node['label'], element_id=s_node.element_id)
            target_thing = Thing(label=t_node['label'], value=t_node.get('valence'), element_id=t_node.element_id)
            rel_type_thing = self.get_or_add_thing(r_node['type'])

            return Relationship(
                rel_type=rel_type_thing,
                source=source_thing,
                target=target_thing,
                weight=r_node['weight'],
                element_id=r_node.element_id
            )

    @staticmethod
    def _add_action_outcome_tx(tx, action_label, outcome_label, valence, weight):
        # Ensure the action, outcome, and relationship type 'Things' exist
        tx.run("MERGE (:Thing:Action {label: $action_label})", action_label=action_label)
        tx.run("MERGE (o:Thing:Outcome {label: $outcome_label}) SET o.valence = $valence", outcome_label=outcome_label, valence=valence)
        tx.run("MERGE (:Thing {label: 'LEADS_TO'})")

        query = (
            "MATCH (s:Thing:Action {label: $action_label}) "
            "MATCH (t:Thing:Outcome {label: $outcome_label}) "
            "MERGE (s)-[:SOURCE]->(r:Relationship {type: 'LEADS_TO', source_label: $action_label, target_label: $outcome_label})-[:TARGET]->(t) "
            "ON CREATE SET r.weight = $weight, r.created = timestamp(), r.last_updated = timestamp() "
            "ON MATCH SET r.weight = $weight, r.last_updated = timestamp() "
            "RETURN s, r, t"
        )
        result = tx.run(query, action_label=action_label, outcome_label=outcome_label, weight=weight)
        return result.single()

    def find_outcomes_for_state(self, state_label: str) -> list[dict]:
        """
        Queries the graph to find actions and outcomes conditionally linked to a given state.
        This is the "read" part of the learning loop.
        """
        with self._driver.session() as session:
            result = session.execute_read(self._find_outcomes_for_state_tx, state_label)
            return result

    @staticmethod
    def _find_outcomes_for_state_tx(tx, state_label):
        query = (
            "MATCH (state:Thing {label: $state_label})<-[:TARGET]-(base_r:Relationship {type: 'has_pattern'})-[:HAS_CLAUSE]->(clause_r:Relationship)-[:TARGET]->(outcome:Thing:Outcome) "
            "MATCH (clause_r)-[:SOURCE]->(action:Thing:Action) "
            "RETURN action.label AS action, outcome.label AS outcome, outcome.valence AS valence, clause_r.weight AS weight"
        )
        results = tx.run(query, state_label=state_label)
        return [record.data() for record in results]