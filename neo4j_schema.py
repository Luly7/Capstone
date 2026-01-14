"""
from dotenv import load_dotenv

load_dotenv()
Neo4j Schema and Data Model for GC-MS Retention Time Prediction
Defines the graph structure for molecules, features, and retention times
"""

from neo4j import GraphDatabase
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class Neo4jGCMSSchema:
    """
    Graph Schema:
    - Molecule nodes: SMILES, molecular properties
    - MolecularFeature nodes: RDKit descriptors
    - RetentionTime nodes: experimental RT data
    - Column nodes: GC column specifications
    - Relationships: HAS_FEATURE, MEASURED_ON, SIMILAR_TO
    """

    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def create_schema(self):
        """Create indexes and constraints for optimal query performance"""
        with self.driver.session() as session:
            # Constraints
            session.run("""
                CREATE CONSTRAINT molecule_smiles IF NOT EXISTS
                FOR (m:Molecule) REQUIRE m.smiles IS UNIQUE
            """)

            session.run("""
                CREATE CONSTRAINT feature_name IF NOT EXISTS
                FOR (f:MolecularFeature) REQUIRE f.name IS UNIQUE
            """)

            # Indexes for fast retrieval
            session.run("""
                CREATE INDEX molecule_mw IF NOT EXISTS
                FOR (m:Molecule) ON (m.molecular_weight)
            """)

            session.run("""
                CREATE INDEX molecule_logp IF NOT EXISTS
                FOR (m:Molecule) ON (m.logp)
            """)

            session.run("""
                CREATE INDEX rt_value IF NOT EXISTS
                FOR (r:RetentionTime) ON (r.rt_minutes)
            """)

            session.run("""
                CREATE VECTOR INDEX molecule_embedding IF NOT EXISTS
                FOR (m:Molecule) ON (m.embedding)
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 512,
                    `vector.similarity_function`: 'cosine'
                }}
            """)

            print("Schema created successfully")

    def add_molecule(self, smiles: str, properties: Dict, embedding: List[float]):
        """Add a molecule node with properties and vector embedding"""
        with self.driver.session() as session:
            result = session.run("""
                MERGE (m:Molecule {smiles: $smiles})
                SET m.molecular_weight = $mw,
                    m.logp = $logp,
                    m.tpsa = $tpsa,
                    m.num_rotatable_bonds = $rotatable,
                    m.num_h_acceptors = $h_acceptors,
                    m.num_h_donors = $h_donors,
                    m.num_aromatic_rings = $aromatic_rings,
                    m.embedding = $embedding,
                    m.updated_at = datetime()
                RETURN m
            """, smiles=smiles,
                                 mw=properties.get('molecular_weight'),
                                 logp=properties.get('logp'),
                                 tpsa=properties.get('tpsa'),
                                 rotatable=properties.get(
                                     'num_rotatable_bonds'),
                                 h_acceptors=properties.get('num_h_acceptors'),
                                 h_donors=properties.get('num_h_donors'),
                                 aromatic_rings=properties.get(
                                     'num_aromatic_rings'),
                                 embedding=embedding)
            return result.single()

    def add_molecular_features(self, smiles: str, features: Dict[str, float]):
        """Add molecular feature nodes and relationships"""
        with self.driver.session() as session:
            for feature_name, feature_value in features.items():
                session.run("""
                    MATCH (m:Molecule {smiles: $smiles})
                    MERGE (f:MolecularFeature {name: $feature_name})
                    MERGE (m)-[r:HAS_FEATURE]->(f)
                    SET r.value = $feature_value,
                        r.updated_at = datetime()
                """, smiles=smiles, feature_name=feature_name, feature_value=feature_value)

    def add_retention_time(self, smiles: str, rt_minutes: float,
                           column_type: str, temperature_program: str,
                           method_details: Dict):
        """Add retention time measurement for a molecule"""
        with self.driver.session() as session:
            session.run("""
                MATCH (m:Molecule {smiles: $smiles})
                MERGE (c:Column {type: $column_type})
                CREATE (r:RetentionTime {
                    rt_minutes: $rt_minutes,
                    temperature_program: $temp_program,
                    flow_rate: $flow_rate,
                    measured_date: datetime()
                })
                CREATE (m)-[:MEASURED_ON]->(r)
                CREATE (r)-[:USING_COLUMN]->(c)
            """, smiles=smiles, rt_minutes=rt_minutes,
                        column_type=column_type, temp_program=temperature_program,
                        flow_rate=method_details.get('flow_rate'))

    def create_similarity_relationships(self, similarity_threshold: float = 0.8):
        """Create SIMILAR_TO relationships based on molecular similarity"""
        with self.driver.session() as session:
            session.run("""
                MATCH (m1:Molecule), (m2:Molecule)
                WHERE id(m1) < id(m2)
                WITH m1, m2,
                     gds.similarity.cosine(m1.embedding, m2.embedding) AS similarity
                WHERE similarity > $threshold
                MERGE (m1)-[s:SIMILAR_TO]->(m2)
                SET s.similarity_score = similarity
            """, threshold=similarity_threshold)
            print(
                f"Created similarity relationships with threshold {similarity_threshold}")

    def get_molecule_context(self, smiles: str, limit: int = 10) -> List[Dict]:
        """Retrieve full context for a molecule including similar molecules"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Molecule {smiles: $smiles})
                OPTIONAL MATCH (m)-[hf:HAS_FEATURE]->(f:MolecularFeature)
                OPTIONAL MATCH (m)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column)
                OPTIONAL MATCH (m)-[sim:SIMILAR_TO]-(similar:Molecule)
                OPTIONAL MATCH (similar)-[:MEASURED_ON]->(similar_rt:RetentionTime)
                RETURN m, 
                       collect(DISTINCT {feature: f.name, value: hf.value}) as features,
                       collect(DISTINCT {rt: r.rt_minutes, column: c.type}) as measurements,
                       collect(DISTINCT {
                           smiles: similar.smiles, 
                           similarity: sim.similarity_score,
                           rt: similar_rt.rt_minutes
                       })[0..$limit] as similar_molecules
            """, smiles=smiles, limit=limit)
            return [record.data() for record in result]


# Environment setup
def get_neo4j_connection():
    """Get Neo4j Aura connection details from environment"""
    uri = os.getenv("NEO4J_URI", "neo4j+s://b97b9197.databases.neo4j.io")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv(
        "NEO4J_PASSWORD", "BBs3f8U81f7gHRZdbZBao-6F94jxeN9gkoxib0M4LQQ")
    return uri, username, password
