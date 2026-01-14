"""
RAG (Retrieval-Augmented Generation) System for Neo4j GC-MS Knowledge Graph
Provides context retrieval for CrewAI agents to make informed predictions
"""

from typing import List, Dict, Optional, Tuple
from neo4j import GraphDatabase
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import json


class MolecularRAGRetriever:
    """
    RAG retriever for molecular knowledge graph
    Provides relevant context for retention time predictions
    """

    def __init__(self, neo4j_uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(
            neo4j_uri, auth=(username, password))
        self.embedding_dim = 512

    def close(self):
        self.driver.close()

    def generate_molecular_embedding(self, smiles: str) -> List[float]:
        """Generate molecular fingerprint embedding using RDKit"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0] * self.embedding_dim

        # Morgan fingerprint (ECFP4)
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 2, nBits=self.embedding_dim)
        return [float(x) for x in fp]

    def extract_molecular_properties(self, smiles: str) -> Dict:
        """Extract RDKit molecular descriptors"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        return {
            'molecular_weight': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_h_acceptors': Descriptors.NumHAcceptors(mol),
            'num_h_donors': Descriptors.NumHDonors(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
            'fraction_csp3': Descriptors.FractionCSP3(mol)
        }

    def vector_similarity_search(self, query_smiles: str, top_k: int = 5) -> List[Dict]:
        """Find similar molecules using vector similarity"""
        query_embedding = self.generate_molecular_embedding(query_smiles)

        with self.driver.session() as session:
            result = session.run("""
                CALL db.index.vector.queryNodes('molecule_embedding', $k, $query_vector)
                YIELD node, score
                OPTIONAL MATCH (node)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column)
                RETURN node.smiles as smiles,
                       node.molecular_weight as mw,
                       node.logp as logp,
                       score,
                       collect({rt: r.rt_minutes, column: c.type, temp: r.temperature_program}) as measurements
                ORDER BY score DESC
            """, k=top_k, query_vector=query_embedding)

            return [record.data() for record in result]

    def property_range_search(self, target_properties: Dict,
                              tolerance: float = 0.1) -> List[Dict]:
        """Find molecules with similar properties within tolerance range"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Molecule)
                
                WHERE m.molecular_weight >= $mw_low AND m.molecular_weight <= $mw_high
                    AND m.logp >= $logp_low AND m.logp <= $logp_high
                    AND m.tpsa >= $tpsa_low AND m.tpsa <= $tpsa_high
                OPTIONAL MATCH (m)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column)
                RETURN m.smiles as smiles,
                       m.molecular_weight as mw,
                       m.logp as logp,
                       m.tpsa as tpsa,
                       collect({rt: r.rt_minutes, column: c.type}) as measurements
                LIMIT 10
            """,
                                 mw_low=target_properties['molecular_weight'] * (
                                     1 - tolerance),
                                 mw_high=target_properties['molecular_weight'] * (
                                     1 + tolerance),
                                 logp_low=target_properties['logp'] -
                                 tolerance * 2,
                                 logp_high=target_properties['logp'] +
                                 tolerance * 2,
                                 tpsa_low=target_properties['tpsa'] *
                                 (1 - tolerance),
                                 tpsa_high=target_properties['tpsa'] * (1 + tolerance))

            return [record.data() for record in result]

    def feature_based_retrieval(self, smiles: str, top_features: List[str]) -> Dict:
        """Retrieve molecules with similar key features"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Molecule {smiles: $smiles})-[hf:HAS_FEATURE]->(f:MolecularFeature)
                WHERE f.name IN $top_features
                WITH m, collect({feature: f.name, value: hf.value}) as query_features
                
                MATCH (other:Molecule)-[ohf:HAS_FEATURE]->(of:MolecularFeature)
                WHERE other.smiles <> $smiles 
                  AND of.name IN $top_features
                WITH other, 
                     collect({feature: of.name, value: ohf.value}) as other_features,
                     query_features
                
                OPTIONAL MATCH (other)-[:MEASURED_ON]->(r:RetentionTime)
                RETURN other.smiles as smiles,
                       other_features as features,
                       collect(r.rt_minutes) as retention_times
                LIMIT 10
            """, smiles=smiles, top_features=top_features)

            return [record.data() for record in result]

    def get_column_specific_data(self, column_type: str,
                                 property_range: Dict = None) -> List[Dict]:
        """Retrieve all RT data for a specific column type"""
        with self.driver.session() as session:
            query = """
                MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column {type: $column_type})
            """

            if property_range:
                query += """
                WHERE m.molecular_weight >=  $mw_low AND m.molecular_weight <= $mw_high
                """

            query += """
                RETURN m.smiles as smiles,
                       m.molecular_weight as mw,
                       m.logp as logp,
                       r.rt_minutes as rt,
                       r.temperature_program as temp_program
                ORDER BY r.rt_minutes
            """

            params = {'column_type': column_type}
            if property_range:
                params.update({
                    'mw_low': property_range.get('mw_min', 0),
                    'mw_high': property_range.get('mw_max', 1000)
                })

            result = session.run(query, **params)
            return [record.data() for record in result]

    def retrieve_prediction_context(self, query_smiles: str,
                                    column_type: str = "HP-5MS",
                                    top_k: int = 10) -> Dict:
        """
        Main RAG retrieval method for CrewAI agents
        Returns comprehensive context for RT prediction
        """
        # Extract query molecule properties
        query_props = self.extract_molecular_properties(query_smiles)

        # Vector similarity search
        similar_molecules = self.vector_similarity_search(
            query_smiles, top_k=top_k)

        # Property-based search
        property_matches = self.property_range_search(
            query_props, tolerance=0.15)

        # Column-specific historical data
        column_data = self.get_column_specific_data(
            column_type,
            property_range={'mw_min': query_props['molecular_weight'] - 50,
                            'mw_max': query_props['molecular_weight'] + 50}
        )

        # Compile context
        context = {
            'query_molecule': {
                'smiles': query_smiles,
                'properties': query_props
            },
            'similar_molecules': similar_molecules,
            'property_matches': property_matches,
            'column_specific_data': column_data,
            'statistics': {
                'num_similar': len(similar_molecules),
                'num_property_matches': len(property_matches),
                'num_column_records': len(column_data),
                'avg_rt_similar': np.mean([
                    m['rt'] for m in similar_molecules[0].get('measurements', [])
                    if m.get('rt') is not None
                ]) if similar_molecules else None
            }
        }

        return context

    def format_context_for_llm(self, context: Dict) -> str:
        """Format retrieved context into natural language for LLM agents"""
        query_props = context['query_molecule']['properties']
        stats = context['statistics']

        formatted = f"""
Molecular Analysis Context for GC-MS Retention Time Prediction
================================================================

Query Molecule: {context['query_molecule']['smiles']}

Molecular Properties:
- Molecular Weight: {query_props.get('molecular_weight', 'N/A'):.2f} g/mol
- LogP: {query_props.get('logp', 'N/A'):.2f}
- TPSA: {query_props.get('tpsa', 'N/A'):.2f} Ų
- Rotatable Bonds: {query_props.get('num_rotatable_bonds', 'N/A')}
- H-Bond Acceptors: {query_props.get('num_h_acceptors', 'N/A')}
- H-Bond Donors: {query_props.get('num_h_donors', 'N/A')}

Retrieved Similar Molecules: {stats['num_similar']}
"""

        if context['similar_molecules']:
            formatted += "\nTop Similar Molecules with Known Retention Times:\n"
            for i, mol in enumerate(context['similar_molecules'][:5], 1):
                measurements = mol.get('measurements', [])
                if measurements:
                    rt_values = [m['rt'] for m in measurements if m.get('rt')]
                    if rt_values:
                        formatted += f"{i}. SMILES: {mol['smiles']}\n"
                        formatted += f"   Similarity: {mol['score']:.3f}\n"
                        formatted += f"   MW: {mol['mw']:.2f}, LogP: {mol['logp']:.2f}\n"
                        formatted += f"   Measured RT: {', '.join([f'{rt:.2f} min' for rt in rt_values])}\n\n"

        if stats['avg_rt_similar']:
            formatted += f"\nAverage RT of Similar Molecules: {stats['avg_rt_similar']:.2f} minutes\n"

        formatted += f"\nTotal Historical Records Retrieved: {stats['num_column_records']}\n"

        return formatted
