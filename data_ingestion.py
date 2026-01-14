"""
Data Ingestion Pipeline for Neo4j GC-MS Knowledge Graph
Loads molecular data, features, and retention times into the graph database
"""
import json
from typing import Dict, List
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import Chem
from rag_retriever import MolecularRAGRetriever
from neo4j_schema import Neo4jGCMSSchema, get_neo4j_connection
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


class GCMSDataIngestion:
    """Pipeline for ingesting GC-MS experimental data into Neo4j"""

    def __init__(self):
        uri, username, password = get_neo4j_connection()
        self.schema = Neo4jGCMSSchema(uri, username, password)
        self.retriever = MolecularRAGRetriever(uri, username, password)

    def extract_all_molecular_features(self, smiles: str) -> Dict[str, float]:
        """Extract comprehensive RDKit molecular descriptors"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}

        features = {
            # Basic properties
            'MolecularWeight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),

            # Structural features
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),

            # Complexity
            'FractionCSP3': Descriptors.FractionCSP3(mol),

            # Electronic properties
            'MolMR': Descriptors.MolMR(mol),  # Molar refractivity
            'BalabanJ': Descriptors.BalabanJ(mol),

            # Topological indices
            'Chi0v': Descriptors.Chi0v(mol),
            'Chi1v': Descriptors.Chi1v(mol),
            'Kappa1': Descriptors.Kappa1(mol),
            'Kappa2': Descriptors.Kappa2(mol),

            # Surface area and volume
            'LabuteASA': Descriptors.LabuteASA(mol),
            'PEOE_VSA1': Descriptors.PEOE_VSA1(mol),
            'PEOE_VSA2': Descriptors.PEOE_VSA2(mol),

            # Charge
            'MaxPartialCharge': Descriptors.MaxPartialCharge(mol),
            'MinPartialCharge': Descriptors.MinPartialCharge(mol),
        }

        # Add optional descriptors if available in this RDKit version
        try:
            features['NumBridgeheadAtoms'] = Descriptors.NumBridgeheadAtoms(
                mol)
        except AttributeError:
            pass

        try:
            features['NumSpiroAtoms'] = Descriptors.NumSpiroAtoms(mol)
        except AttributeError:
            pass

        return features

    def ingest_from_csv(self, csv_path: str,
                        smiles_col: str = 'SMILES',
                        rt_col: str = 'RT',
                        column_col: str = 'Column',
                        temp_program_col: str = 'TempProgram'):
        """Ingest GC-MS data from CSV file"""

        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        print(f"Processing {len(df)} compounds...")

        for idx, row in df.iterrows():
            smiles = row[smiles_col]

            if pd.isna(smiles):
                continue

            try:
                # Extract molecular properties
                properties = self.retriever.extract_molecular_properties(
                    smiles)

                # Generate embedding
                embedding = self.retriever.generate_molecular_embedding(smiles)

                # Add molecule to graph
                self.schema.add_molecule(smiles, properties, embedding)

                # Extract and add all molecular features
                features = self.extract_all_molecular_features(smiles)
                self.schema.add_molecular_features(smiles, features)

                # Add retention time if available
                if not pd.isna(row.get(rt_col)):
                    self.schema.add_retention_time(
                        smiles=smiles,
                        rt_minutes=float(row[rt_col]),
                        column_type=row.get(column_col, 'Unknown'),
                        temperature_program=row.get(
                            temp_program_col, 'Unknown'),
                        method_details={'flow_rate': row.get('FlowRate', 1.0)}
                    )

                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1} compounds...")

            except Exception as e:
                print(f"Error processing {smiles}: {e}")
                continue

        print("Data ingestion complete!")

    def ingest_from_dataframe(self, df: pd.DataFrame,
                              smiles_col: str = 'SMILES',
                              rt_col: str = 'RT',
                              column_col: str = 'Column',
                              temp_program_col: str = 'TempProgram'):
        """Ingest GC-MS data from pandas DataFrame"""

        print(f"Processing {len(df)} compounds from DataFrame...")

        for idx, row in df.iterrows():
            smiles = row[smiles_col]

            if pd.isna(smiles):
                continue

            try:
                # Extract molecular properties
                properties = self.retriever.extract_molecular_properties(
                    smiles)

                # Generate embedding
                embedding = self.retriever.generate_molecular_embedding(smiles)

                # Add molecule to graph
                self.schema.add_molecule(smiles, properties, embedding)

                # Extract and add all molecular features
                features = self.extract_all_molecular_features(smiles)
                self.schema.add_molecular_features(smiles, features)

                # Add retention time if available
                if not pd.isna(row.get(rt_col)):
                    self.schema.add_retention_time(
                        smiles=smiles,
                        rt_minutes=float(row[rt_col]),
                        column_type=row.get(column_col, 'HP-5MS'),
                        temperature_program=row.get(
                            temp_program_col, '40-300C'),
                        method_details={'flow_rate': row.get('FlowRate', 1.0)}
                    )

                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1} compounds...")

            except Exception as e:
                print(f"Error processing {smiles}: {e}")
                continue

        print("DataFrame ingestion complete!")

    def create_sample_dataset(self) -> pd.DataFrame:
        """Create a sample dataset for testing"""
        data = {
            'SMILES': [
                'CCO',  # Ethanol
                'CC(C)O',  # Isopropanol
                'CCCCCO',  # Pentanol
                'c1ccccc1',  # Benzene
                'Cc1ccccc1',  # Toluene
                'c1ccc(C)cc1C',  # Xylene
                'CC(=O)O',  # Acetic acid
                'CCCCCCCC',  # Octane
                'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # Caffeine
                'CC(C)Cc1ccc(cc1)C(C)C(=O)O',  # Ibuprofen
            ],
            'RT': [2.3, 2.8, 5.6, 4.2, 6.1, 8.3, 3.5, 7.8, 12.4, 15.2],
            'Column': ['HP-5MS'] * 10,
            'TempProgram': ['40-300C'] * 10,
            'FlowRate': [1.0] * 10
        }

        return pd.DataFrame(data)

    def build_similarity_graph(self, threshold: float = 0.8):
        """Create similarity relationships after data ingestion"""
        print("Building similarity relationships...")
        self.schema.create_similarity_relationships(
            similarity_threshold=threshold)
        print("Similarity graph complete!")

    def close(self):
        """Clean up connections"""
        self.schema.close()
        self.retriever.close()


def setup_database():
    """Initialize Neo4j database with schema"""
    print("Setting up Neo4j database schema...")
    uri, username, password = get_neo4j_connection()
    schema = Neo4jGCMSSchema(uri, username, password)
    schema.create_schema()
    schema.close()
    print("Schema setup complete!")


def ingest_sample_data():
    """Ingest sample GC-MS data"""
    ingestion = GCMSDataIngestion()

    # Create and ingest sample data
    sample_df = ingestion.create_sample_dataset()
    print("\nSample Dataset:")
    print(sample_df)

    print("\nIngesting sample data...")
    ingestion.ingest_from_dataframe(sample_df)

    # Build similarity relationships
    ingestion.build_similarity_graph(threshold=0.7)

    ingestion.close()
    print("\nSample data ingestion complete!")


def main():
    """Main execution"""
    # Setup database
    setup_database()

    # Ingest sample data
    ingest_sample_data()

    print("\n" + "="*80)
    print("Neo4j GC-MS Knowledge Graph Ready!")
    print("="*80)
    print("\nYou can now:")
    print("1. Query the graph using RAG retrieval")
    print("2. Run CrewAI predictions")
    print("3. Ingest your own GC-MS experimental data")
    print("\nTo ingest your own CSV data:")
    print("ingestion = GCMSDataIngestion()")
    print("ingestion.ingest_from_csv('your_data.csv')")


if __name__ == "__main__":
    main()
