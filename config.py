"""
Configuration settings for Neo4j RAG GC-MS Prediction System
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class Neo4jConfig:
    """Neo4j database configuration"""
    # Vector embedding dimensions
    embedding_dim: int = 512
    
    # Similarity search parameters
    default_top_k: int = 10
    similarity_threshold: float = 0.7
    
    # Property range tolerance for searches
    property_tolerance: float = 0.15
    
    # Batch processing
    batch_size: int = 100
    max_retries: int = 3


@dataclass
class CrewAIConfig:
    """CrewAI agent configuration"""
    # LLM settings
    default_model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Agent behavior
    verbose: bool = True
    allow_delegation: bool = True
    
    # Workflow
    process_type: str = "sequential"  # or "hierarchical"


@dataclass
class GCMSConfig:
    """GC-MS specific configuration"""
    # Default column types
    default_column: str = "HP-5MS"
    supported_columns: List[str] = None
    
    # Temperature programs
    default_temp_program: str = "40°C to 300°C at 10°C/min"
    
    # Data validation
    min_rt: float = 0.5  # minutes
    max_rt: float = 60.0  # minutes
    
    # Feature selection
    top_features: List[str] = None
    
    def __post_init__(self):
        if self.supported_columns is None:
            self.supported_columns = [
                "HP-5MS",
                "DB-5",
                "DB-1",
                "DB-WAX",
                "RTX-5",
                "VF-5ms"
            ]
        
        if self.top_features is None:
            self.top_features = [
                'MolecularWeight',
                'LogP',
                'TPSA',
                'NumRotatableBonds',
                'NumHAcceptors',
                'NumHDonors',
                'NumAromaticRings',
                'FractionCSP3',
                'Chi0v',
                'Kappa1'
            ]


@dataclass
class DataIngestionConfig:
    """Data ingestion configuration"""
    # CSV column mappings
    smiles_column: str = "SMILES"
    rt_column: str = "RT"
    column_column: str = "Column"
    temp_program_column: str = "TempProgram"
    
    # Validation rules
    validate_smiles: bool = True
    skip_invalid: bool = True
    
    # Performance
    chunk_size: int = 1000
    parallel_processing: bool = False
    
    # Logging
    log_frequency: int = 100


@dataclass
class RAGConfig:
    """RAG retrieval configuration"""
    # Context retrieval
    max_similar_molecules: int = 10
    max_property_matches: int = 10
    max_column_records: int = 50
    
    # Ranking weights
    similarity_weight: float = 0.5
    property_weight: float = 0.3
    recency_weight: float = 0.2
    
    # Context formatting
    max_context_length: int = 4000  # tokens
    include_metadata: bool = True


class SystemConfig:
    """Main system configuration"""
    
    def __init__(self):
        self.neo4j = Neo4jConfig()
        self.crewai = CrewAIConfig()
        self.gcms = GCMSConfig()
        self.ingestion = DataIngestionConfig()
        self.rag = RAGConfig()
    
    def to_dict(self) -> Dict:
        """Export configuration as dictionary"""
        return {
            'neo4j': self.neo4j.__dict__,
            'crewai': self.crewai.__dict__,
            'gcms': self.gcms.__dict__,
            'ingestion': self.ingestion.__dict__,
            'rag': self.rag.__dict__
        }
    
    def update_from_dict(self, config_dict: Dict):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)


# Global configuration instance
config = SystemConfig()


# Example usage
if __name__ == "__main__":
    import json
    
    # Display current configuration
    print("Current System Configuration:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # Example: Update a configuration value
    config.neo4j.similarity_threshold = 0.8
    config.gcms.default_column = "DB-5"
    
    print("\nUpdated Configuration:")
    print(f"Similarity threshold: {config.neo4j.similarity_threshold}")
    print(f"Default column: {config.gcms.default_column}")
