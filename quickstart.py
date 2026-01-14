"""
Quick Start Script for Neo4j RAG GC-MS System
Run this to test the complete workflow
"""

import os
from dotenv import load_dotenv
from crewai_gcms_prediction import GCMSPredictionCrew
from data_ingestion import GCMSDataIngestion, setup_database, ingest_sample_data
from rag_retriever import MolecularRAGRetriever
from neo4j_schema import get_neo4j_connection


def check_environment():
    """Verify environment variables are set"""
    load_dotenv()
    
    required_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD', 'OPENAI_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print("❌ Missing environment variables:")
        for var in missing:
            print(f"   - {var}")
        print("\nPlease copy .env.template to .env and fill in your credentials")
        return False
    
    print("✅ Environment variables configured")
    return True


def test_neo4j_connection():
    """Test Neo4j connection"""
    try:
        uri, username, password = get_neo4j_connection()
        retriever = MolecularRAGRetriever(uri, username, password)
        retriever.close()
        print("✅ Neo4j connection successful")
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False


def initialize_database():
    """Setup database and ingest sample data"""
    print("\n" + "="*60)
    print("INITIALIZING DATABASE")
    print("="*60)
    
    try:
        setup_database()
        ingest_sample_data()
        print("✅ Database initialized with sample data")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


def test_rag_retrieval():
    """Test RAG retrieval system"""
    print("\n" + "="*60)
    print("TESTING RAG RETRIEVAL")
    print("="*60)
    
    try:
        uri, username, password = get_neo4j_connection()
        retriever = MolecularRAGRetriever(uri, username, password)
        
        # Test with benzene
        test_smiles = "c1ccccc1"
        print(f"\nRetrieving context for: {test_smiles} (Benzene)")
        
        context = retriever.retrieve_prediction_context(
            query_smiles=test_smiles,
            column_type="HP-5MS",
            top_k=5
        )
        
        formatted = retriever.format_context_for_llm(context)
        print("\n" + formatted)
        
        retriever.close()
        print("\n✅ RAG retrieval successful")
        return True
        
    except Exception as e:
        print(f"❌ RAG retrieval failed: {e}")
        return False


def test_crewai_prediction():
    """Test full CrewAI prediction workflow"""
    print("\n" + "="*60)
    print("TESTING CREWAI PREDICTION WORKFLOW")
    print("="*60)
    
    try:
        crew = GCMSPredictionCrew()
        
        # Test with caffeine
        test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        print(f"\nPredicting retention time for: {test_smiles}")
        print("Molecule: Caffeine")
        print("Column: HP-5MS")
        print("Temperature: 40°C to 300°C at 10°C/min")
        print("\nThis may take 1-2 minutes as agents collaborate...\n")
        
        result = crew.predict_retention_time(
            smiles=test_smiles,
            column_type="HP-5MS",
            temperature_program="40°C to 300°C at 10°C/min"
        )
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(result['prediction_result'])
        
        print("\n✅ CrewAI prediction successful")
        return True
        
    except Exception as e:
        print(f"❌ CrewAI prediction failed: {e}")
        print(f"Error details: {str(e)}")
        return False


def main():
    """Run complete quick start workflow"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  Neo4j Aura RAG for GC-MS Retention Time Prediction      ║
    ║  CrewAI Multi-Agent System - Quick Start                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check environment
    print("\n[1/5] Checking environment configuration...")
    if not check_environment():
        return
    
    # Step 2: Test connection
    print("\n[2/5] Testing Neo4j connection...")
    if not test_neo4j_connection():
        return
    
    # Step 3: Initialize database
    print("\n[3/5] Initializing database...")
    if not initialize_database():
        return
    
    # Step 4: Test RAG retrieval
    print("\n[4/5] Testing RAG retrieval system...")
    if not test_rag_retrieval():
        return
    
    # Step 5: Test CrewAI prediction
    print("\n[5/5] Testing CrewAI prediction workflow...")
    test_crewai_prediction()
    
    print("""
    
    ╔═══════════════════════════════════════════════════════════╗
    ║  🎉 QUICK START COMPLETE!                                ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Your system is ready to use. Next steps:
    
    1. Ingest your own GC-MS data:
       from data_ingestion import GCMSDataIngestion
       ingestion = GCMSDataIngestion()
       ingestion.ingest_from_csv('your_data.csv')
    
    2. Make predictions:
       from crewai_gcms_prediction import GCMSPredictionCrew
       crew = GCMSPredictionCrew()
       result = crew.predict_retention_time('your_smiles')
    
    3. Query the knowledge graph:
       from rag_retriever import MolecularRAGRetriever
       retriever = MolecularRAGRetriever(uri, user, pass)
       context = retriever.retrieve_prediction_context('smiles')
    
    See README.md for detailed documentation.
    """)


if __name__ == "__main__":
    main()
