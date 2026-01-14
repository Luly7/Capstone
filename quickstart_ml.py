"""
ML-Enhanced Quick Start Script for Neo4j RAG GC-MS System
Tests complete workflow including machine learning models
"""

import os
from dotenv import load_dotenv
from crewai_ml_prediction import MLEnhancedGCMSCrew
from ml_model_trainer import GCMSMLTrainer, train_all_models
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


def train_ml_models():
    """Train machine learning models on Neo4j data"""
    print("\n" + "="*60)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*60)
    
    try:
        # Check if models already exist
        if os.path.exists("models") and os.listdir("models"):
            print("✅ ML models already trained (models/ directory exists)")
            print("   To retrain, delete the models/ directory and run again")
            return True
        
        print("Training Random Forest and Gradient Boosting models...")
        print("This will take 30-60 seconds...\n")
        
        results = train_all_models()
        
        if results:
            print("\n✅ ML models trained successfully")
            print(f"   Models saved in: ./models/")
            return True
        else:
            print("⚠️  Not enough training data for ML models")
            print("   Need at least 10 molecules with retention times")
            return False
            
    except Exception as e:
        print(f"❌ ML model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_prediction():
    """Test ML-based prediction"""
    print("\n" + "="*60)
    print("TESTING ML PREDICTIONS")
    print("="*60)
    
    try:
        trainer = GCMSMLTrainer()
        
        # Load models
        metadata = trainer.load_models("HP-5MS")
        print(f"✅ Loaded models: {list(trainer.trained_models.keys())}")
        
        # Test prediction on benzene
        test_smiles = "c1ccccc1"
        print(f"\nTesting ML prediction for: {test_smiles} (Benzene)")
        
        # Get features
        uri, username, password = get_neo4j_connection()
        retriever = MolecularRAGRetriever(uri, username, password)
        features = retriever.extract_molecular_properties(test_smiles)
        
        # Add more features
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(test_smiles)
        if mol:
            features['Chi0v'] = Descriptors.Chi0v(mol)
            features['Chi1v'] = Descriptors.Chi1v(mol)
            features['Kappa1'] = Descriptors.Kappa1(mol)
            features['Kappa2'] = Descriptors.Kappa2(mol)
        
        retriever.close()
        
        # Predict
        rf_pred = trainer.predict_with_model(features, 'random_forest')
        gb_pred = trainer.predict_with_model(features, 'gradient_boosting')
        
        print(f"\nRandom Forest:     {rf_pred['predicted_rt']:.2f} min "
              f"(95% CI: [{rf_pred['lower_bound']:.2f}, {rf_pred['upper_bound']:.2f}])")
        print(f"Gradient Boosting: {gb_pred['predicted_rt']:.2f} min "
              f"(95% CI: [{gb_pred['lower_bound']:.2f}, {gb_pred['upper_bound']:.2f}])")
        print(f"Ensemble Average:  {(rf_pred['predicted_rt'] + gb_pred['predicted_rt'])/2:.2f} min")
        
        trainer.close()
        print("\n✅ ML prediction successful")
        return True
        
    except Exception as e:
        print(f"❌ ML prediction failed: {e}")
        import traceback
        traceback.print_exc()
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
        print("\n" + formatted[:500] + "...")  # Show first 500 chars
        
        retriever.close()
        print("\n✅ RAG retrieval successful")
        return True
        
    except Exception as e:
        print(f"❌ RAG retrieval failed: {e}")
        return False


def test_ml_enhanced_crewai():
    """Test full ML-enhanced CrewAI prediction workflow"""
    print("\n" + "="*60)
    print("TESTING ML-ENHANCED CREWAI WORKFLOW")
    print("="*60)
    
    try:
        crew = MLEnhancedGCMSCrew()
        
        # Test with caffeine
        test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        print(f"\nPredicting retention time for: {test_smiles}")
        print("Molecule: Caffeine")
        print("Column: HP-5MS")
        print("Temperature: 40°C to 300°C at 10°C/min")
        print("\nThis combines ML models + RAG + Multi-agent reasoning...")
        print("This may take 2-3 minutes as 6 agents collaborate...\n")
        
        result = crew.predict_retention_time(
            smiles=test_smiles,
            column_type="HP-5MS",
            temperature_program="40°C to 300°C at 10°C/min"
        )
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(result['prediction_result'])
        
        print("\n✅ ML-enhanced CrewAI prediction successful")
        return True
        
    except Exception as e:
        print(f"❌ ML-enhanced CrewAI prediction failed: {e}")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run complete ML-enhanced quick start workflow"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║  Neo4j Aura RAG + Machine Learning for GC-MS             ║
    ║  ML-Enhanced Multi-Agent System - Quick Start            ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check environment
    print("\n[1/7] Checking environment configuration...")
    if not check_environment():
        return
    
    # Step 2: Test connection
    print("\n[2/7] Testing Neo4j connection...")
    if not test_neo4j_connection():
        return
    
    # Step 3: Initialize database
    print("\n[3/7] Initializing database...")
    if not initialize_database():
        return
    
    # Step 4: Train ML models (NEW!)
    print("\n[4/7] Training machine learning models...")
    ml_trained = train_ml_models()
    
    # Step 5: Test ML predictions
    if ml_trained:
        print("\n[5/7] Testing ML prediction system...")
        test_ml_prediction()
    else:
        print("\n[5/7] Skipping ML tests (not enough training data)")
    
    # Step 6: Test RAG retrieval
    print("\n[6/7] Testing RAG retrieval system...")
    test_rag_retrieval()
    
    # Step 7: Test ML-enhanced CrewAI
    if ml_trained:
        print("\n[7/7] Testing ML-enhanced CrewAI workflow...")
        test_ml_enhanced_crewai()
    else:
        print("\n[7/7] Skipping ML-enhanced CrewAI (models not trained)")
        print("\n⚠️  To use ML features, load more data and run again:")
        print("   python data_ingestion.py  # Add your GC-MS data")
        print("   python quickstart_ml.py    # Re-run with ML")
    
    print("""
    
    ╔═══════════════════════════════════════════════════════════╗
    ║  🎉 ML-ENHANCED QUICK START COMPLETE!                    ║
    ╚═══════════════════════════════════════════════════════════╝
    
    Your system is ready with:
    
    ✅ Neo4j graph database with molecular knowledge
    ✅ RAG retrieval for similar molecules
    ✅ Machine Learning models (Random Forest + Gradient Boosting)
    ✅ 6-agent CrewAI workflow combining ML + RAG + Expert reasoning
    
    Next steps:
    
    1. Make ML-enhanced predictions:
       from crewai_ml_prediction import MLEnhancedGCMSCrew
       crew = MLEnhancedGCMSCrew()
       result = crew.predict_retention_time('your_smiles')
    
    2. Train models on your data:
       python ml_model_trainer.py
    
    3. Query models directly:
       from ml_model_trainer import GCMSMLTrainer
       trainer = GCMSMLTrainer()
       trainer.load_models("HP-5MS")
       prediction = trainer.predict_with_model(features, 'random_forest')
    
    4. Use RAG + ML without agents:
       from crewai_ml_prediction import MLPredictionTool
       tool = MLPredictionTool()
       result = tool._run('CCO', 'HP-5MS')
    
    See documentation for more examples!
    """)


if __name__ == "__main__":
    main()
