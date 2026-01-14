"""
Simple ML-Based GC-MS Retention Time Predictor (No CrewAI needed)
Demonstrates: Neo4j + RAG + Machine Learning
"""

import os
from dotenv import load_dotenv
from ml_model_trainer import GCMSMLTrainer
from rag_retriever import MolecularRAGRetriever
from neo4j_schema import get_neo4j_connection
from rdkit import Chem
from rdkit.Chem import Descriptors


class SimpleMLPredictor:
    """ML-based retention time predictor without CrewAI dependencies"""
    
    def __init__(self):
        """Initialize ML models and RAG retriever"""
        self.trainer = GCMSMLTrainer()
        uri, username, password = get_neo4j_connection()
        self.retriever = MolecularRAGRetriever(uri, username, password)
        self._models_loaded = False
    
    def ensure_models_loaded(self, column_type="HP-5MS"):
        """Load ML models if not already loaded"""
        if not self._models_loaded:
            try:
                self.trainer.load_models(column_type)
                self._models_loaded = True
                print(f"✅ Loaded ML models for {column_type}")
            except FileNotFoundError:
                print("⚠️  ML models not found. Training now...")
                from ml_model_trainer import train_all_models
                train_all_models()
                self.trainer.load_models(column_type)
                self._models_loaded = True
    
    def get_molecular_features(self, smiles):
        """Extract all molecular features for prediction"""
        # Basic properties from database/RDKit
        features = self.retriever.extract_molecular_properties(smiles)
        
        # Additional RDKit descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            features['Chi0v'] = Descriptors.Chi0v(mol)
            features['Chi1v'] = Descriptors.Chi1v(mol)
            features['Kappa1'] = Descriptors.Kappa1(mol)
            features['Kappa2'] = Descriptors.Kappa2(mol)
            features['MolMR'] = Descriptors.MolMR(mol)
            features['BalabanJ'] = Descriptors.BalabanJ(mol)
            features['FractionCSP3'] = Descriptors.FractionCSP3(mol)
            features['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
            features['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            features['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
        
        return features
    
    def predict(self, smiles, column_type="HP-5MS", show_rag=True, show_details=True):
        """
        Make retention time prediction for a molecule
        
        Args:
            smiles: SMILES string of molecule
            column_type: GC column type (e.g., "HP-5MS")
            show_rag: Whether to show similar molecules from database
            show_details: Whether to show detailed analysis
        
        Returns:
            dict with prediction results
        """
        # Ensure models are loaded
        self.ensure_models_loaded(column_type)
        
        # Get molecular name
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"error": "Invalid SMILES string"}
        
        mol_name = Chem.MolToSmiles(mol)  # Canonical SMILES
        
        print("\n" + "="*70)
        print(f"RETENTION TIME PREDICTION")
        print("="*70)
        print(f"Molecule: {smiles}")
        print(f"Column: {column_type}")
        
        # Get molecular features
        features = self.get_molecular_features(smiles)
        
        if show_details:
            print(f"\nMolecular Properties:")
            print(f"  MW: {features.get('molecular_weight', 0):.2f} g/mol")
            print(f"  LogP: {features.get('logp', 0):.2f}")
            print(f"  TPSA: {features.get('tpsa', 0):.2f} Ų")
            print(f"  Rotatable Bonds: {features.get('num_rotatable_bonds', 0)}")
            print(f"  H-Acceptors: {features.get('num_h_acceptors', 0)}")
            print(f"  H-Donors: {features.get('num_h_donors', 0)}")
            print(f"  Aromatic Rings: {features.get('num_aromatic_rings', 0)}")
        
        # Get RAG context (similar molecules)
        if show_rag:
            print(f"\n" + "-"*70)
            print("RAG RETRIEVAL: Similar Molecules from Database")
            print("-"*70)
            
            try:
                context = self.retriever.retrieve_prediction_context(
                    query_smiles=smiles,
                    column_type=column_type,
                    top_k=5
                )
                
                if context.get('similar_molecules'):
                    print(f"\nFound {len(context['similar_molecules'])} similar molecules:")
                    for i, sim_mol in enumerate(context['similar_molecules'][:3], 1):
                        print(f"  {i}. {sim_mol.get('smiles', 'N/A')}")
                        print(f"     Similarity: {sim_mol.get('similarity_score', 0):.3f}")
                        if sim_mol.get('retention_time'):
                            print(f"     RT: {sim_mol.get('retention_time', 0):.2f} min")
                else:
                    print("  No similar molecules found in database")
            except Exception as e:
                print(f"  ⚠️  RAG retrieval skipped: {e}")
        
        # ML Predictions
        print(f"\n" + "-"*70)
        print("MACHINE LEARNING PREDICTIONS")
        print("-"*70)
        
        # Random Forest
        rf_pred = self.trainer.predict_with_model(features, 'random_forest')
        print(f"\nRandom Forest Model:")
        print(f"  Predicted RT: {rf_pred['predicted_rt']:.2f} minutes")
        print(f"  95% CI: [{rf_pred['lower_bound']:.2f}, {rf_pred['upper_bound']:.2f}]")
        print(f"  Std Error: ±{rf_pred['std_error']:.2f} minutes")
        
        # Gradient Boosting
        gb_pred = self.trainer.predict_with_model(features, 'gradient_boosting')
        print(f"\nGradient Boosting Model:")
        print(f"  Predicted RT: {gb_pred['predicted_rt']:.2f} minutes")
        print(f"  95% CI: [{gb_pred['lower_bound']:.2f}, {gb_pred['upper_bound']:.2f}]")
        print(f"  Std Error: ±{gb_pred['std_error']:.2f} minutes")
        
        # Ensemble
        ensemble_rt = (rf_pred['predicted_rt'] + gb_pred['predicted_rt']) / 2
        ensemble_std = (rf_pred['std_error'] + gb_pred['std_error']) / 2
        model_agreement = abs(rf_pred['predicted_rt'] - gb_pred['predicted_rt'])
        
        print(f"\n" + "="*70)
        print("FINAL ENSEMBLE PREDICTION")
        print("="*70)
        print(f"Predicted RT: {ensemble_rt:.2f} ± {ensemble_std:.2f} minutes")
        print(f"95% CI: [{ensemble_rt - 1.96*ensemble_std:.2f}, {ensemble_rt + 1.96*ensemble_std:.2f}]")
        print(f"Model Agreement: {model_agreement:.2f} minutes difference")
        
        # Confidence assessment
        if model_agreement < 0.5:
            confidence = "High"
            print(f"Confidence: {confidence} ✅ (models agree closely)")
        elif model_agreement < 1.0:
            confidence = "Medium"
            print(f"Confidence: {confidence} ⚠️  (models somewhat agree)")
        else:
            confidence = "Low"
            print(f"Confidence: {confidence} ⚠️  (models disagree)")
        
        print("="*70 + "\n")
        
        return {
            'smiles': smiles,
            'column': column_type,
            'random_forest': rf_pred,
            'gradient_boosting': gb_pred,
            'ensemble_rt': ensemble_rt,
            'ensemble_std': ensemble_std,
            'confidence': confidence,
            'model_agreement': model_agreement
        }
    
    def batch_predict(self, smiles_list, column_type="HP-5MS"):
        """Predict retention times for multiple molecules"""
        results = []
        
        print(f"\n{'='*70}")
        print(f"BATCH PREDICTION: {len(smiles_list)} molecules")
        print('='*70)
        
        for i, smiles in enumerate(smiles_list, 1):
            print(f"\n[{i}/{len(smiles_list)}] Processing: {smiles}")
            result = self.predict(smiles, column_type, show_rag=False, show_details=False)
            results.append(result)
        
        # Summary
        print(f"\n{'='*70}")
        print("BATCH PREDICTION SUMMARY")
        print('='*70)
        print(f"{'Molecule':<20} {'Ensemble RT':<15} {'Confidence':<12}")
        print('-'*70)
        
        for result in results:
            if 'error' not in result:
                smiles_short = result['smiles'][:17] + '...' if len(result['smiles']) > 20 else result['smiles']
                print(f"{smiles_short:<20} {result['ensemble_rt']:>8.2f} ± {result['ensemble_std']:.2f}   {result['confidence']:<12}")
        
        return results
    
    def close(self):
        """Clean up connections"""
        self.trainer.close()
        self.retriever.close()


def main():
    """Demo of ML predictor"""
    load_dotenv()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║  Simple ML-Based GC-MS Retention Time Predictor              ║
    ║  Neo4j + RAG + Machine Learning (No CrewAI needed!)          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    predictor = SimpleMLPredictor()
    
    # Example 1: Single prediction
    print("\n🔬 Example 1: Single Molecule Prediction")
    result1 = predictor.predict(
        smiles="CCO",  # Ethanol
        column_type="HP-5MS",
        show_rag=True,
        show_details=True
    )
    
    # Example 2: Another molecule
    print("\n🔬 Example 2: Benzene Prediction")
    result2 = predictor.predict(
        smiles="c1ccccc1",  # Benzene
        column_type="HP-5MS",
        show_rag=True,
        show_details=True
    )
    
    # Example 3: Batch prediction
    print("\n🔬 Example 3: Batch Prediction")
    molecules = [
        "CCO",           # Ethanol
        "CC(C)O",        # Isopropanol
        "CCCCCO",        # Pentanol
        "c1ccccc1",      # Benzene
        "CC(=O)O"        # Acetic acid
    ]
    
    batch_results = predictor.batch_predict(molecules, "HP-5MS")
    
    predictor.close()
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  ✅ PREDICTION COMPLETE!                                     ║
    ║                                                               ║
    ║  Your ML system is working:                                   ║
    ║  • Random Forest predictions with confidence intervals        ║
    ║  • Gradient Boosting for validation                          ║
    ║  • Ensemble averaging for reliability                        ║
    ║  • RAG retrieval of similar molecules                        ║
    ║                                                               ║
    ║  No CrewAI needed - pure ML + Neo4j + RAG! 🚀               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
