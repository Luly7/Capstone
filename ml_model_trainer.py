"""
Machine Learning Model Trainer for GC-MS Retention Time Prediction
Trains models using data from Neo4j graph database
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json
from typing import Dict, Tuple, List
from neo4j_schema import get_neo4j_connection
from neo4j import GraphDatabase
import os


class GCMSMLTrainer:
    """Train ML models for retention time prediction using Neo4j data"""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize trainer with model storage directory"""
        uri, username, password = get_neo4j_connection()
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Models to train
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.trained_models = {}
        
    def close(self):
        """Close database connection"""
        self.driver.close()
    
    def extract_training_data(self, column_type: str = "HP-5MS") -> pd.DataFrame:
        """Extract training data from Neo4j"""
        print(f"Extracting training data for {column_type} column...")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column {type: $column})
                MATCH (m)-[hf:HAS_FEATURE]->(f:MolecularFeature)
                RETURN m.smiles as smiles,
                       m.molecular_weight as mw,
                       m.logp as logp,
                       m.tpsa as tpsa,
                       m.num_rotatable_bonds as rotatable,
                       m.num_h_acceptors as h_acceptors,
                       m.num_h_donors as h_donors,
                       m.num_aromatic_rings as aromatic,
                       collect({feature: f.name, value: hf.value}) as features,
                       r.rt_minutes as rt
                ORDER BY m.smiles
            """, column=column_type)
            
            data = []
            for record in result:
                row = {
                    'smiles': record['smiles'],
                    'rt': record['rt'],
                    'molecular_weight': record['mw'],
                    'logp': record['logp'],
                    'tpsa': record['tpsa'],
                    'num_rotatable_bonds': record['rotatable'],
                    'num_h_acceptors': record['h_acceptors'],
                    'num_h_donors': record['h_donors'],
                    'num_aromatic_rings': record['aromatic']
                }
                
                # Add feature values
                for feat in record['features']:
                    row[feat['feature']] = feat['value']
                
                data.append(row)
        
        df = pd.DataFrame(data)
        print(f"Extracted {len(df)} samples with {df.shape[1]-2} features")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and target vector"""
        # Drop non-feature columns
        X_df = df.drop(['smiles', 'rt'], axis=1)
        y = df['rt'].values
        
        # Store feature names
        feature_names = X_df.columns.tolist()
        
        # Convert to numpy array
        X = X_df.values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        return X, y, feature_names
    
    def train_models(self, column_type: str = "HP-5MS", test_size: float = 0.2):
        """Train all models and evaluate performance"""
        print("\n" + "="*70)
        print("TRAINING ML MODELS FOR RETENTION TIME PREDICTION")
        print("="*70)
        
        # Extract data
        df = self.extract_training_data(column_type)
        
        if len(df) < 10:
            print("⚠️  Warning: Not enough training data (need at least 10 samples)")
            print("   Run data_ingestion.py to load more molecules first")
            return None
        
        # Prepare features
        X, y, feature_names = self.prepare_features(df)
        self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train each model
        results = {}
        for model_name, model in self.models.items():
            print(f"\n--- Training {model_name.replace('_', ' ').title()} ---")
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Evaluate
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Cross-validation (if enough data)
            if len(X_train) >= 20:
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=min(5, len(X_train)//4),
                    scoring='r2'
                )
                cv_r2 = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                cv_r2 = test_r2
                cv_std = 0.0
            
            results[model_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'cv_r2': cv_r2,
                'cv_std': cv_std
            }
            
            print(f"Training R²:   {train_r2:.4f}")
            print(f"Test R²:       {test_r2:.4f}")
            print(f"Test MAE:      {test_mae:.4f} minutes")
            print(f"Test RMSE:     {test_rmse:.4f} minutes")
            if len(X_train) >= 20:
                print(f"CV R² (±std):  {cv_r2:.4f} (±{cv_std:.4f})")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[-10:]  # Top 10
                
                print("\nTop 10 Important Features:")
                for i, idx in enumerate(reversed(indices), 1):
                    print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
            
            # Save model
            self.trained_models[model_name] = model
        
        # Save models and metadata
        self.save_models(column_type, results)
        
        # Print summary
        print("\n" + "="*70)
        print("TRAINING COMPLETE - MODEL COMPARISON")
        print("="*70)
        print(f"{'Model':<25} {'Test R²':<12} {'Test MAE':<15} {'CV R²':<12}")
        print("-"*70)
        for model_name, metrics in results.items():
            print(f"{model_name.replace('_', ' ').title():<25} "
                  f"{metrics['test_r2']:<12.4f} "
                  f"{metrics['test_mae']:<15.4f} "
                  f"{metrics['cv_r2']:<12.4f}")
        
        # Best model
        best_model = max(results.items(), key=lambda x: x[1]['test_r2'])
        print("\n" + "="*70)
        print(f"✅ Best Model: {best_model[0].replace('_', ' ').title()}")
        print(f"   Test R² = {best_model[1]['test_r2']:.4f}")
        print(f"   Test MAE = {best_model[1]['test_mae']:.4f} minutes")
        print("="*70)
        
        return results
    
    def save_models(self, column_type: str, results: Dict):
        """Save trained models and metadata"""
        # Save each model
        for model_name, model in self.trained_models.items():
            model_path = os.path.join(self.model_dir, f"{model_name}_{column_type}.pkl")
            joblib.dump(model, model_path)
            print(f"Saved model: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(self.model_dir, f"scaler_{column_type}.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'column_type': column_type,
            'feature_names': self.feature_names,
            'model_results': results,
            'n_features': len(self.feature_names)
        }
        
        metadata_path = os.path.join(self.model_dir, f"metadata_{column_type}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata: {metadata_path}")
    
    def load_models(self, column_type: str = "HP-5MS"):
        """Load trained models"""
        print(f"Loading models for {column_type}...")
        
        # Load metadata
        metadata_path = os.path.join(self.model_dir, f"metadata_{column_type}.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        
        # Load scaler
        scaler_path = os.path.join(self.model_dir, f"scaler_{column_type}.pkl")
        self.scaler = joblib.load(scaler_path)
        
        # Load models
        for model_name in ['random_forest', 'gradient_boosting']:
            model_path = os.path.join(self.model_dir, f"{model_name}_{column_type}.pkl")
            if os.path.exists(model_path):
                self.trained_models[model_name] = joblib.load(model_path)
                print(f"✓ Loaded {model_name}")
        
        print(f"✅ Loaded {len(self.trained_models)} models")
        return metadata
    
    def predict_with_model(self, features: Dict, model_name: str = 'random_forest') -> Dict:
        """Make prediction using trained model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Prepare feature vector
        feature_vector = []
        for fname in self.feature_names:
            feature_vector.append(features.get(fname, 0.0))
        
        X = np.array([feature_vector])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        model = self.trained_models[model_name]
        prediction = model.predict(X_scaled)[0]
        
        # Get prediction interval (approximate)
        if model_name == 'random_forest':
            # Use individual tree predictions for uncertainty
            tree_predictions = np.array([tree.predict(X_scaled)[0] 
                                        for tree in model.estimators_])
            std_error = np.std(tree_predictions)
            lower_bound = prediction - 1.96 * std_error
            upper_bound = prediction + 1.96 * std_error
        else:
            # Use fixed percentage for other models
            std_error = prediction * 0.1
            lower_bound = prediction - 1.96 * std_error
            upper_bound = prediction + 1.96 * std_error
        
        return {
            'predicted_rt': float(prediction),
            'lower_bound': float(max(0, lower_bound)),
            'upper_bound': float(upper_bound),
            'std_error': float(std_error),
            'model_used': model_name
        }


def train_all_models():
    """Train models for all available column types"""
    trainer = GCMSMLTrainer()
    
    # Get available columns
    with trainer.driver.session() as session:
        result = session.run("""
            MATCH (c:Column)
            RETURN DISTINCT c.type as column_type
        """)
        columns = [record['column_type'] for record in result]
    
    if not columns:
        print("No columns found in database. Run data_ingestion.py first.")
        trainer.close()
        return
    
    print(f"Found {len(columns)} column types: {', '.join(columns)}")
    
    # Train for each column
    all_results = {}
    for column in columns:
        print(f"\n{'='*70}")
        print(f"Training models for column: {column}")
        print('='*70)
        results = trainer.train_models(column_type=column)
        if results:
            all_results[column] = results
    
    trainer.close()
    
    print("\n" + "="*70)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("="*70)
    print(f"Models saved in: {trainer.model_dir}/")
    print("\nYou can now use ML predictions in CrewAI agents!")
    
    return all_results


if __name__ == "__main__":
    # Train models
    results = train_all_models()
