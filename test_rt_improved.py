"""
Improved RT Prediction Test with Weighted Averaging
Shows detailed information about predictions
"""
import pandas as pd
from rag_retriever import MolecularRAGRetriever
import os
from dotenv import load_dotenv

load_dotenv()
TEST_CASES = [
    {"name": "Ethylamine", "smiles": "CCN", "column": "Rtx-5", "temp_program": "60-280C at 10C/min", "flow_rate": 1.0, "expected_rt": None},
    {"name": "4-Nitrotoluene", "smiles": "Cc1ccc(cc1)[N+](=O)[O-]", "column": "DB-624", "temp_program": "40-250C at 10C/min", "flow_rate": 1.0, "expected_rt": None},
    {"name": "n-Heptane", "smiles": "CCCCCCC", "column": "ZB-5", "temp_program": "60-280C at 10C/min", "flow_rate": 1.0, "expected_rt": None},
    {"name": "n-Octane", "smiles": "CCCCCCCC", "column": "DB-624", "temp_program": "40-250C at 10C/min", "flow_rate": 1.0, "expected_rt": None},
    {"name": "2-Butanol", "smiles": "CCC(C)O", "column": "DB-624", "temp_program": "40-250C at 10C/min", "flow_rate": 1.0, "expected_rt": None},
    {"name": "n-Decane", "smiles": "CCCCCCCCCC", "column": "BPX5", "temp_program": "60-280C at 10C/min", "flow_rate": 1.0, "expected_rt": None},
    {"name": "Anthracene", "smiles": "c1ccc2cc3ccccc3cc2c1", "column": "DB-1", "temp_program": "60-280C at 10C/min", "flow_rate": 1.0, "expected_rt": None},
]

def load_actual_values(test_cases, data_file='data/synthetic_gcms_data.csv'):
    """Load actual RT values from synthetic dataset"""
    df = pd.read_csv(data_file)
    
    for case in test_cases:
        # Find matching entry
        match = df[
            (df['SMILES'] == case['smiles']) &
            (df['Column'] == case['column']) &
            (df['TempProgram'] == case['temp_program'])
        ]
        
        if not match.empty:
            case['expected_rt'] = match.iloc[0]['RT']
            case['compound_class'] = match.iloc[0].get('compound_class', 'Unknown')
    
    return test_cases

def predict_rt_weighted(smiles, column, temp_program, retriever):
    """Improved prediction using weighted averaging"""
    print(f"\n  🔍 Retrieving similar compounds from Neo4j...")
    
    try:
        context = retriever.vector_similarity_search(smiles, top_k=15)
        print(f"  ✓ Retrieved {len(context) if context else 0} similar compounds")
    except Exception as e:
        print(f"  ❌ Error during retrieval: {e}")
        return None, {}
    
    if not context:
        print("  ❌ No similar compounds found")
        return None, {}
    
    # Check for exact compound match
    exact_compound = [c for c in context if c.get('score', 0) > 0.999]
    if exact_compound:
        exact_measurements = []
        for measurement in exact_compound[0].get('measurements', []):
            if measurement.get('column') == column and measurement.get('temp') == temp_program:
                exact_measurements.append(measurement['rt'])
        
        if exact_measurements:
            predicted = sum(exact_measurements) / len(exact_measurements)
            print(f"  ✨ Found EXACT compound match - using direct measurement!")
            print(f"  🎯 Predicted RT: {predicted:.2f} min (average of {len(exact_measurements)} measurements)")
            stats = {
                'n_measurements': len(exact_measurements),
                'n_exact': len(exact_measurements),
                'n_approx': 0,
                'rt_min': min(exact_measurements),
                'rt_max': max(exact_measurements),
                'rt_std': pd.Series(exact_measurements).std() if len(exact_measurements) > 1 else 0,
                'avg_similarity': 1.0,
                'exact_match_available': True
            }
            return predicted, stats
    
    # Extract all measurements
    measurements = []
    for compound in context:
        for measurement in compound.get('measurements', []):
            if measurement.get('rt') and measurement.get('column'):
                column_match = 1.0 if measurement.get('column') == column else 0.5
                temp_match = 1.0 if measurement.get('temp') == temp_program else 0.7
                weight = compound.get('score', 0) * column_match * temp_match
                
                measurements.append({
                    'rt': measurement['rt'],
                    'column': measurement['column'],
                    'weight': weight,
                    'exact_match': (measurement.get('column') == column and measurement.get('temp') == temp_program)
                })
    
    if not measurements:
        print("  ❌ No RT measurements found")
        return None, {}
    
    # Use best matches
    measurements.sort(key=lambda x: x['weight'], reverse=True)
    use_measurements = measurements[:10]
    
    # Calculate weighted average
    total_weight = sum(m['weight'] for m in use_measurements)
    if total_weight > 0:
        predicted = sum(m['rt'] * m['weight'] for m in use_measurements) / total_weight
    else:
        predicted = sum(m['rt'] for m in use_measurements) / len(use_measurements)
    
    print(f"  📊 Using {len(use_measurements)} measurements")
    print(f"  📈 RT range: {min(m['rt'] for m in use_measurements):.2f} - {max(m['rt'] for m in use_measurements):.2f} min")
    
    stats = {
        'n_measurements': len(use_measurements),
        'n_exact': len([m for m in use_measurements if m['exact_match']]),
        'n_approx': len([m for m in use_measurements if not m['exact_match']]),
        'rt_min': min(m['rt'] for m in use_measurements),
        'rt_max': max(m['rt'] for m in use_measurements),
        'rt_std': pd.Series([m['rt'] for m in use_measurements]).std(),
        'avg_similarity': 0.8,
        'exact_match_available': any(m['exact_match'] for m in use_measurements)
    }
    
    return predicted, stats
    # Continue with rest of function...

def run_validation():
    """Run validation tests"""
    print("="*80)
    print("IMPROVED RT PREDICTION VALIDATION TEST")
    print("Using weighted averaging with similarity scores")
    print("="*80)
    
    # Load actual values
    print("\n[1/3] Loading test cases...")
    test_cases = load_actual_values(TEST_CASES)
    valid_cases = [c for c in test_cases if c['expected_rt'] is not None]
    print(f"✅ Loaded {len(valid_cases)} test cases with known RT values")
    
    # Initialize RAG retriever
    print("\n[2/3] Connecting to Neo4j RAG system...")
    try:
        retriever = MolecularRAGRetriever(
            neo4j_uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD")
        )
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    # Run predictions
    print("\n[3/3] Running predictions...")
    print("="*80)
    
    results = []
    for i, case in enumerate(valid_cases, 1):
        print(f"\n{'='*80}")
        print(f"🧪 Test {i}/{len(valid_cases)}: {case['name']}")
        print(f"{'='*80}")
        print(f"   SMILES: {case['smiles']}")
        print(f"   Column: {case['column']}")
        print(f"   Temp Program: {case['temp_program']}")
        print(f"   🎯 Expected RT: {case['expected_rt']:.2f} min")
        
        # Get prediction
        predicted, stats = predict_rt_weighted(
            case['smiles'], 
            case['column'], 
            case['temp_program'],
            retriever
        )
        
        if predicted:
            error = abs(predicted - case['expected_rt'])
            percent_error = (error / case['expected_rt']) * 100
            
            print(f"\n  {'='*76}")
            print(f"  🎲 PREDICTED RT: {predicted:.2f} min")
            print(f"  📏 Error: {error:.2f} min ({percent_error:.1f}%)")
            print(f"  {'='*76}")
            
            # Quality assessment
            if percent_error < 5:
                quality = "🌟 EXCELLENT"
            elif percent_error < 10:
                quality = "✅ VERY GOOD"
            elif percent_error < 20:
                quality = "👍 GOOD"
            elif percent_error < 30:
                quality = "⚠️  ACCEPTABLE"
            else:
                quality = "❌ POOR"
            
            print(f"  Quality: {quality}")
            
            # Add confidence indicator
            confidence = "HIGH" if stats.get('exact_match_available') else "MEDIUM"
            if stats['n_measurements'] < 3:
                confidence = "LOW"
            print(f"  Confidence: {confidence} ({stats['n_measurements']} measurements)")
            
            results.append({
                'compound': case['name'],
                'expected': case['expected_rt'],
                'predicted': predicted,
                'error': error,
                'percent_error': percent_error,
                'n_measurements': stats['n_measurements'],
                'n_exact': stats['n_exact'],
                'confidence': confidence,
                'quality': quality
            })
        else:
            print("\n  ❌ Prediction failed")
    
    retriever.close()
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if results:
        df_results = pd.DataFrame(results)
        
        print(f"\n✅ Successful predictions: {len(results)}/{len(valid_cases)}")
        print(f"\n📊 ACCURACY METRICS:")
        print(f"   Mean error: {df_results['error'].mean():.2f} min")
        print(f"   Median error: {df_results['error'].median():.2f} min")
        print(f"   Mean % error: {df_results['percent_error'].mean():.1f}%")
        print(f"   Median % error: {df_results['percent_error'].median():.1f}%")
        print(f"   Best prediction: {df_results['percent_error'].min():.1f}%")
        print(f"   Worst prediction: {df_results['percent_error'].max():.1f}%")
        
        print(f"\n🎯 QUALITY BREAKDOWN:")
        quality_counts = df_results['quality'].value_counts()
        for quality, count in quality_counts.items():
            print(f"   {quality}: {count}")
        
        print(f"\n📈 DETAILED RESULTS:")
        print(df_results[['compound', 'expected', 'predicted', 'error', 'percent_error', 'n_exact', 'confidence']].to_string(index=False))
    else:
        print("\n❌ No successful predictions")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    run_validation()
