"""
Test RT Prediction System
Compare RAG/CrewAI predictions against known values
"""
import pandas as pd
from rag_retriever import MolecularRAGRetriever
import os
from dotenv import load_dotenv

load_dotenv()

# Test compounds with known RT values from synthetic data
TEST_CASES = [
    {
        "name": "Benzene",
        "smiles": "c1ccccc1",
        "column": "HP-5MS",
        "temp_program": "60-280C at 10C/min",
        "flow_rate": 1.0,
        "expected_rt": None
    },
    {
        "name": "Toluene",
        "smiles": "Cc1ccccc1",
        "column": "HP-5MS",
        "temp_program": "60-280C at 10C/min",
        "flow_rate": 1.0,
        "expected_rt": None
    },
    {
        "name": "Naphthalene",
        "smiles": "c1ccc2ccccc2c1",
        "column": "HP-5MS",
        "temp_program": "60-280C at 10C/min",
        "flow_rate": 1.0,
        "expected_rt": None
    },
    {
        "name": "Ethanol",
        "smiles": "CCO",
        "column": "DB-5",
        "temp_program": "60-300C at 15C/min",
        "flow_rate": 1.2,
        "expected_rt": None
    },
    {
        "name": "Acetone",
        "smiles": "CC(=O)C",
        "column": "HP-5MS",
        "temp_program": "60-280C at 10C/min",
        "flow_rate": 1.0,
        "expected_rt": None
    }
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
            case['compound_class'] = match.iloc[0]['compound_class']

    return test_cases


def predict_rt_simple(smiles, column, retriever):
    """Simple prediction using RAG retrieval only"""
    print(f"\n  Retrieving similar compounds from Neo4j...")

    try:
        context = retriever.vector_similarity_search(smiles, top_k=5)
        print(f"  DEBUG: Retrieved {len(context) if context else 0} results")
    except Exception as e:
        print(f"  DEBUG: Error during retrieval: {e}")
        return None

    if not context:
        print("  ❌ No similar compounds found")
        return None

    # Extract RT values - ONLY from same column
    rt_values = []
    same_column_compounds = 0

    for compound in context:
        measurements = compound.get('measurements', [])

        # Only use measurements from the same column
        same_column_rts = [m['rt']
                           for m in measurements if m.get('column') == column]

        if same_column_rts:
            rt_values.extend(same_column_rts)
            same_column_compounds += 1

    if rt_values:
        predicted = sum(rt_values) / len(rt_values)
        print(
            f"  Found {len(rt_values)} RT measurements on {column} from {same_column_compounds} similar compounds")
        print(f"  RT range: {min(rt_values):.2f} - {max(rt_values):.2f} min")
        return predicted

    print(f"  ❌ No RT measurements found on {column}")
    return None


def run_validation():
    """Run validation tests"""
    print("="*70)
    print("RT PREDICTION VALIDATION TEST")
    print("="*70)

    # Load actual values
    print("\n[1/3] Loading test cases...")
    test_cases = load_actual_values(TEST_CASES)
    valid_cases = [c for c in test_cases if c['expected_rt'] is not None]
    print(f"✅ Loaded {len(valid_cases)} test cases with known RT values")

    # Initialize RAG retriever
    print("\n[2/3] Connecting to Neo4j RAG system...")
    try:
        # Debug: print what we're passing
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")

        print(f"  URI: {uri}")
        print(f"  Username: {username}")
        print(f"  Password: {'***' if password else None}")

        retriever = MolecularRAGRetriever(
            neo4j_uri=uri,
            username=username,
            password=password

        )
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return

    # Run predictions
    print("\n[3/3] Running predictions...")
    print("="*70)

    results = []
    for i, case in enumerate(valid_cases, 1):
        print(f"\n🧪 Test {i}/{len(valid_cases)}: {case['name']}")
        print(f"   SMILES: {case['smiles']}")
        print(f"   Column: {case['column']}")
        print(f"   Expected RT: {case['expected_rt']:.2f} min")

        # Get prediction
        predicted = predict_rt_simple(
            case['smiles'], case['column'], retriever)

        if predicted:
            error = abs(predicted - case['expected_rt'])
            percent_error = (error / case['expected_rt']) * 100

            print(f"   Predicted RT: {predicted:.2f} min")
            print(f"   Error: {error:.2f} min ({percent_error:.1f}%)")

            if percent_error < 10:
                print("   ✅ Good prediction!")
            elif percent_error < 20:
                print("   ⚠️  Acceptable")
            else:
                print("   ❌ Poor prediction")

            results.append({
                'compound': case['name'],
                'expected': case['expected_rt'],
                'predicted': predicted,
                'error': error,
                'percent_error': percent_error
            })
        else:
            print("   ❌ Prediction failed")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if results:
        df_results = pd.DataFrame(results)
        print(f"\nSuccessful predictions: {len(results)}/{len(valid_cases)}")
        print(f"Average error: {df_results['error'].mean():.2f} min")
        print(f"Average % error: {df_results['percent_error'].mean():.1f}%")
        print(f"\nResults table:")
        print(df_results.to_string(index=False))
    else:
        print("\n❌ No successful predictions")

    print("\n" + "="*70)


if __name__ == "__main__":
    run_validation()
