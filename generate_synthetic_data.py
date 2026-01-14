"""
Generate Synthetic GC-MS Retention Time Data
Combines molecules_database.csv and column_conditions.csv to create
realistic training data for the Neo4j RAG system.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def calculate_molecular_descriptors(smiles):
    """Calculate key molecular descriptors for RT prediction"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_rotatable': Descriptors.NumRotatableBonds(mol),
        'num_aromatic': Descriptors.NumAromaticRings(mol)
    }

def estimate_base_rt(descriptors, column_type):
    """
    Estimate base retention time from molecular properties
    Based on typical GC-MS behavior:
    - Higher MW = longer RT
    - Higher logP (lipophilicity) = longer RT
    - Aromatic rings = longer RT
    - Column polarity affects retention
    """
    if descriptors is None:
        return None
    
    # Base RT from molecular weight (major factor)
    base_rt = descriptors['mw'] / 10.0  # Rough approximation
    
    # LogP contribution (lipophilicity)
    base_rt += descriptors['logp'] * 1.5
    
    # Aromatic contribution
    base_rt += descriptors['num_aromatic'] * 2.0
    
    # Column type modifiers
    column_polarity = {
        'DB-5': 1.0,      # Non-polar baseline
        'HP-5MS': 1.0,    # Similar to DB-5
        'Rtx-5': 1.0,     # Similar to DB-5
        'ZB-5': 1.0,      # Similar to DB-5
        'DB-1': 0.95,     # More non-polar (shorter RT)
        'SPB-1': 0.95,    # Similar to DB-1
        'Rtx-1': 0.95,    # Similar to DB-1
        'DB-WAX': 1.3,    # Polar column (longer RT for polar compounds)
        'DB-35': 1.1,     # Medium polarity
        'DB-17': 1.15,    # Medium-high polarity
        'Rtx-35': 1.1,    # Similar to DB-35
        'DB-624': 1.2,    # Polar
        'Rtx-624': 1.2,   # Polar
        'DB-VRX': 1.05    # Slightly polar
    }
    
    polarity_factor = column_polarity.get(column_type, 1.0)
    base_rt *= polarity_factor
    
    return max(0.5, base_rt)  # Minimum RT of 0.5 min

def adjust_for_temperature_program(base_rt, temp_program):
    """
    Adjust RT based on temperature program
    Faster ramps = shorter overall retention times
    """
    # Extract ramp rate from program string (e.g., "60-280C at 10C/min")
    try:
        ramp_rate = float(temp_program.split('at')[1].split('C/min')[0].strip())
    except:
        ramp_rate = 10.0  # default
    
    # Faster ramp = shorter RT
    ramp_factor = 10.0 / ramp_rate
    
    return base_rt * ramp_factor

def add_noise(rt, noise_level=0.05):
    """Add realistic experimental noise"""
    noise = np.random.normal(0, rt * noise_level)
    return max(0.5, rt + noise)

def generate_synthetic_dataset(molecules_file, columns_file, 
                               samples_per_molecule=3, output_file='synthetic_gcms_data.csv'):
    """
    Generate synthetic GC-MS dataset
    
    Args:
        molecules_file: Path to molecules_database.csv
        columns_file: Path to column_conditions.csv
        samples_per_molecule: Number of column conditions to test per molecule
        output_file: Output CSV filename
    """
    
    print("Loading reference databases...")
    molecules_df = pd.read_csv(molecules_file, quotechar='"', on_bad_lines='skip')
    columns_df = pd.read_csv(columns_file)
    
    print(f"Loaded {len(molecules_df)} molecules and {len(columns_df)} column configurations")
    
    # Generate synthetic data
    synthetic_data = []
    
    print("\nGenerating synthetic retention times...")
    for idx, mol_row in molecules_df.iterrows():
        smiles = mol_row['smiles']
        compound_name = mol_row['compound_name']
        
        # Calculate molecular descriptors
        descriptors = calculate_molecular_descriptors(smiles)
        if descriptors is None:
            print(f"  ⚠️  Skipping {compound_name} - invalid SMILES")
            continue
        
        # Randomly select N column conditions for this molecule
        selected_columns = columns_df.sample(n=min(samples_per_molecule, len(columns_df)))
        
        for _, col_row in selected_columns.iterrows():
            # Estimate base RT
            base_rt = estimate_base_rt(descriptors, col_row['column_type'])
            
            # Adjust for temperature program
            adjusted_rt = adjust_for_temperature_program(base_rt, col_row['temperature_program'])
            
            # Add experimental noise
            final_rt = add_noise(adjusted_rt)
            
            # Create record
            record = {
                'SMILES': smiles,
                'RT': round(final_rt, 2),
                'Column': col_row['column_type'],
                'TempProgram': col_row['temperature_program'],
                'FlowRate': col_row['flow_rate_ml_min'],
                'compound_name': compound_name,
                'compound_class': mol_row['compound_class'],
                'column_id': col_row['column_id'],
                'carrier_gas': col_row['carrier_gas']
            }
            
            synthetic_data.append(record)
        
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(molecules_df)} molecules...")
    
    # Create DataFrame
    result_df = pd.DataFrame(synthetic_data)
    
    # Sort by RT for easier viewing
    result_df = result_df.sort_values('RT').reset_index(drop=True)
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Generated {len(result_df)} synthetic GC-MS measurements")
    print(f"📁 Saved to: {output_file}")
    print(f"\nDataset statistics:")
    print(f"  RT range: {result_df['RT'].min():.2f} - {result_df['RT'].max():.2f} min")
    print(f"  Unique compounds: {result_df['compound_name'].nunique()}")
    print(f"  Unique columns: {result_df['Column'].nunique()}")
    print(f"\nSample data:")
    print(result_df.head(10).to_string(index=False))
    
    return result_df

if __name__ == "__main__":
    # Generate dataset
    df = generate_synthetic_dataset(
        molecules_file='/mnt/user-data/uploads/molecules_database.csv',
        columns_file='/mnt/user-data/uploads/column_conditions.csv',
        samples_per_molecule=3,  # Each molecule tested on 3 different columns
        output_file='synthetic_gcms_data.csv'
    )
    
    print("\n" + "="*60)
    print("Ready to ingest into Neo4j!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the synthetic_gcms_data.csv file")
    print("2. Run: python ingest_synthetic_data.py")
    print("   OR use the ingestion class directly:")
    print("   >>> from gcms_ingestion import GCMSDataIngestion")
    print("   >>> ingestion = GCMSDataIngestion()")
    print("   >>> ingestion.ingest_from_csv('synthetic_gcms_data.csv')")
