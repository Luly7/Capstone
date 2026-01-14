"""
Ingest synthetic GC-MS data into Neo4j
"""
from data_ingestion import GCMSDataIngestion
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path to import gcms_ingestion
sys.path.insert(0, os.getcwd())


def main():
    print("="*60)
    print("Ingesting Synthetic GC-MS Data into Neo4j")
    print("="*60)

    # Initialize ingestion
    ingestion = GCMSDataIngestion()

    # Ingest the synthetic data
    print("\nIngesting synthetic_gcms_data.csv...")
    ingestion.ingest_from_csv('data/synthetic_gcms_data.csv')

    print("\n" + "="*60)
    print("✅ Ingestion Complete!")
    print("="*60)
    print("\nYour Neo4j database now contains:")
    print("  - 95 unique compounds")
    print("  - 285 retention time measurements")
    print("  - 15 different column configurations")
    print("\nNext step: Run predictions with CrewAI")
    print("  python run_crew.py")


if __name__ == "__main__":
    main()
