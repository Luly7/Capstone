"""
Visualization utilities for Neo4j GC-MS Knowledge Graph
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
from neo4j_schema import get_neo4j_connection
from neo4j import GraphDatabase


class GCMSVisualizer:
    """Visualizations for molecular knowledge graph and predictions"""

    def __init__(self):
        uri, username, password = get_neo4j_connection()
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        sns.set_style("whitegrid")

    def close(self):
        self.driver.close()

    def get_all_molecules(self) -> pd.DataFrame:
        """Retrieve all molecules from database"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Molecule)
                OPTIONAL MATCH (m)-[:MEASURED_ON]->(r:RetentionTime)
                RETURN m.smiles as smiles,
                       m.molecular_weight as mw,
                       m.logp as logp,
                       m.tpsa as tpsa,
                       collect(r.rt_minutes) as retention_times
            """)
            data = [record.data() for record in result]
            return pd.DataFrame(data)

    def plot_molecular_space(self, save_path: str = None):
        """Plot molecular property space (MW vs LogP)"""
        df = self.get_all_molecules()

        plt.figure(figsize=(12, 8))

        # Color by TPSA if available
        if 'tpsa' in df.columns and df['tpsa'].notna().any():
            scatter = plt.scatter(df['mw'], df['logp'],
                                  c=df['tpsa'], cmap='viridis',
                                  s=100, alpha=0.6, edgecolors='black')
            plt.colorbar(scatter, label='TPSA (Ų)')
        else:
            plt.scatter(df['mw'], df['logp'], s=100,
                        alpha=0.6, edgecolors='black')

        plt.xlabel('Molecular Weight (g/mol)', fontsize=12)
        plt.ylabel('LogP', fontsize=12)
        plt.title('Molecular Property Space', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_rt_distribution(self, column_type: str = "HP-5MS", save_path: str = None):
        """Plot retention time distribution"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column {type: $col})
                RETURN r.rt_minutes as rt, m.molecular_weight as mw
            """, col=column_type)
            data = [record.data() for record in result]

        if not data:
            print(f"No data found for column type: {column_type}")
            return

        df = pd.DataFrame(data)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Histogram
        axes[0].hist(df['rt'], bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Retention Time (min)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(
            f'RT Distribution - {column_type}', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # RT vs MW
        axes[1].scatter(df['mw'], df['rt'], alpha=0.6,
                        s=100, edgecolors='black')
        axes[1].set_xlabel('Molecular Weight (g/mol)', fontsize=12)
        axes[1].set_ylabel('Retention Time (min)', fontsize=12)
        axes[1].set_title(
            f'RT vs MW - {column_type}', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Add trendline
        z = np.polyfit(df['mw'], df['rt'], 1)
        p = np.poly1d(z)
        axes[1].plot(df['mw'], p(df['mw']), "r--",
                     alpha=0.8, linewidth=2, label='Trend')
        axes[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_importance(self, features: Dict[str, float], top_n: int = 15,
                                save_path: str = None):
        """Plot feature importance for predictions"""
        sorted_features = sorted(
            features.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

        feature_names = [f[0] for f in sorted_features]
        feature_values = [f[1] for f in sorted_features]

        plt.figure(figsize=(10, 8))
        colors = ['green' if v > 0 else 'red' for v in feature_values]
        plt.barh(feature_names, feature_values,
                 color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel('Feature Value', fontsize=12)
        plt.ylabel('Molecular Feature', fontsize=12)
        plt.title('Top Molecular Features', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_similarity_network(self, center_smiles: str, depth: int = 1,
                                save_path: str = None):
        """Plot molecular similarity network"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (m:Molecule {smiles: $smiles})-[:SIMILAR_TO*1..$depth]-(similar)
                WITH m, similar, relationships(path) as rels
                RETURN m.smiles as center,
                       collect(DISTINCT similar.smiles) as similar_molecules,
                       collect(DISTINCT [rel.similarity_score for rel in rels]) as scores
            """, smiles=center_smiles, depth=depth)
            data = result.single()

        if not data:
            print(f"No similarity data found for {center_smiles}")
            return

        # This is a simplified visualization
        # For a full network graph, consider using networkx
        print(f"Center molecule: {data['center']}")
        print(f"Found {len(data['similar_molecules'])} similar molecules")
        print("\nTop 10 most similar:")
        for i, mol in enumerate(data['similar_molecules'][:10], 1):
            print(f"{i}. {mol}")

    def plot_prediction_confidence(self, predictions: List[Dict], save_path: str = None):
        """Plot prediction confidence analysis"""
        df = pd.DataFrame(predictions)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Confidence distribution
        if 'confidence' in df.columns:
            axes[0].hist(df['confidence'], bins=15,
                         edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Confidence Score', fontsize=12)
            axes[0].set_ylabel('Frequency', fontsize=12)
            axes[0].set_title('Prediction Confidence Distribution',
                              fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)

        # Predicted vs Actual (if actual values available)
        if 'predicted_rt' in df.columns and 'actual_rt' in df.columns:
            axes[1].scatter(df['actual_rt'],
                            df['predicted_rt'], alpha=0.6, s=100)

            # Add diagonal line
            min_val = min(df['actual_rt'].min(), df['predicted_rt'].min())
            max_val = max(df['actual_rt'].max(), df['predicted_rt'].max())
            axes[1].plot([min_val, max_val], [
                         min_val, max_val], 'r--', linewidth=2)

            axes[1].set_xlabel('Actual RT (min)', fontsize=12)
            axes[1].set_ylabel('Predicted RT (min)', fontsize=12)
            axes[1].set_title('Predicted vs Actual RT',
                              fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)

            # Calculate R²
            from sklearn.metrics import r2_score
            r2 = r2_score(df['actual_rt'], df['predicted_rt'])
            axes[1].text(0.05, 0.95, f'R² = {r2:.3f}',
                         transform=axes[1].transAxes, fontsize=12,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_summary_report(self):
        """Generate summary statistics of the knowledge graph"""
        with self.driver.session() as session:
            # Count nodes
            molecule_count = session.run(
                "MATCH (m:Molecule) RETURN count(m) as count").single()['count']
            feature_count = session.run(
                "MATCH (f:MolecularFeature) RETURN count(f) as count").single()['count']
            rt_count = session.run(
                "MATCH (r:RetentionTime) RETURN count(r) as count").single()['count']

            # Count relationships
            has_feature = session.run(
                "MATCH ()-[r:HAS_FEATURE]->() RETURN count(r) as count").single()['count']
            similar_to = session.run(
                "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count").single()['count']

            # RT statistics
            rt_stats = session.run("""
                MATCH (r:RetentionTime)
                RETURN min(r.rt_minutes) as min_rt,
                       max(r.rt_minutes) as max_rt,
                       avg(r.rt_minutes) as avg_rt
            """).single()

        report = f"""
        ╔═══════════════════════════════════════════════════════╗
        ║     Neo4j GC-MS Knowledge Graph Summary Report        ║
        ╚═══════════════════════════════════════════════════════╝
        
        Node Statistics:
        ────────────────
        • Molecules:          {molecule_count:>6}
        • Molecular Features: {feature_count:>6}
        • Retention Times:    {rt_count:>6}
        
        Relationship Statistics:
        ────────────────────────
        • HAS_FEATURE:        {has_feature:>6}
        • SIMILAR_TO:         {similar_to:>6}
        
        Retention Time Statistics:
        ──────────────────────────
        • Min RT:             {rt_stats['min_rt']:>6.2f} min
        • Max RT:             {rt_stats['max_rt']:>6.2f} min
        • Avg RT:             {rt_stats['avg_rt']:>6.2f} min
        
        Graph Density:
        ──────────────
        • Features per Molecule: {has_feature/molecule_count if molecule_count > 0 else 0:.1f}
        • Similarity Connections: {similar_to/(molecule_count**2) if molecule_count > 0 else 0:.4f}
        """

        print(report)
        return report


def main():
    """Generate example visualizations"""
    visualizer = GCMSVisualizer()

    print("Generating visualizations...\n")

    # Summary report
    visualizer.generate_summary_report()

    # Molecular space
    print("\n1. Plotting molecular property space...")
    visualizer.plot_molecular_space(save_path='molecular_space.png')

    # RT distribution
    print("\n2. Plotting retention time distribution...")
    visualizer.plot_rt_distribution(
        column_type="HP-5MS", save_path='rt_distribution.png')

    # Example feature importance
    print("\n3. Plotting example feature importance...")
    example_features = {
        'MolecularWeight': 180.5,
        'LogP': 2.3,
        'TPSA': 45.2,
        'NumRotatableBonds': 5,
        'NumHAcceptors': 3,
        'NumHDonors': 2,
        'NumAromaticRings': 1,
        'Chi0v': 3.5,
        'Kappa1': 7.2
    }
    visualizer.plot_feature_importance(
        example_features, save_path='feature_importance.png')

    visualizer.close()
    print("\nVisualizations complete! Check PNG files in current directory.")


if __name__ == "__main__":
    main()
