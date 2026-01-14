"""
Enhanced CrewAI Architecture with Machine Learning Integration
Combines ML predictions with RAG retrieval and multi-agent reasoning
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from typing import Dict, List
from pydantic import BaseModel, Field
from rag_retriever import MolecularRAGRetriever
from neo4j_schema import get_neo4j_connection
from ml_model_trainer import GCMSMLTrainer
import os
import json


class MLPredictionTool(BaseTool):
    """Custom CrewAI tool for ML-based retention time prediction"""
    name: str = "ml_retention_predictor"
    description: str = "Predicts retention time using trained machine learning models (Random Forest, Gradient Boosting)"
    
    class Config:
        arbitrary_types_allowed = True
    
    _trainer: GCMSMLTrainer = None
    _models_loaded: bool = False
    
    def __init__(self):
        super().__init__()
        object.__setattr__(self, "_trainer", GCMSMLTrainer())
    
    def _ensure_models_loaded(self, column_type: str = "HP-5MS"):
        """Lazy load models when needed"""
        if not self._models_loaded:
            try:
                self._trainer.load_models(column_type)
                self._models_loaded = True
            except FileNotFoundError:
                raise ValueError(
                    "ML models not found. Please run 'python ml_model_trainer.py' first to train models."
                )
    
    def _run(self, smiles: str, column_type: str = "HP-5MS") -> str:
        """Execute ML prediction"""
        self._ensure_models_loaded(column_type)
        
        # Get molecular features from retriever
        uri, username, password = get_neo4j_connection()
        
        # Extract features
        features = self._self._retriever.extract_molecular_properties(smiles)
        
        # Add RDKit descriptors
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
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
        
        self._self._retriever.close()
        
        # Get predictions from both models
        predictions = {}
        for model_name in ['random_forest', 'gradient_boosting']:
            pred = self._trainer.predict_with_model(features, model_name)
            predictions[model_name] = pred
        
        # Format output
        output = f"""
Machine Learning Predictions for {smiles}:

Random Forest Model:
  Predicted RT: {predictions['random_forest']['predicted_rt']:.2f} minutes
  95% CI: [{predictions['random_forest']['lower_bound']:.2f}, {predictions['random_forest']['upper_bound']:.2f}]
  Std Error: ±{predictions['random_forest']['std_error']:.2f} minutes

Gradient Boosting Model:
  Predicted RT: {predictions['gradient_boosting']['predicted_rt']:.2f} minutes
  95% CI: [{predictions['gradient_boosting']['lower_bound']:.2f}, {predictions['gradient_boosting']['upper_bound']:.2f}]
  Std Error: ±{predictions['gradient_boosting']['std_error']:.2f} minutes

Ensemble Average: {(predictions['random_forest']['predicted_rt'] + predictions['gradient_boosting']['predicted_rt'])/2:.2f} minutes
"""
        return output


class RAGRetrievalTool(BaseTool):
    """Custom CrewAI tool for Neo4j RAG retrieval"""
    name: str = "molecular_knowledge_graph"
    description: str = "Retrieves relevant molecular data and retention times from Neo4j knowledge graph"
    
    class Config:
        arbitrary_types_allowed = True
    
    _retriever: MolecularRAGRetriever = None
    
    def __init__(self):
        super().__init__()
        uri, username, password = get_neo4j_connection()
        object.__setattr__(self, "_retriever", MolecularRAGRetriever(uri, username, password))
    
    def _run(self, smiles: str, column_type: str = "HP-5MS", top_k: int = 10) -> str:
        """Execute RAG retrieval and return formatted context"""
        context = self._retriever.retrieve_prediction_context(
            query_smiles=smiles,
            column_type=column_type,
            top_k=top_k
        )
        return self._retriever.format_context_for_llm(context)


class MLEnhancedGCMSCrew:
    """CrewAI workflow with ML predictions + RAG + Expert reasoning"""
    
    def __init__(self):
        self.rag_tool = RAGRetrievalTool()
        self.ml_tool = MLPredictionTool()
        self.setup_agents()
    
    def setup_agents(self):
        """Initialize specialized agents including ML expert"""
        
        # Agent 1: Molecular Analysis Expert
        self.molecular_analyst = Agent(
            role="Molecular Structure Analyst",
            goal="Analyze molecular structure and extract relevant chemical features for retention time prediction",
            backstory="""You are an expert analytical chemist specializing in molecular structure 
            analysis. You understand how molecular properties like polarity, molecular weight, 
            functional groups, and structural features influence chromatographic behavior.""",
            tools=[],
            verbose=True,
            allow_delegation=False
        )
        
        # Agent 2: Machine Learning Expert (NEW!)
        self.ml_expert = Agent(
            role="Machine Learning Prediction Specialist",
            goal="Generate ML-based retention time predictions using trained Random Forest and Gradient Boosting models",
            backstory="""You are an expert in machine learning for chemical property prediction.
            You use trained models (Random Forest and Gradient Boosting) that have learned from
            historical GC-MS data. You provide quantitative predictions with confidence intervals.""",
            tools=[self.ml_tool],
            verbose=True,
            allow_delegation=False
        )
        
        # Agent 3: Knowledge Graph Retrieval Expert
        self.kg_retrieval_agent = Agent(
            role="Knowledge Graph Retrieval Specialist",
            goal="Query the molecular knowledge graph to find relevant historical retention time data and similar molecules",
            backstory="""You are an expert in querying graph databases and retrieving relevant 
            molecular data. You understand how to find similar molecules and relevant experimental 
            data that can inform retention time predictions.""",
            tools=[self.rag_tool],
            verbose=True,
            allow_delegation=False
        )
        
        # Agent 4: Chromatography Expert
        self.chromatography_expert = Agent(
            role="GC-MS Chromatography Expert",
            goal="Interpret ML predictions and experimental data to provide expert assessment of retention time",
            backstory="""You are a senior chromatographer with 20+ years of experience in GC-MS. 
            You understand how molecules interact with stationary phases, how temperature programs 
            affect retention, and can assess whether ML predictions align with chromatographic principles.""",
            tools=[],
            verbose=True,
            allow_delegation=True
        )
        
        # Agent 5: Data Validation Expert
        self.validation_expert = Agent(
            role="Experimental Data Validator",
            goal="Compare ML predictions with historical experimental data and assess prediction confidence",
            backstory="""You are a meticulous analytical chemist who validates predictions against 
            experimental evidence. You compare ML model outputs with similar molecules' measured 
            retention times to assess reliability.""",
            tools=[self.rag_tool],
            verbose=True,
            allow_delegation=False
        )
        
        # Agent 6: Prediction Synthesis Coordinator
        self.synthesis_agent = Agent(
            role="Prediction Synthesis Coordinator",
            goal="Synthesize ML predictions, RAG data, and expert analysis into final retention time prediction",
            backstory="""You are an expert in combining multiple sources of evidence including 
            machine learning models, historical data, and expert reasoning. You produce well-calibrated 
            predictions with appropriate uncertainty estimates by weighing all available information.""",
            tools=[],
            verbose=True,
            allow_delegation=False
        )
    
    def create_ml_enhanced_tasks(self, smiles: str, column_type: str = "HP-5MS", 
                                 temperature_program: str = "40°C to 300°C at 10°C/min") -> List[Task]:
        """Create task workflow with ML predictions"""
        
        # Task 1: Molecular Analysis
        molecular_analysis_task = Task(
            description=f"""Analyze the molecular structure of {smiles}.
            
            Extract key molecular properties that influence GC-MS retention time:
            - Molecular weight
            - LogP (lipophilicity)
            - Polarity (TPSA)
            - Functional groups
            - Structural features (aromatic rings, rotatable bonds)
            
            Explain how these properties typically affect retention time on {column_type} columns.
            """,
            agent=self.molecular_analyst,
            expected_output="Detailed molecular analysis with property values and chromatographic implications"
        )
        
        # Task 2: ML Predictions (NEW!)
        ml_prediction_task = Task(
            description=f"""Use the ml_retention_predictor tool to generate ML-based predictions for {smiles}.
            
            Parameters:
            - SMILES: {smiles}
            - Column: {column_type}
            
            Report:
            - Random Forest prediction with confidence interval
            - Gradient Boosting prediction with confidence interval
            - Ensemble average
            - Prediction uncertainty
            
            The ML models were trained on historical GC-MS data and use molecular descriptors 
            to predict retention times.
            """,
            agent=self.ml_expert,
            expected_output="ML model predictions with confidence intervals and uncertainty estimates",
            context=[molecular_analysis_task]
        )
        
        # Task 3: Knowledge Graph Retrieval
        kg_retrieval_task = Task(
            description=f"""Query the molecular knowledge graph for {smiles} using the molecular_knowledge_graph tool.
            
            Retrieve:
            - Similar molecules with known retention times
            - Molecules with similar properties measured on {column_type}
            - Historical retention time patterns
            
            Parameters:
            - SMILES: {smiles}
            - Column: {column_type}
            - Top K: 10
            
            Summarize the most relevant findings from the knowledge graph.
            """,
            agent=self.kg_retrieval_agent,
            expected_output="Summary of retrieved molecular data and retention time patterns",
            context=[molecular_analysis_task]
        )
        
        # Task 4: Chromatographic Assessment
        chromatography_task = Task(
            description=f"""Assess the ML predictions and retrieved data from a chromatography perspective.
            
            Consider:
            - Do the ML predictions (Random Forest: ~X min, Gradient Boosting: ~Y min) make sense?
            - How do they compare to similar molecules' experimental RTs?
            - Are there any chromatographic principles that support or contradict the predictions?
            - What is the effect of {temperature_program} on retention?
            
            Provide expert chromatographic interpretation of the ML model outputs.
            """,
            agent=self.chromatography_expert,
            expected_output="Expert chromatographic assessment of ML predictions",
            context=[molecular_analysis_task, ml_prediction_task, kg_retrieval_task]
        )
        
        # Task 5: Validation
        validation_task = Task(
            description=f"""Validate the ML predictions against experimental data.
            
            Compare:
            - ML model predictions (RF and GB)
            - Retention times of similar molecules from knowledge graph
            - Expected chromatographic behavior
            
            Assess:
            - Agreement between models
            - Consistency with experimental data
            - Prediction confidence level
            - Potential sources of error
            
            Provide a confidence score (0-100%) for the final prediction.
            """,
            agent=self.validation_expert,
            expected_output="Validation report with confidence assessment",
            context=[ml_prediction_task, kg_retrieval_task, chromatography_task]
        )
        
        # Task 6: Final Synthesis
        synthesis_task = Task(
            description=f"""Synthesize all analyses into a final retention time prediction for {smiles}.
            
            Integrate:
            - ML model predictions (Random Forest and Gradient Boosting)
            - RAG-retrieved experimental data
            - Chromatographic expert assessment
            - Validation analysis
            
            Provide:
            - Final predicted retention time (minutes) - use ensemble of ML models weighted by validation
            - 95% confidence interval
            - Prediction confidence score (0-100%)
            - Key factors influencing the prediction
            - Model agreement analysis
            - Limitations and uncertainties
            
            Format the output as a structured prediction report.
            """,
            agent=self.synthesis_agent,
            expected_output="Final prediction report with ML-enhanced retention time, confidence interval, and comprehensive analysis",
            context=[molecular_analysis_task, ml_prediction_task, kg_retrieval_task, 
                    chromatography_task, validation_task]
        )
        
        return [
            molecular_analysis_task,
            ml_prediction_task,
            kg_retrieval_task,
            chromatography_task,
            validation_task,
            synthesis_task
        ]
    
    def predict_retention_time(self, smiles: str, column_type: str = "HP-5MS",
                              temperature_program: str = "40°C to 300°C at 10°C/min") -> Dict:
        """Execute the full ML-enhanced prediction workflow"""
        
        # Create tasks
        tasks = self.create_ml_enhanced_tasks(smiles, column_type, temperature_program)
        
        # Create crew
        crew = Crew(
            agents=[
                self.molecular_analyst,
                self.ml_expert,
                self.kg_retrieval_agent,
                self.chromatography_expert,
                self.validation_expert,
                self.synthesis_agent
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        # Execute workflow
        result = crew.kickoff()
        
        return {
            'smiles': smiles,
            'column_type': column_type,
            'temperature_program': temperature_program,
            'prediction_result': result,
            'ml_enhanced': True
        }


def main():
    """Example usage of ML-enhanced prediction system"""
    
    # Check if models exist
    model_dir = "models"
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        print("="*70)
        print("⚠️  ML Models Not Found")
        print("="*70)
        print("\nYou need to train ML models first. Run:")
        print("  python ml_model_trainer.py")
        print("\nThis will:")
        print("  1. Extract data from Neo4j")
        print("  2. Train Random Forest and Gradient Boosting models")
        print("  3. Save models to ./models/ directory")
        print("\nAfter training, run this script again.")
        return
    
    # Initialize ML-enhanced prediction system
    print("\n" + "="*70)
    print("ML-ENHANCED GC-MS RETENTION TIME PREDICTION")
    print("="*70)
    
    gcms_crew = MLEnhancedGCMSCrew()
    
    # Example molecule: Caffeine
    test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    
    print(f"\nPredicting retention time for: {test_smiles}")
    print("This combines:")
    print("  - Machine Learning models (Random Forest + Gradient Boosting)")
    print("  - RAG retrieval from Neo4j knowledge graph")
    print("  - Multi-agent expert reasoning")
    print("\nThis may take 2-3 minutes...\n")
    
    # Run prediction
    result = gcms_crew.predict_retention_time(
        smiles=test_smiles,
        column_type="HP-5MS",
        temperature_program="40°C to 300°C at 10°C/min"
    )
    
    print("\n" + "="*70)
    print("ML-ENHANCED PREDICTION RESULT")
    print("="*70)
    print(result['prediction_result'])
    
    print("\n" + "="*70)
    print("PREDICTION COMPLETE")
    print("="*70)
    print("\nThe system combined:")
    print("✓ ML model predictions (quantitative)")
    print("✓ Similar molecules from knowledge graph")
    print("✓ Expert chromatographic reasoning")
    print("✓ Validation against experimental data")


if __name__ == "__main__":
    main()
