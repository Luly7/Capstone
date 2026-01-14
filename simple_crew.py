
"""
Simplified GC-MS Prediction using CrewAI
Direct integration without custom tools
"""
import os
from rag_retriever import MolecularRAGRetriever
from neo4j_schema import get_neo4j_connection
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
load_dotenv()


class SimplifiedGCMSCrew:
    def __init__(self):
        # Initialize RAG retriever
        uri, username, password = get_neo4j_connection()
        self.retriever = MolecularRAGRetriever(uri, username, password)

        # Create agent without custom tools
        self.prediction_agent = Agent(
            role='GC-MS Retention Time Expert',
            goal='Predict GC-MS retention times based on molecular similarity',
            backstory='Expert analytical chemist specializing in chromatography',
            verbose=True,
            llm='gpt-4'
        )

    def predict(self, smiles: str, column_type: str = "HP-5MS"):
        # Get context from RAG
        print(f"\n🔍 Retrieving similar molecules for: {smiles}")
        context = self.retriever.retrieve_prediction_context(
            query_smiles=smiles,
            column_type=column_type,
            top_k=10
        )

        # Format context for LLM
        formatted_context = self.retriever.format_context_for_llm(context)

        # Create prediction task
        task = Task(
            description=f"""
            Based on the following molecular data, predict the GC-MS retention time:
            
            {formatted_context}
            
            Provide:
            1. Predicted retention time (in minutes)
            2. Confidence level (High/Medium/Low)
            3. Key reasoning factors
            """,
            agent=self.prediction_agent,
            expected_output="Retention time prediction with confidence and reasoning"
        )

        # Run prediction
        crew = Crew(agents=[self.prediction_agent], tasks=[task], verbose=True)
        result = crew.kickoff()

        return result


if __name__ == "__main__":
    crew = SimplifiedGCMSCrew()

    # Test with caffeine
    test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    print(f"\n{'='*70}")
    print(f"Testing GC-MS Prediction for Caffeine")
    print(f"SMILES: {test_smiles}")
    print(f"{'='*70}")

    result = crew.predict(test_smiles)

    print(f"\n{'='*70}")
    print(f"PREDICTION RESULT:")
    print(f"{'='*70}")
    print(result)
