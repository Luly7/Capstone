# fix it
import re

with open('crewai_gcms_prediction.py', 'r') as f:
    content = f.read()

# Find and replace the RAGRetrievalTool class
old_class = '''class RAGRetrievalTool(BaseTool):
    """Custom CrewAI tool for Neo4j RAG retrieval"""
    name: str = "molecular_knowledge_graph"
    description: str = "Retrieves relevant molecular data and retention times from Neo4j knowledge graph"
    
    def __init__(self):
        super().__init__()
        uri, username, password = get_neo4j_connection()
        self.retriever = MolecularRAGRetriever(uri, username, password)'''

new_class = '''class RAGRetrievalTool(BaseTool):
    """Custom CrewAI tool for Neo4j RAG retrieval"""
    name: str = "molecular_knowledge_graph"
    description: str = "Retrieves relevant molecular data and retention times from Neo4j knowledge graph"
    
    class Config:
        arbitrary_types_allowed = True
    
    _retriever: MolecularRAGRetriever = None
    
    def __init__(self):
        super().__init__()
        uri, username, password = get_neo4j_connection()
        object.__setattr__(self, '_retriever', MolecularRAGRetriever(uri, username, password))'''

content = content.replace(old_class, new_class)

# Also fix the reference in _run method
content = content.replace('self.retriever.retrieve_prediction_context',
                          'self._retriever.retrieve_prediction_context')
content = content.replace(
    'self.retriever.format_context_for_llm', 'self._retriever.format_context_for_llm')

with open('crewai_gcms_prediction.py', 'w') as f:
    f.write(content)

print("✅ Fixed RAGRetrievalTool!")
