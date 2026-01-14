from neo4j_schema import Neo4jGCMSSchema, get_neo4j_connection
from dotenv import load_dotenv
load_dotenv()

uri, username, password = get_neo4j_connection()
print(f"Connecting to: {uri}")

schema = Neo4jGCMSSchema(uri, username, password)
print("Creating schema...")
schema.create_schema()
print("✅ Database schema created!")

schema.close()
