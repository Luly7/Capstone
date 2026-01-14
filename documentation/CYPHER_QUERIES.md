# Cypher Query Reference for GC-MS Knowledge Graph
## Run these in Neo4j Browser (Aura Console → Open → Neo4j Browser)

---

## Database Inspection Queries

### Count All Nodes
```cypher
// Count all nodes by type
MATCH (n)
RETURN labels(n) as NodeType, count(n) as Count
ORDER BY Count DESC
```

### Count All Relationships
```cypher
// Count all relationships by type
MATCH ()-[r]->()
RETURN type(r) as RelationshipType, count(r) as Count
ORDER BY Count DESC
```

### Database Summary
```cypher
// Complete database statistics
CALL apoc.meta.stats() YIELD nodeCount, relCount, labels, relTypes
RETURN nodeCount, relCount, labels, relTypes
```

---

## Molecule Queries

### List All Molecules
```cypher
MATCH (m:Molecule)
RETURN m.smiles, m.molecular_weight, m.logp, m.tpsa
ORDER BY m.molecular_weight
LIMIT 20
```

### Find Molecule by SMILES
```cypher
MATCH (m:Molecule {smiles: "CCO"})  // Change to your SMILES
RETURN m
```

### Search by Molecular Weight Range
```cypher
MATCH (m:Molecule)
WHERE m.molecular_weight >= 100 AND m.molecular_weight <= 200
RETURN m.smiles, m.molecular_weight, m.logp
ORDER BY m.molecular_weight
LIMIT 20
```

### Search by LogP Range
```cypher
MATCH (m:Molecule)
WHERE m.logp >= 1.0 AND m.logp <= 3.0
RETURN m.smiles, m.molecular_weight, m.logp, m.tpsa
ORDER BY m.logp
LIMIT 20
```

### Most Polar Molecules (Highest TPSA)
```cypher
MATCH (m:Molecule)
WHERE m.tpsa IS NOT NULL
RETURN m.smiles, m.tpsa, m.molecular_weight
ORDER BY m.tpsa DESC
LIMIT 10
```

### Most Lipophilic Molecules (Highest LogP)
```cypher
MATCH (m:Molecule)
WHERE m.logp IS NOT NULL
RETURN m.smiles, m.logp, m.molecular_weight
ORDER BY m.logp DESC
LIMIT 10
```

---

## Molecular Feature Queries

### All Features for a Molecule
```cypher
MATCH (m:Molecule {smiles: "CCO"})-[hf:HAS_FEATURE]->(f:MolecularFeature)
RETURN f.name as Feature, hf.value as Value
ORDER BY f.name
```

### Molecules with High Chi0v
```cypher
MATCH (m:Molecule)-[hf:HAS_FEATURE]->(f:MolecularFeature {name: "Chi0v"})
WHERE hf.value > 5.0
RETURN m.smiles, hf.value as Chi0v
ORDER BY hf.value DESC
LIMIT 10
```

### Feature Distribution
```cypher
MATCH (m:Molecule)-[hf:HAS_FEATURE]->(f:MolecularFeature {name: "MolecularWeight"})
RETURN min(hf.value) as Min, max(hf.value) as Max, avg(hf.value) as Avg
```

---

## Retention Time Queries

### All Retention Times
```cypher
MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column)
RETURN m.smiles, r.rt_minutes, c.type, r.temperature_program
ORDER BY r.rt_minutes
```

### Retention Times by Column
```cypher
MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column {type: "HP-5MS"})
RETURN m.smiles, r.rt_minutes, m.molecular_weight, m.logp
ORDER BY r.rt_minutes
```

### RT Statistics by Column
```cypher
MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column)
RETURN c.type as Column, 
       count(r) as Measurements,
       min(r.rt_minutes) as MinRT,
       max(r.rt_minutes) as MaxRT,
       avg(r.rt_minutes) as AvgRT
ORDER BY c.type
```

### Molecules Without Retention Times
```cypher
MATCH (m:Molecule)
WHERE NOT (m)-[:MEASURED_ON]->()
RETURN m.smiles, m.molecular_weight, m.logp
LIMIT 20
```

### Fastest Eluting Compounds
```cypher
MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)
RETURN m.smiles, r.rt_minutes, m.molecular_weight
ORDER BY r.rt_minutes ASC
LIMIT 10
```

### Slowest Eluting Compounds
```cypher
MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)
RETURN m.smiles, r.rt_minutes, m.molecular_weight
ORDER BY r.rt_minutes DESC
LIMIT 10
```

---

## Similarity Queries

### Most Similar Molecules to a Compound
```cypher
MATCH (m:Molecule {smiles: "c1ccccc1"})-[s:SIMILAR_TO]->(similar:Molecule)
RETURN similar.smiles, s.similarity_score
ORDER BY s.similarity_score DESC
LIMIT 10
```

### Similarity Network (Graph View)
```cypher
MATCH (m1:Molecule)-[s:SIMILAR_TO]->(m2:Molecule)
WHERE s.similarity_score > 0.8
RETURN m1, s, m2
LIMIT 50
```

### Find Molecules Similar to Query with Known RT
```cypher
MATCH (query:Molecule {smiles: "CCO"})-[s:SIMILAR_TO]->(similar:Molecule)
MATCH (similar)-[:MEASURED_ON]->(r:RetentionTime)
RETURN similar.smiles, s.similarity_score, r.rt_minutes
ORDER BY s.similarity_score DESC
LIMIT 10
```

### Average RT of Similar Molecules
```cypher
MATCH (query:Molecule {smiles: "CCO"})-[s:SIMILAR_TO]->(similar:Molecule)
MATCH (similar)-[:MEASURED_ON]->(r:RetentionTime)
RETURN avg(r.rt_minutes) as AvgRT, count(similar) as NumSimilar
```

---

## Property Correlation Queries

### MW vs RT Correlation
```cypher
MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column {type: "HP-5MS"})
RETURN m.molecular_weight, r.rt_minutes
ORDER BY m.molecular_weight
```

### LogP vs RT Correlation
```cypher
MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column {type: "HP-5MS"})
RETURN m.logp, r.rt_minutes
ORDER BY m.logp
```

### Multi-Property Analysis
```cypher
MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)
RETURN m.smiles, 
       m.molecular_weight, 
       m.logp, 
       m.tpsa,
       m.num_rotatable_bonds,
       r.rt_minutes
ORDER BY r.rt_minutes
```

---

## Advanced Queries

### Find Neighbors of Neighbors
```cypher
MATCH (m:Molecule {smiles: "CCO"})-[:SIMILAR_TO*1..2]->(similar:Molecule)
RETURN DISTINCT similar.smiles, m.smiles
LIMIT 20
```

### Molecules with Similar Properties
```cypher
// Find molecules with similar MW and LogP
MATCH (target:Molecule {smiles: "CCO"})
MATCH (similar:Molecule)
WHERE similar <> target
  AND abs(similar.molecular_weight - target.molecular_weight) < 20
  AND abs(similar.logp - target.logp) < 1.0
RETURN similar.smiles, 
       similar.molecular_weight, 
       similar.logp,
       target.molecular_weight as target_mw,
       target.logp as target_logp
LIMIT 10
```

### Complex Similarity + RT Query
```cypher
MATCH (query:Molecule {smiles: "c1ccccc1"})-[s:SIMILAR_TO]->(similar:Molecule)
MATCH (similar)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column)
WHERE s.similarity_score > 0.7
RETURN similar.smiles, 
       s.similarity_score, 
       r.rt_minutes, 
       c.type
ORDER BY s.similarity_score DESC
```

### Shortest Path Between Molecules
```cypher
MATCH path = shortestPath(
  (m1:Molecule {smiles: "CCO"})-[:SIMILAR_TO*]-(m2:Molecule {smiles: "c1ccccc1"})
)
RETURN path
```

---

## Data Quality Queries

### Molecules Without Features
```cypher
MATCH (m:Molecule)
WHERE NOT (m)-[:HAS_FEATURE]->()
RETURN m.smiles
LIMIT 20
```

### Duplicate SMILES Check
```cypher
MATCH (m:Molecule)
WITH m.smiles as smiles, collect(m) as molecules
WHERE size(molecules) > 1
RETURN smiles, size(molecules) as count
```

### Missing Properties
```cypher
MATCH (m:Molecule)
WHERE m.molecular_weight IS NULL 
   OR m.logp IS NULL 
   OR m.tpsa IS NULL
RETURN m.smiles, 
       m.molecular_weight IS NULL as missing_mw,
       m.logp IS NULL as missing_logp,
       m.tpsa IS NULL as missing_tpsa
LIMIT 20
```

---

## Data Modification Queries

### Update Molecule Property
```cypher
MATCH (m:Molecule {smiles: "CCO"})
SET m.custom_property = "value"
RETURN m
```

### Add New Relationship
```cypher
MATCH (m1:Molecule {smiles: "CCO"})
MATCH (m2:Molecule {smiles: "CC(C)O"})
CREATE (m1)-[s:SIMILAR_TO {similarity_score: 0.85}]->(m2)
RETURN m1, s, m2
```

### Delete Old Data
```cypher
// BE CAREFUL - This deletes data!
MATCH (m:Molecule {smiles: "to_delete"})
DETACH DELETE m
```

---

## Visualization Queries

### Small Network Graph
```cypher
MATCH (m:Molecule)-[s:SIMILAR_TO]->(similar:Molecule)
WHERE s.similarity_score > 0.8
RETURN m, s, similar
LIMIT 25
```

### Molecule with All Relationships
```cypher
MATCH (m:Molecule {smiles: "CCO"})
OPTIONAL MATCH (m)-[r1]->(n1)
OPTIONAL MATCH (m)<-[r2]-(n2)
RETURN m, r1, n1, r2, n2
```

### Column Usage Network
```cypher
MATCH (c:Column)<-[:USING_COLUMN]-(r:RetentionTime)<-[:MEASURED_ON]-(m:Molecule)
RETURN c, r, m
LIMIT 50
```

---

## Performance Queries

### Check Indexes
```cypher
SHOW INDEXES
```

### Check Constraints
```cypher
SHOW CONSTRAINTS
```

### Explain Query Plan
```cypher
EXPLAIN
MATCH (m:Molecule {smiles: "CCO"})
RETURN m
```

### Profile Query Performance
```cypher
PROFILE
MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)
RETURN m.smiles, r.rt_minutes
LIMIT 100
```

---

## Vector Search Queries

### Vector Similarity (if vector index exists)
```cypher
// This uses the vector index for fast similarity search
CALL db.index.vector.queryNodes('molecule_embedding', 10, $embedding_vector)
YIELD node, score
RETURN node.smiles, score
ORDER BY score DESC
```

---

## Export Queries

### Export All Molecules to CSV
```cypher
MATCH (m:Molecule)
RETURN m.smiles, m.molecular_weight, m.logp, m.tpsa
```

### Export RT Data
```cypher
MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column)
RETURN m.smiles, r.rt_minutes, c.type, r.temperature_program
```

---

## Tips

1. **Run queries in Neo4j Browser** (not in Python)
2. **Start with LIMIT** to avoid huge results
3. **Use EXPLAIN/PROFILE** to optimize slow queries
4. **Check indexes** if queries are slow
5. **Graph view** works best with <100 nodes
6. **Table view** better for large result sets
7. **Click nodes** in graph to see properties
8. **Double-click relationships** to see details

---

## Common Patterns

### Pattern 1: Find Related Data
```cypher
MATCH (start)-[relationship]->(end)
WHERE start.property = "value"
RETURN start, relationship, end
```

### Pattern 2: Count Relationships
```cypher
MATCH (n)-[r]-()
RETURN n, count(r) as num_relationships
ORDER BY num_relationships DESC
```

### Pattern 3: Filter by Properties
```cypher
MATCH (n:Label)
WHERE n.property > value
  AND n.other_property < value2
RETURN n
```

### Pattern 4: Aggregate Data
```cypher
MATCH (n)-[r]->(m)
RETURN n, count(m) as count, avg(m.property) as average
```

---

## Questions?

All these queries run in **Neo4j Browser**:
1. Go to https://console.neo4j.io/
2. Click your database
3. Click "Open" → "Neo4j Browser"
4. Paste query and press Enter or click play button ▶

You can also modify queries to fit your specific needs!
