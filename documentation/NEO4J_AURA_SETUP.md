# Neo4j Aura Setup Guide
## Complete Instructions for Creating Your Cloud Database

---

## What is Neo4j Aura?

Neo4j Aura is a **fully-managed cloud database** - there's nothing to install locally. You create your database through their web interface, and it runs in the cloud. Your Python code connects to it remotely.

---

## Step-by-Step Setup

### 1. Create Free Neo4j Aura Account

**Go to:** https://neo4j.com/cloud/aura/

**Option A: Sign up with Email**
1. Click "Start Free"
2. Enter your email and create a password
3. Verify your email

**Option B: Sign up with Google** (faster)
1. Click "Sign in with Google"
2. Select your Google account
3. Done!

---

### 2. Create Your Database Instance

After logging in:

1. **Click "Create Database"** (or "New Instance")

2. **Select the FREE tier:**
   - Name: `gcms-prediction` (or any name you want)
   - Cloud Provider: Choose any (AWS, Google Cloud, Azure)
   - Region: Choose closest to you
   - Size: **Free** (0.5 GB storage, good for 10K+ molecules)

3. **Click "Create"**

4. **CRITICAL: Save Your Credentials!**
   
   A popup will appear with:
   ```
   Connection URI: neo4j+s://xxxxx.databases.neo4j.io
   Username: neo4j
   Password: xxxxxxxxxxxxxxxx
   ```
   
   **⚠️ SAVE THIS PASSWORD NOW - You can't retrieve it later!**
   
   Options to save:
   - Click "Download and continue" (saves a .txt file)
   - Or copy to password manager
   - Or write it down temporarily

5. **Wait 1-2 minutes** for the database to spin up
   - Status will show "Running" when ready
   - The instance will appear in your dashboard

---

### 3. Configure Your Project

Now that your cloud database is running, configure your local project:

```bash
# Copy the template
cp .env.template .env

# Edit .env with your saved credentials
nano .env  # or use any text editor
```

Put your actual values in `.env`:
```bash
NEO4J_URI=neo4j+s://your-actual-uri.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-actual-password-from-step-2
OPENAI_API_KEY=sk-your-openai-key
```

---

### 4. Test the Connection

```bash
# Run the quickstart script
python quickstart.py
```

You should see:
```
✅ Neo4j connection successful
✅ Database initialized with sample data
```

---

## Access Your Database

### Web Browser Access (Neo4j Browser)

1. Go to your Aura Console: https://console.neo4j.io/
2. Click on your database instance
3. Click "Open" → "Neo4j Browser"
4. Enter your password when prompted

Now you can run Cypher queries directly!

---

## Useful Cypher Queries to Run in Browser

Once in Neo4j Browser, try these queries:

### See All Molecules
```cypher
MATCH (m:Molecule)
RETURN m.smiles, m.molecular_weight, m.logp
LIMIT 10
```

### Count Your Data
```cypher
// Count molecules
MATCH (m:Molecule) RETURN count(m) as molecule_count

// Count retention times
MATCH (r:RetentionTime) RETURN count(r) as rt_count

// Count molecular features
MATCH (f:MolecularFeature) RETURN count(f) as feature_count
```

### See Molecular Similarity Network
```cypher
MATCH (m1:Molecule)-[s:SIMILAR_TO]->(m2:Molecule)
RETURN m1.smiles, m2.smiles, s.similarity_score
ORDER BY s.similarity_score DESC
LIMIT 20
```

### Find Similar Molecules to Benzene
```cypher
MATCH (m:Molecule {smiles: "c1ccccc1"})-[s:SIMILAR_TO]->(similar:Molecule)
RETURN similar.smiles, s.similarity_score
ORDER BY s.similarity_score DESC
LIMIT 5
```

### Molecules with Retention Times
```cypher
MATCH (m:Molecule)-[:MEASURED_ON]->(r:RetentionTime)-[:USING_COLUMN]->(c:Column)
RETURN m.smiles, r.rt_minutes, c.type
ORDER BY r.rt_minutes
```

### Molecular Features for a Compound
```cypher
MATCH (m:Molecule {smiles: "CCO"})-[hf:HAS_FEATURE]->(f:MolecularFeature)
RETURN f.name, hf.value
ORDER BY f.name
```

### Search by Molecular Weight Range
```cypher
MATCH (m:Molecule)
WHERE m.molecular_weight >= 100 AND m.molecular_weight <= 200
RETURN m.smiles, m.molecular_weight, m.logp
ORDER BY m.molecular_weight
```

---

## Understanding Your Database

### Node Types
- **Molecule**: Your compounds with properties and embeddings
- **MolecularFeature**: RDKit descriptors (Chi0v, Kappa1, etc.)
- **RetentionTime**: Experimental RT measurements
- **Column**: GC column specifications

### Relationship Types
- **HAS_FEATURE**: Links molecules to their features
- **MEASURED_ON**: Links molecules to RT measurements
- **USING_COLUMN**: Links RT to the column used
- **SIMILAR_TO**: Similarity relationships between molecules

---

## Managing Your Database

### Aura Console (https://console.neo4j.io/)

**View Database:**
- See database status (Running/Stopped)
- View connection details
- Monitor usage

**Pause/Resume:**
- Free tier: Auto-pauses after 3 days inactivity
- Click "Resume" to restart
- No data is lost when paused

**Delete Database:**
- Click "..." menu → "Delete"
- ⚠️ This is permanent!

**Export Data:**
- Click "..." menu → "Export"
- Downloads all data as CSV files

---

## No Local Installation Needed!

Unlike traditional databases, you **DO NOT**:
- ❌ Install Neo4j on your computer
- ❌ Run `neo4j start` commands
- ❌ Manage a local server
- ❌ Configure ports or networking

You **ONLY**:
- ✅ Create instance in web interface
- ✅ Copy connection credentials to .env
- ✅ Run your Python code
- ✅ Python connects to cloud database automatically

---

## Troubleshooting

### "Connection Refused" Error

**Problem:** Can't connect to Neo4j
**Solutions:**
1. Check database is "Running" in Aura console
2. Verify URI in .env matches Aura console exactly
3. Ensure password is correct (try resetting if needed)
4. Check internet connection

### "Authentication Failed"

**Problem:** Wrong password
**Solutions:**
1. In Aura console, click "..." → "Reset password"
2. Get new password
3. Update .env file
4. Try connecting again

### Database Not Showing Up

**Problem:** Can't see your database
**Solutions:**
1. Make sure you're logged into correct Neo4j account
2. Check your email for the region it was created in
3. Database might still be creating (wait 2-3 min)

### Free Tier Limitations

- **Storage:** 0.5 GB (enough for ~10K-50K molecules)
- **Auto-pause:** After 3 days of inactivity
- **Memory:** Limited concurrent queries
- **Solution:** Upgrade to paid tier if needed ($65/month for 2GB)

---

## Resetting Your Password

If you lost your password:

1. Go to Aura Console
2. Click on your database
3. Click "..." menu → "Reset password"
4. Copy new password
5. Update .env file
6. Restart your application

---

## Monitoring Usage

In Aura Console, click on your database to see:
- Storage used (e.g., "45 MB / 500 MB")
- Number of nodes
- Number of relationships
- Query performance

---

## Connecting from Python

The Python code handles all connection details:

```python
# Your code just needs the credentials in .env
from neo4j_schema import get_neo4j_connection

uri, username, password = get_neo4j_connection()
# Automatically reads from .env file

# Creates cloud connection
from neo4j import GraphDatabase
driver = GraphDatabase.driver(uri, auth=(username, password))
```

That's it! The Python neo4j driver connects to your cloud database automatically.

---

## Summary

**Neo4j Aura = Cloud Database**
- No local installation
- No command-line setup
- Everything through web interface
- Python connects remotely

**Setup Steps:**
1. Create account at neo4j.com/cloud/aura (2 min)
2. Create free database instance (1 min)
3. Save credentials (30 sec)
4. Put credentials in .env (30 sec)
5. Run `python quickstart.py` (1 min)

**Total time: ~5 minutes**

-
## Next Steps

Now that your cloud database is set up:

1. ✅ Run `python data_ingestion.py` to populate with sample data
2. ✅ Run `python quickstart.py` to test everything
3. ✅ Open Neo4j Browser and explore with Cypher queries
4. ✅ Try `python crewai_gcms_prediction.py` for predictions
5. ✅ Load your own data with the data ingestion scripts

Your Neo4j Aura instance is ready! 🚀
