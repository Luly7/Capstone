# How to Run - Simple Step-by-Step Guide

## 🚀 Complete Setup and Run Instructions

---

## Step 1: Download the Project (1 minute)

**Download the complete package:**
- Click here: [neo4j_rag_gcms_complete.zip](computer:///mnt/user-data/outputs/neo4j_rag_gcms_complete.zip)
- Save to your computer
- Extract/unzip the file to a folder (e.g., `gcms_project`)

```bash
# On Linux/Mac
unzip neo4j_rag_gcms_complete.zip

# On Windows
# Right-click → Extract All
```

---

## Step 2: Create Neo4j Aura Database (5 minutes)

**This is your cloud database - NO local installation needed!**

### 2.1 Sign Up
1. Go to https://neo4j.com/cloud/aura/
2. Click "Start Free"
3. Sign up with email or Google account

### 2.2 Create Database
1. Click "Create Database" or "New Instance"
2. Select **FREE** tier
3. Name it: `gcms-prediction` (or any name)
4. Choose any cloud provider/region
5. Click "Create"

### 2.3 Save Credentials (CRITICAL!)
A popup will show:
```
Connection URI: neo4j+s://xxxxx.databases.neo4j.io
Username: neo4j
Password: xxxxxxxxxxxxxxxx
```

**⚠️ SAVE THIS PASSWORD NOW!** You can't retrieve it later.
- Click "Download and Continue" (saves a .txt file)
- Or copy/paste to a safe place

### 2.4 Wait for Database
- Status will show "Creating..." then "Running"
- Takes 1-2 minutes

✅ Your cloud database is ready!

---

## Step 3: Install Python Dependencies (2 minutes)

Open terminal/command prompt in your project folder:

```bash
# Navigate to project folder
cd path/to/gcms_project

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

**This installs:**
- neo4j (database driver)
- crewai (multi-agent framework)
- rdkit (chemistry library)
- pandas, numpy, etc.

**Note:** RDKit installation might take a few minutes.

---

## Step 4: Configure Credentials (2 minutes)

### 4.1 Copy Template
```bash
# Mac/Linux
cp .env.template .env

# Windows
copy .env.template .env
```

### 4.2 Edit .env File
Open `.env` in any text editor and add your credentials:

```bash
# Neo4j Aura credentials (from Step 2.3)
NEO4J_URI=neo4j+s://your-xxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-actual-password-here

# OpenAI API key (for CrewAI agents)
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Where to get OpenAI API key:**
1. Go to https://platform.openai.com/api-keys
2. Sign in or create account
3. Click "Create new secret key"
4. Copy and paste into .env

**Save the file!**

---

## Step 5: Run the System! (1 minute)

### Option A: Quick Test (Recommended First)

```bash
python quickstart.py
```

**What this does:**
1. ✅ Checks environment variables
2. ✅ Tests Neo4j connection
3. ✅ Initializes database schema
4. ✅ Loads sample data (10 molecules)
5. ✅ Tests RAG retrieval
6. ✅ Runs a prediction with CrewAI agents

**Expected output:**
```
[1/5] Checking environment configuration...
✅ Environment variables configured

[2/5] Testing Neo4j connection...
✅ Neo4j connection successful

[3/5] Initializing database...
✅ Database initialized with sample data

[4/5] Testing RAG retrieval system...
✅ RAG retrieval successful

[5/5] Testing CrewAI prediction workflow...
✅ CrewAI prediction successful

🎉 QUICK START COMPLETE!
```

**Time:** ~2-3 minutes total

---

### Option B: Make a Single Prediction

```bash
python -c "
from crewai_gcms_prediction import GCMSPredictionCrew

# Initialize the system
crew = GCMSPredictionCrew()

# Predict retention time for caffeine
result = crew.predict_retention_time(
    smiles='CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    column_type='HP-5MS',
    temperature_program='40°C to 300°C at 10°C/min'
)

print(result['prediction_result'])
"
```

**Time:** ~1-2 minutes per prediction

---

### Option C: Interactive Python

```bash
python
```

Then run:

```python
from crewai_gcms_prediction import GCMSPredictionCrew

# Create the crew
crew = GCMSPredictionCrew()

# Make predictions
result = crew.predict_retention_time("CCO")  # Ethanol
print(result)

# Try another
result = crew.predict_retention_time("c1ccccc1")  # Benzene
print(result)
```

---

### Option D: Interactive Jupyter Notebook

```bash
# Install Jupyter if needed
pip install jupyter

# Start notebook
jupyter notebook tutorial_notebook.ipynb
```

Then run cells step-by-step!

---

## Step 6: Load Your Own Data (Optional)

### Prepare CSV File

Create a CSV with your GC-MS data:

```csv
SMILES,RT,Column,TempProgram,FlowRate
CCO,2.3,HP-5MS,40-300C,1.0
CC(C)O,2.8,HP-5MS,40-300C,1.0
CCCCCO,5.6,HP-5MS,40-300C,1.0
```

### Load Data

```python
from data_ingestion import GCMSDataIngestion

# Initialize ingestion
ingestion = GCMSDataIngestion()

# Load your CSV
ingestion.ingest_from_csv('your_data.csv')

# Build similarity graph
ingestion.build_similarity_graph(threshold=0.7)

ingestion.close()
print("Data loaded successfully!")
```

---

## Common Commands Reference

### Database Management

```python
from neo4j_schema import Neo4jGCMSSchema, get_neo4j_connection

# Connect to database
uri, username, password = get_neo4j_connection()
schema = Neo4jGCMSSchema(uri, username, password)

# Create schema (first time only)
schema.create_schema()

# Add a molecule
schema.add_molecule("CCO", properties, embedding)
```

### RAG Retrieval

```python
from rag_retriever import MolecularRAGRetriever
from neo4j_schema import get_neo4j_connection

# Setup retriever
uri, username, password = get_neo4j_connection()
retriever = MolecularRAGRetriever(uri, username, password)

# Get context for a molecule
context = retriever.retrieve_prediction_context(
    query_smiles="c1ccccc1",
    column_type="HP-5MS",
    top_k=10
)

# Format for reading
formatted = retriever.format_context_for_llm(context)
print(formatted)
```

### Visualization

```python
from visualizations import GCMSVisualizer

viz = GCMSVisualizer()

# Generate plots
viz.plot_molecular_space()
viz.plot_rt_distribution("HP-5MS")
viz.generate_summary_report()

viz.close()
```

---

## Troubleshooting

### Problem: "Cannot connect to Neo4j"

**Solutions:**
1. Check database is "Running" in Aura console: https://console.neo4j.io/
2. Verify credentials in `.env` match Aura exactly
3. Make sure URI includes `neo4j+s://` (not `neo4j://`)
4. Check internet connection

### Problem: "Module not found"

**Solutions:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall packages
pip install -r requirements.txt
```

### Problem: "OpenAI API error"

**Solutions:**
1. Check API key in `.env` is correct
2. Verify you have credits: https://platform.openai.com/usage
3. Check key starts with `sk-`

### Problem: "RDKit installation failed"

**Solutions:**
```bash
# Try with conda instead
conda install -c conda-forge rdkit

# Or use pip with specific version
pip install rdkit-pypi
```

### Problem: "No data in database"

**Solutions:**
```bash
# Run the data ingestion script
python data_ingestion.py

# This loads sample data
```

---

## What Each Script Does

**quickstart.py** - Tests everything (run this first!)
```bash
python quickstart.py
```

**data_ingestion.py** - Creates schema & loads sample data
```bash
python data_ingestion.py
```

**visualizations.py** - Generates plots
```bash
python visualizations.py
```

**crewai_gcms_prediction.py** - Main prediction system
```python
from crewai_gcms_prediction import GCMSPredictionCrew
crew = GCMSPredictionCrew()
result = crew.predict_retention_time("CCO")
```

---

## File Locations Explained

**Your files are in:** `/path/to/gcms_project/`

**Important files:**
- `.env` - Your credentials (keep private!)
- `requirements.txt` - Dependencies
- `*.py` - Python code
- `*.md` - Documentation
- `tutorial_notebook.ipynb` - Interactive tutorial

**Nothing is installed globally** - everything is in your project folder.

---

## Next Steps After Running

1. ✅ **Verify it works** - Run `python quickstart.py`

2. 📊 **Explore the database:**
   - Go to https://console.neo4j.io/
   - Click your database → "Open" → "Neo4j Browser"
   - Run queries from `CYPHER_QUERIES.md`

3. 🧪 **Try predictions:**
   - Use the examples above
   - Try your own SMILES strings
   - Load your own data

4. 📖 **Read documentation:**
   - `README.md` - Complete guide
   - `PROJECT_OVERVIEW.md` - Technical details
   - `EXECUTIVE_SUMMARY.md` - For presentation

5. 🎓 **Prepare presentation:**
   - Read `EXECUTIVE_SUMMARY.md`
   - Practice with `python quickstart.py`
   - Generate visuals with `visualizations.py`

---

## Quick Reference Card

```bash
# Setup (once)
pip install -r requirements.txt
cp .env.template .env
# Edit .env with credentials

# Test everything
python quickstart.py

# Load data
python data_ingestion.py

# Make prediction
python -c "from crewai_gcms_prediction import GCMSPredictionCrew; \
  crew = GCMSPredictionCrew(); \
  print(crew.predict_retention_time('CCO'))"

# Generate plots
python visualizations.py

# Interactive mode
python
>>> from crewai_gcms_prediction import GCMSPredictionCrew
>>> crew = GCMSPredictionCrew()
>>> result = crew.predict_retention_time("c1ccccc1")
```

---

## Expected Timeline

**First time setup:**
- Download & extract: 1 min
- Create Neo4j Aura: 5 min
- Install dependencies: 5 min
- Configure .env: 2 min
- Test with quickstart.py: 3 min
**Total: ~15 minutes**

**Subsequent runs:**
- Make prediction: 1-2 min
- Load new data: 1 min per 100 molecules
- Generate plots: 30 sec

---

## Need More Help?

1. **Setup issues** → `NEO4J_AURA_SETUP.md`
2. **Usage examples** → `tutorial_notebook.ipynb`
3. **Database queries** → `CYPHER_QUERIES.md`
4. **Understanding code** → `PROJECT_OVERVIEW.md`
5. **Presentation** → `EXECUTIVE_SUMMARY.md`

---

## Success Indicators

You'll know it's working when:

✅ `python quickstart.py` completes all 5 steps  
✅ You see "Neo4j connection successful"  
✅ CrewAI agents run and produce predictions  
✅ Database shows data in Neo4j Browser  
✅ Visualizations generate without errors  

---

## You're All Set! 🚀

**Your system is ready to:**
- Predict retention times
- Query molecular knowledge graph
- Run multi-agent analysis
- Generate visualizations
- Process your GC-MS data

**Good luck with your capstone! 🎓**
