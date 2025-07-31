from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from dotenv import load_dotenv
from app.prompt import (
    DATA_PROFILING_PROMPT,
    STATISTICAL_ANALYSIS_PROMPT,
    VISUALIZATION_PROMPT,
    DATA_TRANSFORMATION_PROMPT
)
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import tempfile


app = FastAPI(title="AnalyticsPro Multi-Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# --- AGENT CLASSES (simples, prompts importés) ---
class BaseAgent:
    def __init__(self, dataframe: pd.DataFrame, openai_api_key: str, system_prompt: str):
        self.df = dataframe
        self.llm = OpenAI(
            api_key=openai_api_key,
            temperature=0.1,
            max_tokens=1500
        )
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            allow_dangerous_code=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=8,
            prefix=system_prompt
        )
    def run(self, question: str):
        try:
            result = self.agent.run(question)
            if isinstance(result, str) and "Agent stopped due to iteration limit" in result:
                return ("⏳ Question trop complexe. Essayez de la simplifier.")
            return result
        except Exception as e:
            return f"⚠️ Erreur: {str(e)[:200]}..."

AGENT_PROMPTS = {
    "profiling": DATA_PROFILING_PROMPT,
    "statistical": STATISTICAL_ANALYSIS_PROMPT,
    "visualization": VISUALIZATION_PROMPT,
    "transformation": DATA_TRANSFORMATION_PROMPT
}

# --- INTENT CLASSIFICATION (simple mots-clés) ---
def classify_intent(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["graphique", "visualis", "chart", "plot", "graph", "affich", "montre", "dessine"]):
        return "visualization"
    if any(k in q for k in ["corrélation", "moyenne", "médiane", "écart", "distribution", "test", "significatif", "variance", "régression"]):
        return "statistical"
    if any(k in q for k in ["qualité", "profil", "résumé", "aperçu", "manquant", "aberrant", "complet", "overview"]):
        return "profiling"
    if any(k in q for k in ["nettoyer", "transformer", "créer", "modifier", "encoder", "normaliser"]):
        return "transformation"
    return "profiling"


# --- ROUTES ---
@app.post("/analyze/")
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return JSONResponse(status_code=400, content={"error": "Clé API OpenAI manquante."})
    # Lecture du fichier
    try:
        suffix = os.path.splitext(file.filename)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        if suffix == ".csv":
            df = pd.read_csv(tmp_path)
        else:
            df = pd.read_excel(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Erreur de lecture du fichier: {str(e)}"})
    # Classification et exécution
    intent = classify_intent(question)
    prompt = AGENT_PROMPTS[intent]
    agent = BaseAgent(df, openai_api_key, prompt)
    result = agent.run(question)
    return {"agent": intent, "response": result}

# --- Résumé du dataset (dashboard) ---
@app.post("/dataset/summary")
async def dataset_summary(file: UploadFile = File(...)):
    try:
        suffix = os.path.splitext(file.filename)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        if suffix == ".csv":
            df = pd.read_csv(tmp_path)
        else:
            df = pd.read_excel(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Erreur de lecture du fichier: {str(e)}"})
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing": {col: int(df[col].isnull().sum()) for col in df.columns},
        "sample": df.head(10).to_dict(orient="records")
    }
    return summary

# --- Visualisation structurée (exemple: histogramme) ---
@app.post("/visualize/")
async def visualize(
    file: UploadFile = File(...),
    column: str = Form(...),
    chart_type: str = Form("histogram")
):
    try:
        suffix = os.path.splitext(file.filename)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        if suffix == ".csv":
            df = pd.read_csv(tmp_path)
        else:
            df = pd.read_excel(tmp_path)
        os.unlink(tmp_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Erreur de lecture du fichier: {str(e)}"})
    if chart_type == "histogram":
        try:
            data = df[column].dropna().tolist()
            import numpy as np
            hist, bin_edges = np.histogram(data, bins=20)
            return {
                "type": "histogram",
                "column": column,
                "bins": bin_edges.tolist(),
                "counts": hist.tolist()
            }
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Erreur histogramme: {str(e)}"})
    return JSONResponse(status_code=400, content={"error": "Type de graphique non supporté."})

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API AnalyticsPro Multi-Agent!"}
