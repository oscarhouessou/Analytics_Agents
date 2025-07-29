import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from datetime import datetime
import warnings
from typing import Dict, List, Any, Tuple
warnings.filterwarnings('ignore')
from app.prompt import SUPERVISOR_ANALYST_PROMPT
import subprocess
import socket

# Configuration de la page
st.set_page_config(
    page_title="AnalyticsPro - Self-Service Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)



# Vérifie si l'API FastAPI est déjà en cours d'exécution sur le port donné
def is_api_running(host="127.0.0.1", port=8000):
    """Vérifie si l'API FastAPI est déjà en cours d'exécution sur le port donné."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
            return True
        except Exception:
            return False

# Démarrage automatique de l'API FastAPI si non lancée
if not is_api_running():
    try:
        subprocess.Popen([
            "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import time
        time.sleep(2)  # Laisse le temps à l'API de démarrer
    except Exception as e:
        st.warning(f"Impossible de démarrer l'API FastAPI automatiquement : {e}")
# CSS personnalisé (identique à votre version)
with open(os.path.join(os.path.dirname(__file__), "style.css")) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Chargement des variables d'environnement
load_dotenv()

# Initialisation de la session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

# ===============================
# AGENTS SPÉCIALISÉS
# ===============================

class BaseAgent:
    """Classe de base pour tous les agents"""
    
    def __init__(self, dataframe: pd.DataFrame, openai_api_key: str):
        self.df = dataframe
        self.llm = OpenAI(
            api_key=openai_api_key,
            temperature=0.1,
            max_tokens=1500
        )
        self.agent_name = self.__class__.__name__
        
    def _create_pandas_agent(self, system_prompt: str):
        """Crée un agent pandas avec un prompt système personnalisé"""
        return create_pandas_dataframe_agent(
            self.llm,
            self.df,
            verbose=True,
            allow_dangerous_code=True,  # Obligatoire pour LangChain agent
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            max_iterations=8,  # Réduit pour éviter les timeouts
            prefix=system_prompt
        )
    
    def _safe_execute(self, question: str) -> str:
        """Exécution sécurisée avec gestion d'erreurs"""
        try:
            result = self.agent.run(question)
            if isinstance(result, str) and "Agent stopped due to iteration limit" in result:
                return ("⏳ Question trop complexe. Essayez de la simplifier.")
            return result
        except Exception as e:
            return f"⚠️ Erreur dans {self.agent_name}: {str(e)[:200]}..."



class SupervisorAnalystAgent(BaseAgent):
    """Agent superviseur qui synthétise et valide les analyses des autres agents"""
    def __init__(self, dataframe: pd.DataFrame, openai_api_key: str):
        super().__init__(dataframe, openai_api_key)
        self.agent = self._create_pandas_agent(SUPERVISOR_ANALYST_PROMPT)

    def synthesize_and_validate(self, analyses: dict) -> str:
        """Synthétise et valide les résultats des autres agents"""
        # Construit un message structuré à partir des résultats des autres agents
        message = "Voici les résultats des analyses spécialisées :\n"
        for agent_name, result in analyses.items():
            message += f"\n---\n[{agent_name.upper()}]\n{result}\n"
        message += "\nMerci de synthétiser, valider et formuler des recommandations globales."
        return self._safe_execute(message)

class DataProfilingAgent(BaseAgent):
    """Agent spécialisé dans l'analyse de qualité et profiling des données"""
    
    def __init__(self, dataframe: pd.DataFrame, openai_api_key: str):
        super().__init__(dataframe, openai_api_key)
        
        system_prompt = """
        Vous êtes un expert en qualité des données et profiling. Votre rôle est d'analyser:
        - La structure et les types de données
        - Les valeurs manquantes et aberrantes
        - La distribution des données
        - La qualité générale du dataset
        
        Donnez des réponses structurées avec des métriques précises.
        """
        
        self.agent = self._create_pandas_agent(system_prompt)
    
    def profile_data(self) -> str:
        """Génère un profil complet des données"""
        question = """
        Analysez ce dataset et donnez-moi:
        1. Nombre de lignes et colonnes
        2. Types de données par colonne
        3. Pourcentage de valeurs manquantes par colonne
        4. Statistiques descriptives pour les variables numériques
        5. Cardinalité des variables catégorielles
        6. Identification des valeurs aberrantes potentielles
        """
        return self._safe_execute(question)
    
    def check_data_quality(self) -> str:
        """Évalue la qualité des données"""
        question = """
        Évaluez la qualité de ce dataset:
        1. Score de complétude (% données non manquantes)
        2. Cohérence des types de données
        3. Doublons potentiels
        4. Valeurs aberrantes critiques
        5. Recommandations de nettoyage prioritaires
        """
        return self._safe_execute(question)


class StatisticalAnalysisAgent(BaseAgent):
    """Agent spécialisé dans les analyses statistiques"""
    
    def __init__(self, dataframe: pd.DataFrame, openai_api_key: str):
        super().__init__(dataframe, openai_api_key)
        
        system_prompt = """
        Vous êtes un statisticien expert. Votre rôle est de:
        - Calculer des statistiques descriptives avancées
        - Identifier les corrélations et patterns
        - Effectuer des tests statistiques
        - Fournir des insights analytiques approfondis
        
        Utilisez des méthodes statistiques appropriées et expliquez vos résultats.
        """
        
        self.agent = self._create_pandas_agent(system_prompt)
    
    def descriptive_analysis(self) -> str:
        """Analyse statistique descriptive"""
        question = """
        Effectuez une analyse statistique descriptive complète:
        1. Moyennes, médianes, écarts-types pour toutes les variables numériques
        2. Quartiles et intervalles interquartiles
        3. Mesures d'asymétrie (skewness) et d'aplatissement (kurtosis)
        4. Distributions des variables catégorielles
        5. Identifiez les 3 insights statistiques les plus intéressants
        """
        return self._safe_execute(question)
    
    def correlation_analysis(self) -> str:
        """Analyse des corrélations"""
        question = """
        Analysez les corrélations dans les données:
        1. Matrice de corrélation pour variables numériques
        2. Identifiez les corrélations fortes (>0.7 ou <-0.7)
        3. Relations entre variables catégorielles et numériques
        4. Patterns et insights sur les relations entre variables
        """
        return self._safe_execute(question)


class VisualizationAgent(BaseAgent):
    """Agent spécialisé dans la génération de visualisations intelligentes"""
    
    def __init__(self, dataframe: pd.DataFrame, openai_api_key: str):
        super().__init__(dataframe, openai_api_key)
        
        system_prompt = """
        Vous êtes un expert en dataviz et storytelling avec les données. Votre rôle est de:
        - Recommander les meilleurs types de graphiques selon les données
        - Identifier les variables clés à visualiser
        - Suggérer des insights visuels pertinents
        - Proposer des combinaisons de variables intéressantes
        
        Pensez en termes d'impact visuel et de compréhension métier.
        """
        
        self.agent = self._create_pandas_agent(system_prompt)
    
    def recommend_visualizations(self) -> Dict[str, Any]:
        """Recommande des visualisations adaptées aux données"""
        question = """
        Analysez ces données et recommandez les meilleures visualisations:
        1. Pour chaque variable numérique: quel type de graphique (histogramme, boxplot, etc.)
        2. Pour les variables catégorielles: graphique en barres, camembert, etc.
        3. Pour les relations entre variables: scatter plots, heatmaps, etc.
        4. Si données temporelles: graphiques temporels recommandés
        5. Top 3 des visualisations les plus importantes à créer en priorité
        
        Donnez une réponse structurée avec le nom des colonnes et types de graphiques.
        """
        
        result = self._safe_execute(question)
        return {"recommendations": result, "visualizations": self._create_smart_visualizations()}
    
    def _create_smart_visualizations(self) -> List[Tuple[str, Any]]:
        """Crée des visualisations intelligentes basées sur l'analyse des données"""
        visualizations = []
        
        # Analyse des types de colonnes
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # 1. Dashboard de métriques clés
        if len(numeric_cols) > 0:
            fig_metrics = self._create_metrics_dashboard(numeric_cols)
            visualizations.append(("📊 Dashboard Métriques", fig_metrics))
        
        # 2. Analyse de distribution intelligente
        if len(numeric_cols) > 0:
            fig_dist = self._create_intelligent_distributions(numeric_cols)
            visualizations.append(("📈 Distributions Intelligentes", fig_dist))
        
        # 3. Analyse des relations
        if len(numeric_cols) > 1:
            fig_relations = self._create_relationship_analysis(numeric_cols)
            visualizations.append(("🔗 Analyse des Relations", fig_relations))
        
        # 4. Analyse catégorielle avancée
        if len(categorical_cols) > 0:
            fig_cat = self._create_categorical_analysis(categorical_cols, numeric_cols)
            visualizations.append(("🏷️ Analyse Catégorielle", fig_cat))
        
        # 5. Série temporelle intelligente
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            fig_time = self._create_time_series_analysis(datetime_cols, numeric_cols)
            visualizations.append(("⏱️ Analyse Temporelle", fig_time))
            
        return visualizations
    
    def _create_metrics_dashboard(self, numeric_cols: List[str]) -> go.Figure:
        """Crée un dashboard de métriques clés"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Moyennes par Variable', 'Écarts-types', 'Min/Max', 'Médiane vs Moyenne'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Moyennes
        means = [self.df[col].mean() for col in numeric_cols[:8]]  # Limite à 8 colonnes
        fig.add_trace(go.Bar(x=numeric_cols[:8], y=means, name="Moyennes"), row=1, col=1)
        
        # Écarts-types
        stds = [self.df[col].std() for col in numeric_cols[:8]]
        fig.add_trace(go.Bar(x=numeric_cols[:8], y=stds, name="Écarts-types"), row=1, col=2)
        
        # Min/Max
        mins = [self.df[col].min() for col in numeric_cols[:8]]
        maxs = [self.df[col].max() for col in numeric_cols[:8]]
        fig.add_trace(go.Bar(x=numeric_cols[:8], y=mins, name="Min"), row=2, col=1)
        fig.add_trace(go.Bar(x=numeric_cols[:8], y=maxs, name="Max"), row=2, col=1)
        
        # Médiane vs Moyenne
        if len(numeric_cols) > 0:
            medians = [self.df[col].median() for col in numeric_cols[:8]]
            fig.add_trace(go.Scatter(x=means[:len(medians)], y=medians, 
                                   mode='markers', name="Médiane vs Moyenne"), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=True, title_text="Dashboard des Métriques Clés")
        return fig
    
    def _create_intelligent_distributions(self, numeric_cols: List[str]) -> go.Figure:
        """Crée des distributions avec analyse intelligente"""
        n_cols = min(len(numeric_cols), 6)  # Limite à 6 variables
        fig = make_subplots(
            rows=(n_cols + 2) // 3, cols=3,
            subplot_titles=[f"{col} (Skew: {self.df[col].skew():.2f})" for col in numeric_cols[:n_cols]]
        )
        
        for i, col in enumerate(numeric_cols[:n_cols]):
            row = i // 3 + 1
            col_pos = i % 3 + 1
            
            # Distribution avec couleur basée sur la skewness
            skew = self.df[col].skew()
            color = 'red' if abs(skew) > 1 else 'orange' if abs(skew) > 0.5 else 'green'
            
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col, showlegend=False, 
                           marker_color=color, opacity=0.7),
                row=row, col=col_pos
            )
        
        fig.update_layout(height=400 * ((n_cols + 2) // 3), 
                         title_text="Distributions avec Analyse de Symétrie")
        return fig
    
    def _create_relationship_analysis(self, numeric_cols: List[str]) -> go.Figure:
        """Analyse intelligente des relations entre variables"""
        if len(numeric_cols) < 2:
            return go.Figure()
            
        # Calcul de la matrice de corrélation
        corr_matrix = self.df[numeric_cols].corr()
        
        # Trouver les paires les plus corrélées
        correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if not np.isnan(corr_val):
                    correlations.append((numeric_cols[i], numeric_cols[j], abs(corr_val), corr_val))
        
        correlations.sort(key=lambda x: x[2], reverse=True)
        
        # Créer des scatter plots pour les top 4 corrélations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"{pair[0]} vs {pair[1]} (r={pair[3]:.3f})" 
                          for pair in correlations[:4]]
        )
        
        for i, (col1, col2, abs_corr, corr) in enumerate(correlations[:4]):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            color = 'red' if corr > 0.7 else 'orange' if corr > 0.3 else 'blue'
            
            fig.add_trace(
                go.Scatter(x=self.df[col1], y=self.df[col2], 
                         mode='markers', name=f"{col1} vs {col2}",
                         marker=dict(color=color, opacity=0.6)),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Analyse des Relations les Plus Fortes")
        return fig
    
    def _create_categorical_analysis(self, categorical_cols: List[str], numeric_cols: List[str]) -> go.Figure:
        """Analyse avancée des variables catégorielles"""
        if not categorical_cols:
            return go.Figure()
            
        cat_col = categorical_cols[0]  # Prend la première variable catégorielle
        
        # Filtre les catégories avec trop peu d'occurrences
        value_counts = self.df[cat_col].value_counts()
        top_categories = value_counts.head(8).index.tolist()  # Top 8 catégories
        
        if numeric_cols:
            # Boxplot par catégorie pour la première variable numérique
            num_col = numeric_cols[0]
            df_filtered = self.df[self.df[cat_col].isin(top_categories)]
            
            fig = go.Figure()
            
            for category in top_categories:
                data = df_filtered[df_filtered[cat_col] == category][num_col]
                fig.add_trace(go.Box(y=data, name=str(category)))
            
            fig.update_layout(
                title=f"Distribution de {num_col} par {cat_col}",
                xaxis_title=cat_col,
                yaxis_title=num_col,
                height=500
            )
            
        else:
            # Simple bar chart si pas de variables numériques
            fig = px.bar(
                x=top_categories, 
                y=value_counts[top_categories].values,
                title=f"Distribution de {cat_col}",
                labels={'x': cat_col, 'y': 'Fréquence'}
            )
            fig.update_layout(height=500)
        
        return fig
    
    def _create_time_series_analysis(self, datetime_cols: List[str], numeric_cols: List[str]) -> go.Figure:
        """Analyse temporelle intelligente"""
        date_col = datetime_cols[0]
        num_col = numeric_cols[0]
        
        # Assurer que la colonne date est au bon format
        df_time = self.df.copy()
        df_time[date_col] = pd.to_datetime(df_time[date_col])
        df_time = df_time.sort_values(date_col)
        
        # Créer des agrégations temporelles intelligentes
        df_time['year_month'] = df_time[date_col].dt.to_period('M')
        monthly_agg = df_time.groupby('year_month')[num_col].agg(['mean', 'sum', 'count']).reset_index()
        monthly_agg['year_month'] = monthly_agg['year_month'].astype(str)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Évolution Temporelle', 'Moyenne Mensuelle', 
                          'Total Mensuel', 'Nombre d\'Observations'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Série temporelle originale
        fig.add_trace(go.Scatter(x=df_time[date_col], y=df_time[num_col], 
                               mode='lines', name='Série Originale'), row=1, col=1)
        
        # Moyennes mensuelles
        fig.add_trace(go.Scatter(x=monthly_agg['year_month'], y=monthly_agg['mean'], 
                               mode='lines+markers', name='Moyenne Mensuelle'), row=1, col=2)
        
        # Totaux mensuels
        fig.add_trace(go.Bar(x=monthly_agg['year_month'], y=monthly_agg['sum'], 
                           name='Total Mensuel'), row=2, col=1)
        
        # Nombre d'observations
        fig.add_trace(go.Bar(x=monthly_agg['year_month'], y=monthly_agg['count'], 
                           name='Nb Observations'), row=2, col=2)
        
        fig.update_layout(height=600, title_text=f"Analyse Temporelle de {num_col}")
        return fig


class DataTransformationAgent(BaseAgent):
    """Agent spécialisé dans la transformation et nettoyage des données"""
    
    def __init__(self, dataframe: pd.DataFrame, openai_api_key: str):
        super().__init__(dataframe, openai_api_key)
        
        system_prompt = """
        Vous êtes un expert en transformation et nettoyage de données. Votre rôle est de:
        - Identifier les transformations nécessaires
        - Proposer des méthodes de nettoyage
        - Suggérer des nouvelles variables dérivées
        - Optimiser la structure des données
        
        Proposez des solutions pratiques et justifiées.
        """
        
        self.agent = self._create_pandas_agent(system_prompt)
    
    def suggest_transformations(self) -> str:
        """Suggère des transformations pertinentes"""
        question = """
        Analysez ces données et suggérez des transformations:
        1. Variables à normaliser ou standardiser
        2. Variables catégorielles à encoder
        3. Valeurs manquantes: stratégies de traitement
        4. Variables dérivées intéressantes à créer
        5. Filtres ou nettoyages prioritaires
        """
        return self._safe_execute(question)


# ===============================
# ORCHESTRATEUR D'AGENTS
# ===============================

class AgentOrchestrator:
    """Orchestrateur qui dirige les requêtes vers les bons agents"""
    
    def __init__(self, dataframe: pd.DataFrame, openai_api_key: str):
        self.df = dataframe
        self.agents = {
            'profiling': DataProfilingAgent(dataframe, openai_api_key),
            'statistical': StatisticalAnalysisAgent(dataframe, openai_api_key),
            'visualization': VisualizationAgent(dataframe, openai_api_key),
            'transformation': DataTransformationAgent(dataframe, openai_api_key),
            'supervisor': SupervisorAnalystAgent(dataframe, openai_api_key)
        }
        self.intent_llm = OpenAI(api_key=openai_api_key, temperature=0.0, max_tokens=10)

    def classify_intent(self, question: str) -> str:
        """Classifie l'intention de la question pour router vers le bon agent, via LLM (LangChain)"""
        prompt = (
            "Tu es un classificateur d'intention pour un assistant d'analyse de données. "
            "Voici les intentions possibles : profiling, statistical, visualization, transformation, supervisor. "
            "Pour la question suivante, réponds uniquement par le mot-clé correspondant à l'intention la plus appropriée (rien d'autre) :\n"
            f"Question : {question}\nIntention : "
        )
        try:
            intent = self.intent_llm(prompt).strip().lower()
            if intent in self.agents:
                return intent
        except Exception:
            pass
        # Fallback: ancienne logique mots-clés
        question_lower = question.lower()
        viz_keywords = ['graphique', 'visualis', 'chart', 'plot', 'graph', 'affich', 'montre', 'dessine']
        if any(keyword in question_lower for keyword in viz_keywords):
            return 'visualization'
        stat_keywords = ['corrélation', 'moyenne', 'médiane', 'écart', 'distribution', 'test', 'significatif', 'variance', 'régression']
        if any(keyword in question_lower for keyword in stat_keywords):
            return 'statistical'
        profiling_keywords = ['qualité', 'profil', 'résumé', 'aperçu', 'manquant', 'aberrant', 'complet', 'overview']
        if any(keyword in question_lower for keyword in profiling_keywords):
            return 'profiling'
        transform_keywords = ['nettoyer', 'transformer', 'créer', 'modifier', 'encoder', 'normaliser']
        if any(keyword in question_lower for keyword in transform_keywords):
            return 'transformation'
        supervisor_keywords = ['synthèse', 'valider', 'recommandation globale', 'superviser', 'cohérence', 'conclusion', 'expert final']
        if any(keyword in question_lower for keyword in supervisor_keywords):
            return 'supervisor'
        return 'profiling'
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """Traite une question en la routant vers le bon agent"""
        intent = self.classify_intent(question)
        agent = self.agents[intent]
        try:
            if intent == 'visualization':
                result = agent.recommend_visualizations()
                return {
                    'agent': intent,
                    'response': result['recommendations'],
                    'visualizations': result['visualizations'],
                    'success': True
                }
            elif intent == 'supervisor':
                # Appelle tous les autres agents et synthétise les résultats
                analyses = {}
                for key in ['profiling', 'statistical', 'visualization', 'transformation']:
                    ag = self.agents[key]
                    if key == 'profiling':
                        analyses[key] = ag.profile_data()
                    elif key == 'statistical':
                        analyses[key] = ag.descriptive_analysis()
                    elif key == 'visualization':
                        analyses[key] = ag.recommend_visualizations()['recommendations']
                    elif key == 'transformation':
                        analyses[key] = ag.suggest_transformations()
                response = agent.synthesize_and_validate(analyses)
                return {
                    'agent': intent,
                    'response': response,
                    'visualizations': [],
                    'success': True
                }
            else:
                if intent == 'profiling' and 'résumé' in question.lower():
                    response = agent.profile_data()
                elif intent == 'statistical' and 'corrélation' in question.lower():
                    response = agent.correlation_analysis()
                elif intent == 'statistical':
                    response = agent.descriptive_analysis()
                elif intent == 'transformation':
                    response = agent.suggest_transformations()
                else:
                    # Utilise la méthode générique d'analyse
                    response = agent._safe_execute(question)
                return {
                    'agent': intent,
                    'response': response,
                    'visualizations': [],
                    'success': True
                }
        except Exception as e:
            return {
                'agent': intent,
                'response': f"⚠️ Erreur dans l'agent {intent}: {str(e)}",
                'visualizations': [],
                'success': False
            }


# ===============================
# FONCTIONS UTILITAIRES AMÉLIORÉES
# ===============================

def display_data_overview(df):
    """Affiche un aperçu des données avec métriques améliorées"""
    st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
    st.subheader("📋 Aperçu des Données")
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Lignes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">Colonnes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{missing_percentage:.1f}%</div>
            <div class="metric-label">Valeurs Manquantes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{memory_usage:.1f} MB</div>
            <div class="metric-label">Taille Mémoire</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Aperçu du dataframe avec types de données colorés
    st.subheader("📊 Échantillon des Données")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Analyse des types de données avec statistiques
    st.subheader("🔍 Analyse des Types de Données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Statistiques par type
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        st.markdown(f"""
        **📊 Répartition des Types:**
        - 🔢 Numériques: {len(numeric_cols)}
        - 🏷️ Catégorielles: {len(categorical_cols)}
        - 📅 Temporelles: {len(datetime_cols)}
        - 🔍 Autres: {len(df.columns) - len(numeric_cols) - len(categorical_cols) - len(datetime_cols)}
        """)
    
    with col2:
        # Tableau détaillé des types
        types_df = pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Nulles': df.count(),
            'Uniques': [df[col].nunique() for col in df.columns],
            'Complétude%': [round((df[col].count() / len(df)) * 100, 1) for col in df.columns]
        })
        st.dataframe(types_df, use_container_width=True, height=200)
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_agent_response(agent_type: str, response: str, visualizations: List = None, history_index: int = 0):
    """Affiche la réponse d'un agent avec style personnalisé"""
    agent_styles = {
        'profiling': 'agent-profiling',
        'statistical': 'agent-statistical', 
        'visualization': 'agent-visualization',
        'transformation': 'agent-transformation'
    }
    agent_names = {
        'profiling': '🔍 Agent Profiling',
        'statistical': '📊 Agent Statistique',
        'visualization': '📈 Agent Visualisation', 
        'transformation': '🔧 Agent Transformation'
    }
    style_class = agent_styles.get(agent_type, 'agent-profiling')
    agent_name = agent_names.get(agent_type, f'🤖 Agent {agent_type}')
    st.markdown(f"""
    <div class="chat-container">
        <div class="agent-message">
            <span class="agent-indicator {style_class}">{agent_name}</span><br>
            {response}
        </div>
    </div>
    """, unsafe_allow_html=True)
    # Afficher les visualisations si disponibles
    if visualizations:
        st.markdown("### 📊 Visualisations Générées")
        tabs = st.tabs([viz[0] for viz in visualizations])
        for i, (title, fig) in enumerate(visualizations):
            with tabs[i]:
                unique_key = f"{agent_type}_{title}_{i}_hist{history_index}"
                st.plotly_chart(fig, use_container_width=True, key=unique_key)


# ===============================
# APPLICATION PRINCIPALE
# ===============================

def main():
    # En-tête principal
    st.markdown("""
    <div class="main-title">📊 AnalyticsPro</div>
    <div class="subtitle">Système de Self-Service Analytics avec Agents Spécialisés</div>
    """, unsafe_allow_html=True)
    
    # Zone d'upload de fichier
    st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Chargement des Données")
    
    uploaded_file = st.file_uploader(
        "Glissez-déposez votre fichier Excel ici ou cliquez pour parcourir",
        type=["xlsx", "xls", "csv"],
        help="Formats supportés: Excel (.xlsx, .xls) et CSV"
    )
    
    if uploaded_file is not None:
        try:
            # Chargement du fichier
            with st.spinner("🔄 Chargement et analyse du fichier..."):
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès!")
            
            # Affichage de l'aperçu des données
            display_data_overview(st.session_state.df)
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement: {str(e)}")
            st.session_state.df = None
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Interface d'analyse si données chargées
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Vérification de la clé API
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("🔑 Clé API OpenAI non trouvée. Veuillez configurer votre variable d'environnement OPENAI_API_KEY.")
            return
        
        # Initialisation de l'orchestrateur
        try:
            orchestrator = AgentOrchestrator(df, openai_api_key)
        except Exception as e:
            st.error(f"❌ Erreur d'initialisation des agents: {str(e)}")
            return
        
        # Interface de chat améliorée
        st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
        st.markdown("### 💬 Assistant d'Analyse Multi-Agents")
        
        # Indicateur des agents disponibles
        st.markdown("""
        **🤖 Agents Disponibles:**
        - <span class="agent-indicator agent-profiling">🔍 Profiling</span> Qualité et aperçu des données
        - <span class="agent-indicator agent-statistical">📊 Statistique</span> Analyses statistiques avancées  
        - <span class="agent-indicator agent-visualization">📈 Visualisation</span> Graphiques intelligents
        - <span class="agent-indicator agent-transformation">🔧 Transformation</span> Nettoyage et modification
        """, unsafe_allow_html=True)
        
        # Zone de saisie de question
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_question = st.text_input(
                "Posez votre question (l'IA choisira automatiquement le bon agent):",
                placeholder="Ex: Montre-moi des graphiques, Analyse la corrélation, Résumé des données...",
                key="user_input"
            )
        with col2:
            analyze_button = st.button("🔍 Analyser", type="primary")

        # Gestion de l'analyse
        if 'last_question' not in st.session_state:
            st.session_state.last_question = ""
        if 'pending_analysis' not in st.session_state:
            st.session_state.pending_analysis = False

        # Si nouvelle question
        if user_question and user_question != st.session_state.last_question:
            st.session_state.pending_analysis = True
            st.session_state.last_question = user_question

        # Exécution de l'analyse
        run_analysis = (analyze_button or st.session_state.pending_analysis)
        
        # Boutons de questions suggérées par agent
        st.markdown("**💡 Questions Suggérées par Agent:**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**🔍 Profiling**")
            if st.button("📋 Profil complet", key="prof1"):
                st.session_state.last_question = "Donne-moi un profil complet de ces données"
                st.session_state.pending_analysis = True
                st.rerun()
            if st.button("🔍 Qualité des données", key="prof2"):
                st.session_state.last_question = "Évalue la qualité de ces données"
                st.session_state.pending_analysis = True
                st.rerun()
        
        with col2:
            st.markdown("**📊 Statistique**")
            if st.button("📈 Statistiques descriptives", key="stat1"):
                st.session_state.last_question = "Donne-moi les statistiques descriptives"
                st.session_state.pending_analysis = True
                st.rerun()
            if st.button("🔗 Analyse des corrélations", key="stat2"):
                st.session_state.last_question = "Analyse les corrélations entre variables"
                st.session_state.pending_analysis = True
                st.rerun()
        
        with col3:
            st.markdown("**📈 Visualisation**")
            if st.button("📊 Recommander graphiques", key="viz1"):
                st.session_state.last_question = "Quels graphiques recommandes-tu pour ces données?"
                st.session_state.pending_analysis = True
                st.rerun()
            if st.button("🎨 Créer visualisations", key="viz2"):
                st.session_state.last_question = "Crée des visualisations intelligentes"
                st.session_state.pending_analysis = True
                st.rerun()
        
        with col4:
            st.markdown("**🔧 Transformation**")
            if st.button("🛠️ Suggérer nettoyage", key="trans1"):
                st.session_state.last_question = "Suggère des transformations pour nettoyer les données"
                st.session_state.pending_analysis = True
                st.rerun()
            if st.button("🔄 Variables dérivées", key="trans2"):
                st.session_state.last_question = "Quelles nouvelles variables puis-je créer?"
                st.session_state.pending_analysis = True
                st.rerun()
        
        # Traitement de la question
        if run_analysis and st.session_state.last_question:
            with st.spinner("🤖 L'IA sélectionne l'agent approprié et analyse..."):
                # Animation de chargement
                st.markdown("""
                <div class="loading-animation">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Traitement par l'orchestrateur
                result = orchestrator.process_question(st.session_state.last_question)
                
                # Ajout à l'historique
                st.session_state.chat_history.append({
                    'question': st.session_state.last_question,
                    'agent': result['agent'],
                    'response': result['response'],
                    'visualizations': result.get('visualizations', []),
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'success': result['success']
                })
                st.session_state.pending_analysis = False
        
        # Affichage de l'historique des conversations
        if st.session_state.chat_history:
            st.markdown("### 💭 Historique des Analyses")
            
        for idx, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            # Message utilisateur
            st.markdown(f"""
            <div class="chat-container">
                <div class="user-message">
                    <strong>🙋‍♂️ Vous ({chat['timestamp']}):</strong><br>
                    {chat['question']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Réponse de l'agent
            display_agent_response(
                chat['agent'], 
                chat['response'], 
                chat.get('visualizations', []),
                history_index=idx
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Export des résultats amélioré
        st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
        st.markdown("### 💾 Export des Résultats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Télécharger Données"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 Télécharger CSV",
                    data=csv,
                    file_name=f"donnees_analysees_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Rapport d'Analyse"):
                report = f"""
# Rapport d'Analyse Multi-Agents - {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Résumé des Données
- Nombre de lignes: {len(df):,}
- Nombre de colonnes: {len(df.columns)}
- Taille: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB

## Analyses Effectuées par Agent
"""
                for chat in st.session_state.chat_history:
                    agent_name = {
                        'profiling': '🔍 Agent Profiling',
                        'statistical': '📊 Agent Statistique',
                        'visualization': '📈 Agent Visualisation',
                        'transformation': '🔧 Agent Transformation'
                    }.get(chat['agent'], chat['agent'])
                    
                    report += f"""
### {agent_name} - {chat['timestamp']}
**Question:** {chat['question']}

**Réponse:**
{chat['response']}

---
"""
                
                st.download_button(
                    label="📄 Télécharger Rapport",
                    data=report,
                    file_name=f"rapport_multi_agents_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    mime="text/markdown"
                )
        
        with col3:
            if st.button("🔄 Nouvelle Session"):
                st.session_state.df = None
                st.session_state.chat_history = []
                st.session_state.analysis_results = []
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Message d'accueil amélioré
        st.markdown("""
        <div class="analytics-card">
            <div style="text-align: center; padding: 2rem;">
                <h3>🚀 Analytics Multi-Agents</h3>
                <p style="font-size: 1.1rem; color: #7f8c8d; line-height: 1.6;">
                    Notre système utilise 4 agents spécialisés pour analyser vos données :
                </p>
                <div style="margin: 2rem 0;">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
                        <div style="background: #e8f4fd; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #1976d2;">
                            <strong>🔍 Agent Profiling</strong><br>
                            <small>Analyse la qualité, structure et complétude des données</small>
                        </div>
                        <div style="background: #f3e5f5; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #7b1fa2;">
                            <strong>📊 Agent Statistique</strong><br>
                            <small>Calculs statistiques, corrélations et tests</small>
                        </div>
                        <div style="background: #e8f5e8; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #388e3c;">
                            <strong>📈 Agent Visualisation</strong><br>
                            <small>Génère des graphiques intelligents et adaptatifs</small>
                        </div>
                        <div style="background: #fff3e0; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #f57c00;">
                            <strong>🔧 Agent Transformation</strong><br>
                            <small>Nettoyage et création de nouvelles variables</small>
                        </div>
                    </div>
                </div>
                <p style="margin-top: 2rem; font-style: italic;">
                    L'IA choisit automatiquement le bon agent selon votre question !
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: rgba(255, 255, 255, 0.7);">
    <hr style="border: 1px solid rgba(255, 255, 255, 0.2); margin: 2rem 0;">
    Made with ❤️ using Multi-Agent Architecture | © 2024 AnalyticsPro v2.0
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()