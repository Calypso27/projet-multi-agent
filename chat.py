import json
from queue import Queue
from datetime import datetime
from typing import Dict, Any
import time


# CLASSE DE BASE AGENT


class Agent:
    """Classe de base pour tous les agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.inbox = Queue()
        self.is_running = False
        
    def send_message(self, to_agent, task: str, payload: Dict[Any, Any]):
        """Envoie un message a un autre agent"""
        message = {
            "message_id": f"msg_{int(time.time() * 1000)}",
            "timestamp": datetime.now().isoformat(),
            "from": self.name,
            "to": to_agent.name,
            "task": task,
            "payload": payload
        }
        print(f"\n[ENVOI] {self.name} -> {to_agent.name}")
        print(f"   Tache: {task}")
        print(f"   Donnees: {json.dumps(payload, ensure_ascii=False)[:100]}...")
        to_agent.inbox.put(message)
        return message
    
    def receive_message(self):
        """Recoit un message de la file d'attente"""
        if not self.inbox.empty():
            return self.inbox.get()
        return None
    
    def process_message(self, message: Dict):
        """Traite un message - a surcharger"""
        print(f"\n[RECEPTION] {self.name} Traitement message...")
        return {"status": "processed"}
    
    def start(self):
        """Demarre l'agent"""
        self.is_running = True
        print(f"[OK] Agent {self.name} demarre")



# AGENT CHEF DE PROJET


class ChefProjetAgent(Agent):
    """Agent orchestrateur qui coordonne les autres"""
    
    def __init__(self):
        super().__init__("Chef de Projet")
        self.agents = {}
    
    def register_agent(self, agent_name: str, agent: Agent):
        """Enregistre un agent"""
        self.agents[agent_name] = agent
        print(f"[REGISTER] {self.name} enregistrement: {agent_name}")
    
    def process_user_request(self, request: str):
        """Traite une requete utilisateur"""
        print("\n" + "="*70)
        print(f"[UTILISATEUR] {request}")
        print("="*70)
        print(f"\n[ANALYSE] {self.name} analyse de la requete...")
        time.sleep(0.3)
        
        # Analyse simple de la requete
        request_lower = request.lower()
        
        if "analyser" in request_lower or "analyse" in request_lower:
            print(f"[DETECTION] {self.name} Demande d'analyse de donnees")
            print(f"[PLAN] {self.name} Plan d'action:")
            print(f"   1. Charger les donnees (Ingenieur)")
            print(f"   2. Faire l'analyse exploratoire (Analyste)")
            
            # Etape 1: Charger donnees
            if "Ingenieur Donnees" in self.agents:
                self.send_message(
                    self.agents["Ingenieur Donnees"],
                    "load_data",
                    {"file": "ventes.csv", "encoding": "utf-8"}
                )
            
            return "Analyse en cours..."
        
        elif "modele" in request_lower or "predire" in request_lower:
            print(f"[DETECTION] {self.name} Demande de modelisation")
            print(f"[PLAN] {self.name} Plan d'action:")
            print(f"   1. Charger les donnees (Ingenieur)")
            print(f"   2. Preparer les donnees (Analyste)")
            print(f"   3. Entrainer le modele (Modelisateur)")
            
            if "Ingenieur Donnees" in self.agents:
                self.send_message(
                    self.agents["Ingenieur Donnees"],
                    "load_data",
                    {"file": "data.csv", "target": "price"}
                )
            
            return "Modelisation en cours..."
        
        else:
            print(f"[INCONNU] {self.name} Requete non reconnue, analyse generique")
            return "Je n'ai pas compris votre demande."



# AGENT INGENIEUR DONNEES


class IngenieurDonneesAgent(Agent):
    """Agent responsable du chargement et validation des donnees"""
    
    def __init__(self):
        super().__init__("Ingenieur Donnees")
    
    def process_message(self, message: Dict):
        """Traite les messages"""
        super().process_message(message)
        
        task = message["task"]
        payload = message["payload"]
        
        if task == "load_data":
            print(f"\n[CHARGEMENT] {self.name} fichier: {payload.get('file', 'fichier')}")
            time.sleep(0.5)
            
            # Simulation validation
            print(f"   [OK] Lecture du fichier")
            print(f"   [OK] Detection encodage: {payload.get('encoding', 'utf-8')}")
            print(f"   [OK] Validation format")
            print(f"   [OK] Detection types de colonnes")
            
            result = {
                "status": "success",
                "data_ref": "df_001",
                "rows": 1000,
                "columns": 15,
                "file": payload.get('file'),
                "column_types": {
                    "date": "datetime",
                    "produit": "categorical",
                    "quantite": "numeric",
                    "prix": "numeric"
                }
            }
            
            print(f"\n[SUCCESS] {self.name} Donnees chargees:")
            print(f"   - {result['rows']} lignes")
            print(f"   - {result['columns']} colonnes")
            print(f"   - Reference: {result['data_ref']}")
            
            return result
        
        return {"status": "unknown_task"}



# AGENT ANALYSTE


class AnalysteAgent(Agent):
    """Agent d'analyse exploratoire"""
    
    def __init__(self):
        super().__init__("Analyste Exploratoire")
    
    def process_message(self, message: Dict):
        """Traite les messages"""
        super().process_message(message)
        
        task = message["task"]
        payload = message["payload"]
        
        if task == "analyze_data":
            data_ref = payload.get("data_ref", "unknown")
            print(f"\n[ANALYSE] {self.name} analyse exploratoire sur: {data_ref}")
            time.sleep(0.7)
            
            print(f"   [PROCESSING] Calcul statistiques descriptives...")
            print(f"   [PROCESSING] Detection valeurs manquantes...")
            print(f"   [PROCESSING] Analyse correlations...")
            print(f"   [PROCESSING] Generation visualisations...")
            
            result = {
                "status": "success",
                "statistics": {
                    "mean_price": 45.50,
                    "median_quantity": 25,
                    "total_sales": 45000
                },
                "quality": {
                    "missing_values": 12,
                    "outliers": 5,
                    "duplicates": 0
                },
                "correlations": {
                    "quantity-price": 0.65,
                    "date-sales": -0.12
                },
                "visualizations": [
                    "plots/histogram_prix.png",
                    "plots/correlation_matrix.png",
                    "plots/time_series.png"
                ]
            }
            
            print(f"\n[SUCCESS] {self.name} Analyse terminee:")
            print(f"   - Statistiques calculees")
            print(f"   - {result['quality']['missing_values']} valeurs manquantes")
            print(f"   - {len(result['visualizations'])} graphiques generes")
            print(f"   - Correlation principale: quantity-price (r=0.65)")
            
            return result
        
        return {"status": "unknown_task"}



# AGENT MODELISATEUR


class ModelisateurAgent(Agent):
    """Agent de modelisation ML"""
    
    def __init__(self):
        super().__init__("Modelisateur ML")
    
    def process_message(self, message: Dict):
        """Traite les messages"""
        super().process_message(message)
        
        task = message["task"]
        payload = message["payload"]
        
        if task == "train_model":
            data_ref = payload.get("data_ref", "unknown")
            target = payload.get("target", "target")
            
            print(f"\n[TRAINING] {self.name} Entrainement modele")
            print(f"   - Donnees: {data_ref}")
            print(f"   - Cible: {target}")
            time.sleep(0.8)
            
            print(f"   [PROCESSING] Preparation des donnees...")
            print(f"   [PROCESSING] Split train/test (80/20)...")
            print(f"   [PROCESSING] Entrainement Random Forest...")
            print(f"   [PROCESSING] Evaluation performance...")
            
            result = {
                "status": "success",
                "model": "RandomForestRegressor",
                "metrics": {
                    "r2_score": 0.87,
                    "mae": 5.23,
                    "rmse": 7.45
                },
                "feature_importance": {
                    "quantite": 0.45,
                    "date": 0.28,
                    "produit": 0.27
                },
                "model_path": "models/rf_model_001.pkl"
            }
            
            print(f"\n[SUCCESS] {self.name} Modele entraine:")
            print(f"   - Type: {result['model']}")
            print(f"   - R2 Score: {result['metrics']['r2_score']}")
            print(f"   - MAE: {result['metrics']['mae']}")
            print(f"   - Sauvegarde: {result['model_path']}")
            
            return result
        
        return {"status": "unknown_task"}



# ORCHESTRATION SYSTEME


def demo_scenario_1():
    """Scenario 1: Analyse exploratoire simple"""
    print("\n" + "=" * 70)
    print("SCENARIO 1: ANALYSE EXPLORATOIRE DE DONNEES")
    print("=" * 70)
    
    # Creer les agents
    chef = ChefProjetAgent()
    ingenieur = IngenieurDonneesAgent()
    analyste = AnalysteAgent()
    
    # Demarrer
    chef.start()
    ingenieur.start()
    analyste.start()
    
    # Enregistrer
    chef.register_agent("Ingenieur Donnees", ingenieur)
    chef.register_agent("Analyste Exploratoire", analyste)
    
    # Requete utilisateur
    chef.process_user_request("Analyser mes donnees de ventes")
    
    # Traitement du workflow
    time.sleep(0.5)
    
    # Ingenieur traite
    msg1 = ingenieur.receive_message()
    if msg1:
        result1 = ingenieur.process_message(msg1)
        
        # Envoyer au analyste
        ingenieur.send_message(
            analyste,
            "analyze_data",
            {"data_ref": result1.get("data_ref")}
        )
    
    # Analyste traite
    time.sleep(0.5)
    msg2 = analyste.receive_message()
    if msg2:
        result2 = analyste.process_message(msg2)
    
    print("\n" + "=" * 70)
    print("SCENARIO 1 TERMINE")
    print("=" * 70)


def demo_scenario_2():
    """Scenario 2: Creation de modele predictif"""
    print("\n\n" + "=" * 70)
    print("SCENARIO 2: MODELISATION PREDICTIVE")
    print("=" * 70)
    
    # Creer les agents
    chef = ChefProjetAgent()
    ingenieur = IngenieurDonneesAgent()
    analyste = AnalysteAgent()
    modelisateur = ModelisateurAgent()
    
    # Demarrer
    chef.start()
    ingenieur.start()
    analyste.start()
    modelisateur.start()
    
    # Enregistrer
    chef.register_agent("Ingenieur Donnees", ingenieur)
    chef.register_agent("Analyste Exploratoire", analyste)
    chef.register_agent("Modelisateur ML", modelisateur)
    
    # Requete
    chef.process_user_request("Creer un modele pour predire les prix")
    
    # Workflow
    time.sleep(0.5)
    
    # Ingenieur
    msg1 = ingenieur.receive_message()
    if msg1:
        result1 = ingenieur.process_message(msg1)
        
        # Vers Modelisateur
        ingenieur.send_message(
            modelisateur,
            "train_model",
            {
                "data_ref": result1.get("data_ref"),
                "target": "price"
            }
        )
    
    # Modelisateur
    time.sleep(0.5)
    msg2 = modelisateur.receive_message()
    if msg2:
        result2 = modelisateur.process_message(msg2)
    
    print("\n" + "=" * 70)
    print("SCENARIO 2 TERMINE")
    print("=" * 70)



# POINT D'ENTREE


if __name__ == "__main__":
    print("=" * 90)
    print(" " * 20 + "HELLO WORLD - SYSTEME MULTI-AGENT")
    print(" " * 15 + "Plateforme Conversationnelle d'Analyse de Donnees")
    print("=" * 90)
    
    # Lancer les scenarios
    demo_scenario_1()
    
    input("\n\n[PAUSE] Appuyez sur Entree pour le Scenario 2...")
    
    demo_scenario_2()
    
    print("\n\n" + "=" * 70)
    print("DEMO TERMINEE AVEC SUCCES")
    print("=" * 70)
    print("\nPROCHAINES ETAPES:")
    print("   1. Integrer vraies donnees avec Pandas")
    print("   2. Ajouter OpenAI API pour Chef de Projet")
    print("   3. Creer interface Streamlit")
    print("   4. Implementer persistance des resultats")
    print("\n[OK] Le systeme multi-agent fonctionne")
    print("=" * 90)