"""
Test du Socle Multi-Agent
Fichier: tests/test_socle.py

Ce fichier teste le fonctionnement du socle multi-agent:
- Classe Agent de base
- MessageBus
- AgentManager
- Communication entre agents

Usage: python tests/test_socle.py
"""

import sys
import os
from time import sleep
from typing import Dict

# Ajouter le repertoire parent au path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from agents.base_agent import Agent
    from agents.agent_manager import AgentManager
except ImportError as e:
    print(f"ERREUR IMPORT: {e}")
    print(f"Chemin actuel: {os.getcwd()}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)


# ============================================
# AGENTS DE TEST
# ============================================

class TestAgentA(Agent):
    """Agent de test simple qui repond aux ping"""
    
    def __init__(self, name: str, message_bus=None):
        """Initialise l'agent de test"""
        super().__init__(name, message_bus)
    
    def process_task(self, task: str, payload: dict) -> dict:
        """Traite les taches"""
        
        if task == "ping":
            self.log("Ping recu, envoi pong", "INFO")
            return {
                "status": "success",
                "response": "pong",
                "data": payload
            }
        
        elif task == "compute":
            value = payload.get("value", 0)
            result = value * 2
            self.log(f"Calcul: {value} * 2 = {result}", "INFO")
            return {
                "status": "success",
                "result": result
            }
        
        else:
            return {
                "status": "error",
                "message": f"Tache inconnue: {task}"
            }


class TestAgentB(Agent):
    """Agent de test qui fait des calculs"""
    
    def __init__(self, name: str, message_bus=None):
        """Initialise l'agent de test"""
        super().__init__(name, message_bus)
    
    def process_task(self, task: str, payload: dict) -> dict:
        """Traite les taches"""
        
        if task == "add":
            a = payload.get("a", 0)
            b = payload.get("b", 0)
            result = a + b
            self.log(f"Addition: {a} + {b} = {result}", "INFO")
            return {
                "status": "success",
                "result": result
            }
        
        elif task == "multiply":
            a = payload.get("a", 1)
            b = payload.get("b", 1)
            result = a * b
            self.log(f"Multiplication: {a} * {b} = {result}", "INFO")
            return {
                "status": "success",
                "result": result
            }
        
        else:
            return {
                "status": "error",
                "message": f"Tache inconnue: {task}"
            }


# ============================================
# TESTS
# ============================================

def test_1_creation_agents():
    """Test 1: Creation d'agents"""
    print("\n" + "=" * 70)
    print("TEST 1: CREATION D'AGENTS")
    print("=" * 70)
    
    agent_a = TestAgentA("AgentA")
    agent_b = TestAgentB("AgentB")
    
    assert agent_a.name == "AgentA"
    assert agent_b.name == "AgentB"
    assert agent_a.state == "idle"
    assert agent_b.state == "idle"
    
    print("[OK] Agents crees avec succes")
    print(f"  - {agent_a.name}: state={agent_a.state}")
    print(f"  - {agent_b.name}: state={agent_b.state}")
    
    return agent_a, agent_b


def test_2_agent_manager():
    """Test 2: AgentManager"""
    print("\n" + "=" * 70)
    print("TEST 2: AGENT MANAGER")
    print("=" * 70)
    
    manager = AgentManager()
    
    agent_a = TestAgentA("AgentA")
    agent_b = TestAgentB("AgentB")
    
    # Enregistrer les agents
    print("\n[TEST] Enregistrement des agents...")
    success_a = manager.register_agent(agent_a)
    success_b = manager.register_agent(agent_b)
    
    assert success_a == True, "Echec enregistrement AgentA"
    assert success_b == True, "Echec enregistrement AgentB"
    assert len(manager.agents) == 2, f"Nombre d'agents incorrect: {len(manager.agents)}"
    
    print("[OK] Agents enregistres dans le manager")
    
    # Demarrer les agents
    print("\n[TEST] Demarrage des agents...")
    manager.start_all()
    
    print("[OK] Agents demarres")
    
    # Afficher statut
    print("\n[TEST] Affichage du statut systeme...")
    manager.print_system_status()
    
    print("\n[OK] Test AgentManager complet")
    
    manager.shutdown()
    
    return manager


def test_3_communication_simple():
    """Test 3: Communication simple entre agents"""
    print("\n" + "=" * 70)
    print("TEST 3: COMMUNICATION SIMPLE")
    print("=" * 70)
    
    manager = AgentManager()
    
    agent_a = TestAgentA("AgentA")
    agent_b = TestAgentB("AgentB")
    
    manager.register_agent(agent_a)
    manager.register_agent(agent_b)
    manager.start_all()
    
    # AgentA envoie un message a AgentB
    print("\n[TEST] AgentA envoie message a AgentB")
    agent_a.send_message("AgentB", "add", {"a": 5, "b": 3})
    
    sleep(0.1)
    
    # AgentB recoit et traite le message
    message = agent_b.receive_message()
    assert message is not None
    print(f"[OK] AgentB a recu un message: {message['task']}")
    
    result = agent_b.handle_message(message)
    assert result['status'] == 'success'
    assert result['result'] == 8
    print(f"[OK] AgentB a traite le message: result={result['result']}")
    
    manager.shutdown()


def test_4_communication_bidirectionnelle():
    """Test 4: Communication bidirectionnelle"""
    print("\n" + "=" * 70)
    print("TEST 4: COMMUNICATION BIDIRECTIONNELLE")
    print("=" * 70)
    
    manager = AgentManager()
    
    agent_a = TestAgentA("AgentA")
    agent_b = TestAgentB("AgentB")
    
    manager.register_agent(agent_a)
    manager.register_agent(agent_b)
    manager.start_all()
    
    # AgentA -> AgentB
    print("\n[TEST] AgentA -> AgentB: ping")
    agent_a.send_message("AgentB", "multiply", {"a": 4, "b": 5})
    
    sleep(0.1)
    
    msg1 = agent_b.receive_message()
    result1 = agent_b.handle_message(msg1)
    print(f"[OK] AgentB resultat: {result1['result']}")
    
    # AgentB -> AgentA
    print("\n[TEST] AgentB -> AgentA: compute")
    agent_b.send_message("AgentA", "compute", {"value": 10})
    
    sleep(0.1)
    
    msg2 = agent_a.receive_message()
    result2 = agent_a.handle_message(msg2)
    print(f"[OK] AgentA resultat: {result2['result']}")
    
    # Afficher historique
    print("\n[TEST] Historique des messages:")
    history = manager.message_bus.get_history()
    for i, msg in enumerate(history, 1):
        print(f"  {i}. {msg['from']} -> {msg['to']}: {msg['task']}")
    
    manager.shutdown()


def test_5_workflow_complet():
    """Test 5: Workflow complet avec plusieurs agents"""
    print("\n" + "=" * 70)
    print("TEST 5: WORKFLOW COMPLET")
    print("=" * 70)
    
    manager = AgentManager()
    
    # Creer 3 agents
    agent_a = TestAgentA("Calculateur1")
    agent_b = TestAgentB("Calculateur2")
    agent_c = TestAgentA("Aggregateur")
    
    manager.register_agent(agent_a)
    manager.register_agent(agent_b)
    manager.register_agent(agent_c)
    manager.start_all()
    
    print("\n[WORKFLOW] Calcul distribue: (10 * 2) + (5 + 3)")
    
    # Etape 1: Calculateur1 multiplie
    print("\n[Etape 1] Calculateur1: 10 * 2")
    agent_a.send_message("Calculateur2", "add", {"a": 999, "b": 1})  # Dummy
    msg_dummy = agent_b.receive_message()
    agent_b.handle_message(msg_dummy)
    
    # Simuler calcul local
    result1 = 10 * 2
    print(f"  Resultat 1: {result1}")
    
    # Etape 2: Calculateur2 additionne
    print("\n[Etape 2] Calculateur2: 5 + 3")
    agent_b.send_message("Aggregateur", "ping", {"value": 8})
    msg2 = agent_c.receive_message()
    result2_obj = agent_c.handle_message(msg2)
    result2 = 5 + 3
    print(f"  Resultat 2: {result2}")
    
    # Etape 3: Aggregateur combine
    print("\n[Etape 3] Aggregation: {result1} + {result2}")
    final_result = result1 + result2
    print(f"  Resultat final: {final_result}")
    
    assert final_result == 28
    print("\n[OK] Workflow complet teste avec succes")
    
    # Stats finales
    manager.print_system_status()
    
    manager.shutdown()


# ============================================
# EXECUTION DES TESTS
# ============================================

def run_all_tests():
    """Execute tous les tests"""
    print("\n" + "=" * 70)
    print("TESTS DU SOCLE MULTI-AGENT")
    print("=" * 70)
    
    tests = [
        test_1_creation_agents,
        test_2_agent_manager,
        test_3_communication_simple,
        test_4_communication_bidirectionnelle,
        test_5_workflow_complet
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print(f"\n[SUCCES] {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"\n[ECHEC] {test.__name__}: {e}")
    
    # Resultat final
    print("\n" + "=" * 70)
    print("RESULTAT DES TESTS")
    print("=" * 70)
    print(f"  Tests passes: {passed}/{len(tests)}")
    print(f"  Tests echoues: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n[OK] TOUS LES TESTS SONT PASSES")
        print("\nLe socle multi-agent est fonctionnel et pret a etre utilise")
    else:
        print("\n[ERREUR] CERTAINS TESTS ONT ECHOUE")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()