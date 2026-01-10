from typing import Dict, List, Optional
from datetime import datetime

from agents.base_agent import Agent
from communication.message_bus import MessageBus


class AgentManager:
    
    def __init__(self):
        #Initialise le gestionnaire d'agents
        self.agents: Dict[str, Agent] = {}
        self.message_bus = MessageBus()
        self.start_time = datetime.now()
        
        print("[AgentManager] Gestionnaire initialise")
    
    def register_agent(self, agent: Agent) -> bool:
        #Enregistre un agent
        if agent is None:
            print("[AgentManager] ERREUR: Agent est None")
            return False
            
        agent_name = agent.name
        
        if agent_name in self.agents:
            print(f"[AgentManager] ERREUR: Agent deja enregistre: {agent_name}")
            return False
        
        # Enregistrer dans le MessageBus d'abord
        self.message_bus.register_agent(agent_name)
        
        # Configurer le MessageBus pour l'agent
        agent.message_bus = self.message_bus
        
        # Enregistrer l'agent
        self.agents[agent_name] = agent
        
        print(f"[AgentManager] Agent enregistre: {agent_name}")
        return True
    
    def unregister_agent(self, agent_name: str) -> bool:
        
        if agent_name not in self.agents:
            print(f"[AgentManager] ERREUR: Agent inconnu: {agent_name}")
            return False
        
        # Arreter l'agent
        self.agents[agent_name].stop()
        
        # Desenregistrer du MessageBus
        self.message_bus.unregister_agent(agent_name)
        
        # Retirer de la liste
        del self.agents[agent_name]
        
        print(f"[AgentManager] Agent desenregistre: {agent_name}")
        return True
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        #Retourne une instance d'agent par nom
        return self.agents.get(agent_name)
    
    def start_all(self):
        #Demarre tous les agents
        print("\n[AgentManager] Demarrage de tous les agents...")
        
        for agent_name, agent in self.agents.items():
            try:
                agent.start()
                print(f"[AgentManager] {agent_name}: OK")
            except Exception as e:
                print(f"[AgentManager] {agent_name}: ERREUR - {e}")
        
        print(f"[AgentManager] {len(self.agents)} agents demarres\n")
    
    def stop_all(self):
        #Arrete tous les agents
        print("\n[AgentManager] Arret de tous les agents...")
        
        for agent_name, agent in self.agents.items():
            try:
                agent.stop()
                print(f"[AgentManager] {agent_name}: Arrete")
            except Exception as e:
                print(f"[AgentManager] {agent_name}: ERREUR - {e}")
        
        print(f"[AgentManager] Tous les agents arretes\n")
    
    def get_all_status(self) -> Dict[str, Dict]:
        #Retourne le statut de tous les agents
        status = {}
        for agent_name, agent in self.agents.items():
            try:
                status[agent_name] = agent.get_status()
            except Exception as e:
                status[agent_name] = {
                    "error": str(e),
                    "state": "unknown"
                }
        return status
    
    def get_system_info(self) -> Dict:
        #Retourne des informations sur le systeme
        uptime = datetime.now() - self.start_time
        
        return {
            "start_time": self.start_time.isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "registered_agents": len(self.agents),
            "agent_names": list(self.agents.keys()),
            "message_bus_stats": self.message_bus.get_stats()
        }
    
    def print_system_status(self):
        #Affiche le statut du systeme multi-agent
        print("\n" + "=" * 70)
        print("STATUT DU SYSTEME MULTI-AGENT")
        print("=" * 70)
        
        # Info systeme
        info = self.get_system_info()
        print(f"\nSysteme demarre: {info['start_time']}")
        print(f"Agents enregistres: {info['registered_agents']}")
        print(f"Agents: {', '.join(info['agent_names'])}")
        
        # Statut agents
        print("\nStatut des agents:")
        print("-" * 70)
        status = self.get_all_status()
        for agent_name, agent_status in status.items():
            state = agent_status.get('state', 'unknown')
            print(f"  {agent_name:30} [{state}]")
        
        # Stats MessageBus
        print("\nMessageBus:")
        print("-" * 70)
        bus_stats = info['message_bus_stats']
        print(f"  Messages totaux: {bus_stats['total_messages_sent']}")
        print(f"  Messages en attente:")
        for agent, count in bus_stats['pending_messages'].items():
            if count > 0:
                print(f"    {agent}: {count}")
        
        print("=" * 70 + "\n")
    
    def send_broadcast(self, task: str, payload: Dict, exclude: Optional[List[str]] = None):
        #Envoie un message broadcast a tous les agents, optionnellement en excluant certains
        exclude = exclude or []
        
        for agent_name in self.agents.keys():
            if agent_name not in exclude:
                message = {
                    "message_id": f"broadcast_{int(datetime.now().timestamp())}",
                    "timestamp": datetime.now().isoformat(),
                    "from": "AgentManager",
                    "to": agent_name,
                    "task": task,
                    "payload": payload
                }
                self.message_bus.send_message(agent_name, message)
        
        print(f"[AgentManager] Broadcast envoye: {task}")
    
    def shutdown(self):
            #Arrete proprement le systeme multi-agent
        print("\n[AgentManager] Arret du systeme...")
        
        # Arreter tous les agents
        self.stop_all()
        
        # Reset le MessageBus
        self.message_bus.reset()
        
        print("[AgentManager] Systeme arrete proprement\n")