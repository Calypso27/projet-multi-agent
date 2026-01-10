"""
Message Bus - Systeme de Communication Inter-Agents
Fichier: communication/message_bus.py

Ce fichier gere la communication entre tous les agents du systeme.
Il agit comme un intermediaire (mediateur) pour router les messages.
"""

from queue import Queue, Empty
from typing import Dict, Any, Optional, List
from datetime import datetime
import json


class MessageBus:
    """
    Bus de messages centralise pour la communication inter-agents.
    
    Responsabilites:
    - Enregistrer les agents
    - Router les messages entre agents
    - Maintenir l'historique des messages
    - Gerer les files d'attente par agent
    """
    
    def __init__(self):
        """Initialise le MessageBus"""
        self.queues: Dict[str, Queue] = {}
        self.history: List[Dict] = []
        self.registered_agents: List[str] = []
        
    def register_agent(self, agent_name: str):
        """
        Enregistre un agent dans le bus
        
        Args:
            agent_name: Nom unique de l'agent
        """
        if agent_name not in self.queues:
            self.queues[agent_name] = Queue()
            self.registered_agents.append(agent_name)
            print(f"[MessageBus] Agent enregistre: {agent_name}")
        else:
            print(f"[MessageBus] Agent deja enregistre: {agent_name}")
    
    def unregister_agent(self, agent_name: str):
        """
        Desenregistre un agent
        
        Args:
            agent_name: Nom de l'agent a desenregistrer
        """
        if agent_name in self.queues:
            del self.queues[agent_name]
            self.registered_agents.remove(agent_name)
            print(f"[MessageBus] Agent desenregistre: {agent_name}")
    
    def send_message(self, to_agent: str, message: Dict[str, Any]) -> bool:
        """
        Envoie un message a un agent
        
        Args:
            to_agent: Nom de l'agent destinataire
            message: Message a envoyer
            
        Returns:
            True si succes, False sinon
        """
        if to_agent not in self.queues:
            print(f"[MessageBus] ERREUR: Agent inconnu: {to_agent}")
            return False
        
        # Ajouter metadata
        message["bus_timestamp"] = datetime.now().isoformat()
        
        # Envoyer dans la queue
        self.queues[to_agent].put(message)
        
        # Sauvegarder dans l'historique
        self.history.append(message.copy())
        
        print(f"[MessageBus] Message route: {message['from']} -> {to_agent}")
        return True
    
    def get_message(self, agent_name: str, timeout: Optional[int] = None) -> Optional[Dict]:
        """
        Recupere un message pour un agent
        
        Args:
            agent_name: Nom de l'agent
            timeout: Temps d'attente max en secondes (None = non bloquant)
            
        Returns:
            Message ou None si aucun message
        """
        if agent_name not in self.queues:
            print(f"[MessageBus] ERREUR: Agent inconnu: {agent_name}")
            return None
        
        try:
            if timeout is None:
                # Non bloquant
                return self.queues[agent_name].get_nowait()
            else:
                # Bloquant avec timeout
                return self.queues[agent_name].get(timeout=timeout)
        except Empty:
            return None
    
    def has_messages(self, agent_name: str) -> bool:
        """
        Verifie si un agent a des messages en attente
        
        Args:
            agent_name: Nom de l'agent
            
        Returns:
            True si des messages sont en attente
        """
        if agent_name not in self.queues:
            return False
        return not self.queues[agent_name].empty()
    
    def get_queue_size(self, agent_name: str) -> int:
        """
        Retourne le nombre de messages en attente pour un agent
        
        Args:
            agent_name: Nom de l'agent
            
        Returns:
            Nombre de messages
        """
        if agent_name not in self.queues:
            return 0
        return self.queues[agent_name].qsize()
    
    def get_history(self, limit: Optional[int] = None, 
                   from_agent: Optional[str] = None,
                   to_agent: Optional[str] = None) -> List[Dict]:
        """
        Recupere l'historique des messages
        
        Args:
            limit: Nombre max de messages a retourner
            from_agent: Filtrer par expediteur
            to_agent: Filtrer par destinataire
            
        Returns:
            Liste des messages
        """
        filtered = self.history
        
        if from_agent:
            filtered = [m for m in filtered if m.get("from") == from_agent]
        
        if to_agent:
            filtered = [m for m in filtered if m.get("to") == to_agent]
        
        if limit:
            filtered = filtered[-limit:]
        
        return filtered
    
    def clear_history(self):
        """Efface l'historique des messages"""
        self.history.clear()
        print("[MessageBus] Historique efface")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur le bus
        
        Returns:
            Dictionnaire de statistiques
        """
        stats = {
            "registered_agents": len(self.registered_agents),
            "agents": self.registered_agents.copy(),
            "total_messages_sent": len(self.history),
            "pending_messages": {}
        }
        
        for agent_name in self.registered_agents:
            stats["pending_messages"][agent_name] = self.get_queue_size(agent_name)
        
        return stats
    
    def export_history(self, filepath: str):
        """
        Export l'historique des messages en JSON
        
        Args:
            filepath: Chemin du fichier de sortie
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
            print(f"[MessageBus] Historique exporte: {filepath}")
        except Exception as e:
            print(f"[MessageBus] Erreur export: {e}")
    
    def reset(self):
        """Reset complet du bus"""
        for agent_name in list(self.queues.keys()):
            self.unregister_agent(agent_name)
        self.clear_history()
        print("[MessageBus] Bus reinitialise")