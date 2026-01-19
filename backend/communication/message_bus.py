"""Bus de communication pour le système multi-agent"""
from typing import Dict, Optional
from queue import Queue, Empty
import threading
from ..models.message import Message


class MessageBus:
    """Bus de messages central pour la communication inter-agent"""
    
    def __init__(self):
        self.agent_queues: Dict[str, Queue] = {}
        self.message_history = []
        self.lock = threading.Lock()
        self.stats = {"total_messages": 0}
    
    def register_agent(self, agent_name: str):
        """Enregistre un agent sur le bus"""
        with self.lock:
            if agent_name not in self.agent_queues:
                self.agent_queues[agent_name] = Queue()
                print(f"Agent '{agent_name}' enregistré")
    
    def send_message(self, message: Message) -> bool:
        """Envoie un message à un agent"""
        with self.lock:
            if message.receiver not in self.agent_queues:
                print(f"Agent destinataire '{message.receiver}' non trouvé")
                return False
            
            self.agent_queues[message.receiver].put(message)
            self.message_history.append(message)
            self.stats["total_messages"] += 1
            
            print(f"Message: {message.sender} -> {message.receiver} [{message.message_type.value}]")
            return True
    
    def receive_message(self, agent_name: str, timeout: float = 0.1) -> Optional[Message]:
        """Récupère le prochain message pour un agent"""
        if agent_name not in self.agent_queues:
            return None
        
        try:
            return self.agent_queues[agent_name].get(timeout=timeout)
        except Empty:
            return None
    
    def has_messages(self, agent_name: str) -> bool:
        """Vérifie si un agent a des messages"""
        if agent_name not in self.agent_queues:
            return False
        return not self.agent_queues[agent_name].empty()
    
    def get_stats(self) -> dict:
        """Retourne les statistiques"""
        return self.stats.copy()


# Instance singleton
_bus = None

def get_message_bus() -> MessageBus:
    """Retourne l'instance du bus"""
    global _bus
    if _bus is None:
        _bus = MessageBus()
    return _bus
