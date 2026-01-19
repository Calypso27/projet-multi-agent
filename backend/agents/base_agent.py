"""Classe de base pour tous les agents"""
from abc import ABC, abstractmethod
from typing import Optional, Any
import threading
import time
from ..models.message import Message, MessageType
from ..communication.message_bus import get_message_bus


class BaseAgent(ABC):
    """Classe abstraite de base pour tous les agents"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.message_bus = get_message_bus()
        self.is_running = False
        self.current_conversation_id = None
        
        self.message_bus.register_agent(self.name)
        self.listener_thread: Optional[threading.Thread] = None
        
        print(f"Agent créé: {self.name} ({self.role})")
    
    def start(self):
        """Démarre l'agent"""
        if not self.is_running:
            self.is_running = True
            self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listener_thread.start()
            print(f"Agent {self.name} démarré")
    
    def stop(self):
        """Arrête l'agent"""
        self.is_running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=2)
        print(f"Agent {self.name} arrêté")
    
    def _listen_loop(self):
        """Boucle d'écoute des messages"""
        while self.is_running:
            message = self.message_bus.receive_message(self.name, timeout=0.1)
            if message:
                try:
                    self.handle_message(message)
                except Exception as e:
                    print(f"Erreur dans {self.name}: {e}")
            time.sleep(0.05)
    
    @abstractmethod
    def handle_message(self, message: Message):
        """Traite un message reçu (à implémenter par chaque agent)"""
        pass
    
    def send_message(self, receiver: str, message_type: MessageType, 
                    content: Any, conversation_id: Optional[str] = None) -> bool:
        """Envoie un message"""
        message = Message(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            content=content,
            conversation_id=conversation_id or self.current_conversation_id
        )
        return self.message_bus.send_message(message)
