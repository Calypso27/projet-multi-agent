
# la classe de base pour tous les agents du systeme.


from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
import logging


class Agent(ABC):

    #Classe abstraite de base pour tous les agents.
    #
    
    def __init__(self, name: str, message_bus=None):
        
        #Initialise l'agent
        
        self.name = name
        self.message_bus = message_bus
        self.state = "idle"  # idle, processing, error
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Configure le logger pour cet agent"""
        logger = logging.getLogger(f"Agent.{self.name}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def send_message(self, to_agent: str, task: str, payload: Dict[str, Any], 
                    priority: str = "normal") -> Optional[str]:
        
        #Envoie un message a un autre agent via le MessageBus
        
    
        if not self.message_bus:
            self.log("Aucun MessageBus configure", "ERROR")
            return None
        
        message = {
            "message_id": self._generate_message_id(),
            "timestamp": datetime.now().isoformat(),
            "from": self.name,
            "to": to_agent,
            "task": task,
            "payload": payload,
            "priority": priority
        }
        
        self.log(f"Envoi message vers [{to_agent}] - Tache: {task}", "INFO")
        self.message_bus.send_message(to_agent, message)
        
        return message["message_id"]
    
    def receive_message(self, timeout: Optional[int] = None) -> Optional[Dict]:
        #Recoit un message de la file d'attente
       
        if not self.message_bus:
            return None
        
        return self.message_bus.get_message(self.name, timeout)
    
    @abstractmethod
    def process_task(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        
        #Traite une tache specifique - A implementer par chaque agent
        
        pass
    
    def handle_message(self, message: Dict) -> Dict[str, Any]:
        
        #Point d'entree principal pour traiter un message recu
        
        self.log(f"Traitement message - Tache: {message.get('task')}", "INFO")
        self.state = "processing"
        
        try:
            result = self.process_task(
                message.get("task"),
                message.get("payload", {})
            )
            self.state = "idle"
            return result
        except Exception as e:
            self.state = "error"
            self.log(f"Erreur traitement: {str(e)}", "ERROR")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def log(self, message: str, level: str = "INFO"):
        
        #Log un message avec le logger de l'agent
        
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(message)
    
    def _generate_message_id(self) -> str:
        """Genere un ID unique pour un message"""
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"msg_{self.name}_{timestamp}"
    
    def get_status(self) -> Dict[str, Any]:
        
        #Retourne le statut actuel de l'agent
        
        return {
            "name": self.name,
            "state": self.state,
            "timestamp": datetime.now().isoformat()
        }
    
    def start(self):
        """Demarre l'agent"""
        self.state = "idle"
        self.log(f"Agent demarre", "INFO")
    
    def stop(self):
        """Arrete l'agent"""
        self.state = "stopped"
        self.log(f"Agent arrete", "INFO")