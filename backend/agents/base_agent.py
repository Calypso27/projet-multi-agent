"""Agent de base"""
from abc import ABC, abstractmethod
from typing import Optional, Any
import threading
import time
from ..models.message import Message, MessageType
from ..communication.message_bus import get_message_bus

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class BaseAgent(ABC):
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
        if not self.is_running:
            self.is_running = True
            self.listener_thread = threading.Thread(target=self._listen_loop, daemon=True)

            if STREAMLIT_AVAILABLE:
                try:
                    add_script_run_ctx(self.listener_thread)
                except Exception:
                    pass

            self.listener_thread.start()
            print(f"Agent {self.name} démarré")

    def stop(self):
        self.is_running = False
        if self.listener_thread:
            self.listener_thread.join(timeout=2)
        print(f"Agent {self.name} arrêté")

    def _listen_loop(self):
        while self.is_running:
            message = self.message_bus.receive_message(self.name, timeout=0.1)
            if message:
                try:
                    self.handle_message(message)
                except Exception as e:
                    error_str = str(e)
                    if "ScriptRunContext" not in error_str:
                        print(f"Erreur dans {self.name}: {e}")
            time.sleep(0.05)

    @abstractmethod
    def handle_message(self, message: Message):
        pass

    def send_message(self, receiver: str, message_type: MessageType,
                    content: Any, conversation_id: Optional[str] = None) -> bool:
        message = Message(
            sender=self.name,
            receiver=receiver,
            message_type=message_type,
            content=content,
            conversation_id=conversation_id or self.current_conversation_id
        )
        return self.message_bus.send_message(message)
