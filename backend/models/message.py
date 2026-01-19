"""Modèles de messages pour la communication inter-agent"""
from typing import Any, Optional, Dict
from datetime import datetime
from enum import Enum


class MessageType(Enum):
    """Types de messages échangés"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    DATA_UPLOAD = "data_upload"
    DATA_VALIDATION = "data_validation"
    ERROR = "error"
    USER_MESSAGE = "user_message"
    AGENT_RESPONSE = "agent_response"


class Message:
    """Classe représentant un message entre agents"""
    
    def __init__(self, sender: str, receiver: str, message_type: MessageType, 
                 content: Any, conversation_id: Optional[str] = None):
        self.id = f"msg_{datetime.now().timestamp()}"
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.content = content
        self.conversation_id = conversation_id
        self.timestamp = datetime.now()
    
    def __repr__(self):
        return f"Message(from={self.sender}, to={self.receiver}, type={self.message_type.value})"
