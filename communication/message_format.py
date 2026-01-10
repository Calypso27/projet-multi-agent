from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

@dataclass
class Message:
    #Format standard de message entre agents
    message_id: str
    timestamp: datetime
    from_agent: str
    to_agent: str
    task: str
    payload: Dict[str, Any]
    priority: str = "normal"