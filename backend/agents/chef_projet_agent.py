"""Agent Chef de Projet - Version debug"""
from typing import Dict, Any
from ..models.message import Message, MessageType
from .base_agent import BaseAgent


class ChefProjetAgent(BaseAgent):
    """Agent Chef de Projet - Orchestrateur"""
    
    def __init__(self):
        super().__init__(name="ChefProjet", role="Orchestrateur")
        self.current_dataset = None
        self.dataset_metadata = None
        self.dataset_profile = None
    
    def handle_message(self, message: Message):
        """Traite les messages reçus"""
        
        if message.message_type == MessageType.DATA_UPLOAD:
            self._handle_data_upload(message)
        
        elif message.message_type == MessageType.DATA_VALIDATION:
            self._handle_data_validation(message)
        
        elif message.message_type == MessageType.USER_MESSAGE:
            self._handle_user_request(message)
        
        elif message.message_type == MessageType.TASK_RESPONSE:
            self._forward_response(message)
        
        elif message.message_type == MessageType.ERROR:
            self._handle_error(message)
    
    def _handle_data_upload(self, message: Message):
        """Redirige l'upload vers l'Ingénieur Données"""
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="DataEngineer",
            message_type=MessageType.DATA_UPLOAD,
            content=message.content
        ))
    
    def _handle_data_validation(self, message: Message):
        """Reçoit la validation de l'Ingénieur et stocke le dataset"""
        if message.content.get('valid'):
            # Important: Stocker dans session_state pour persister entre reruns
            import streamlit as st
            if hasattr(st, 'session_state'):
                st.session_state.shared_dataset = message.content.get('dataset')
                st.session_state.shared_metadata = message.content.get('metadata')
                st.session_state.shared_profile = message.content.get('profile')
            
            self.current_dataset = message.content.get('dataset')
            self.dataset_metadata = message.content.get('metadata')
            self.dataset_profile = message.content.get('profile')
            
            # Transférer au Frontend
            self.message_bus.send_message(Message(
                sender=self.name,
                receiver="Frontend",
                message_type=MessageType.DATA_VALIDATION,
                content=message.content
            ))
    
    def _handle_user_request(self, message: Message):
        """Traite une requête utilisateur"""
        user_message = message.content.get('message', '').lower()
        
        # Récupérer le dataset depuis le message (envoyé par Frontend)
        dataset = message.content.get('dataset')
        
        if 'analyser' in user_message or 'analyse' in user_message:
            self._request_analysis(dataset)
        elif 'statistique' in user_message or 'stats' in user_message:
            self._request_statistics(dataset)
        elif 'resume' in user_message or 'résumé' in user_message:
            self._request_summary(dataset)
        elif 'entrainer' in user_message or 'modele' in user_message or 'predi' in user_message:
            target = message.content.get('target')
            problem_type = message.content.get('problem_type', 'auto')
            self._request_training(dataset, target, problem_type)
        else:
            self._send_to_frontend("Commande non reconnue. Utilisez l'interface graphique.")
    
    def _request_analysis(self, dataset=None):
        """Demande une analyse complète"""
        if dataset is None:
            self._send_error_to_frontend("Aucune donnée chargée")
            return
        
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Analyste",
            message_type=MessageType.TASK_REQUEST,
            content={'task': 'analyse_complete', 'dataset': dataset}
        ))
    
    def _request_statistics(self, dataset=None):
        """Demande des statistiques"""
        if dataset is None:
            self._send_error_to_frontend("Aucune donnée chargée")
            return
        
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Analyste",
            message_type=MessageType.TASK_REQUEST,
            content={'task': 'statistiques', 'dataset': dataset}
        ))
    
    def _request_summary(self, dataset=None):
        """Demande un résumé"""
        if dataset is None:
            self._send_error_to_frontend("Aucune donnée chargée")
            return
        
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Analyste",
            message_type=MessageType.TASK_REQUEST,
            content={'task': 'resume', 'dataset': dataset}
        ))
    
    def _request_training(self, dataset=None, target=None, problem_type='auto'):
        """Demande un entraînement ML"""
        if dataset is None:
            self._send_error_to_frontend("Aucune donnée chargée")
            return
        
        if not target:
            self._send_error_to_frontend("Variable cible non spécifiée")
            return
        
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="ModelisateurML",
            message_type=MessageType.TASK_REQUEST,
            content={'task': 'entrainer', 'dataset': dataset, 'target': target, 'problem_type': problem_type}
        ))
    
    def _forward_response(self, message: Message):
        """Transfère une réponse au Frontend"""
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Frontend",
            message_type=MessageType.AGENT_RESPONSE,
            content=message.content
        ))
    
    def _handle_error(self, message: Message):
        """Gère une erreur"""
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Frontend",
            message_type=MessageType.ERROR,
            content=message.content
        ))
    
    def _send_to_frontend(self, text: str):
        """Envoie un message au Frontend"""
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Frontend",
            message_type=MessageType.AGENT_RESPONSE,
            content={'message': text}
        ))
    
    def _send_error_to_frontend(self, error: str):
        """Envoie une erreur au Frontend"""
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Frontend",
            message_type=MessageType.ERROR,
            content={'error': error}
        ))
