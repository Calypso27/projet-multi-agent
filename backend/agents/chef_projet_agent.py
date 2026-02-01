"""Agent Chef de Projet"""
from typing import Dict, Any
from ..models.message import Message, MessageType
from .base_agent import BaseAgent


class ChefProjetAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="ChefProjet", role="Orchestrateur")
        self.current_dataset = None
        self.dataset_metadata = None
        self.dataset_profile = None

    def handle_message(self, message: Message):
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
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="DataEngineer",
            message_type=MessageType.DATA_UPLOAD,
            content=message.content
        ))

    def _handle_data_validation(self, message: Message):
        if message.content.get('valid'):
            self.current_dataset = message.content.get('dataset')
            self.dataset_metadata = message.content.get('metadata')
            self.dataset_profile = message.content.get('profile')

            self.message_bus.send_message(Message(
                sender=self.name,
                receiver="Frontend",
                message_type=MessageType.DATA_VALIDATION,
                content=message.content
            ))

    def _handle_user_request(self, message: Message):
        user_message = message.content.get('message', '').lower()
        dataset = message.content.get('dataset')

        if 'eda_complet' in user_message or 'eda complet' in user_message:
            self._request_eda_complet(dataset)
        elif 'analyser' in user_message or 'analyse' in user_message:
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

    def _request_eda_complet(self, dataset=None):
        if dataset is None:
            self._send_error_to_frontend("Aucune donnée chargée")
            return

        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Analyste",
            message_type=MessageType.TASK_REQUEST,
            content={'task': 'eda_complet', 'dataset': dataset}
        ))

    def _request_analysis(self, dataset=None):
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
        task = message.content.get('task', '')
        result = message.content.get('result', '')
        heatmap = message.content.get('heatmap', None)
        visualizations = message.content.get('visualizations', None)

        content = {'message': result, 'task': task}

        if heatmap:
            content['heatmap'] = heatmap

        if visualizations:
            content['visualizations'] = visualizations

        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Frontend",
            message_type=MessageType.AGENT_RESPONSE,
            content=content
        ))

    def _handle_error(self, message: Message):
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Frontend",
            message_type=MessageType.ERROR,
            content=message.content
        ))

    def _send_to_frontend(self, text: str):
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Frontend",
            message_type=MessageType.AGENT_RESPONSE,
            content={'message': text}
        ))

    def _send_error_to_frontend(self, error: str):
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver="Frontend",
            message_type=MessageType.ERROR,
            content={'error': error}
        ))
