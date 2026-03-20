"""Agent Chef de Projet"""
from typing import Dict, Any
from ..models.message import Message, MessageType
from .base_agent import BaseAgent
from ..utils.llm_client import call_llm, is_available


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

        intent = self._classify_intent(user_message)

        if intent == 'eda':
            self._request_eda_complet(dataset)
        elif intent == 'analyse':
            self._request_analysis(dataset)
        elif intent == 'statistiques':
            self._request_statistics(dataset)
        elif intent == 'resume':
            self._request_summary(dataset)
        elif intent == 'entrainer':
            target = message.content.get('target')
            problem_type = message.content.get('problem_type', 'auto')
            self._request_training(dataset, target, problem_type)
        else:
            self._send_to_frontend(
                "Je n'ai pas compris votre demande. Vous pouvez demander :\n"
                "- une **analyse EDA** complète\n"
                "- des **statistiques** descriptives\n"
                "- un **résumé** du dataset\n"
                "- **entraîner** un modèle ML"
            )

    def _classify_intent(self, user_message: str) -> str:
        """Classifie l'intention de l'utilisateur.
        Priorité : règles keywords → fallback LLM."""
        msg = user_message.lower()

        # Règles keywords (rapides, sans LLM)
        if any(k in msg for k in ['eda_complet', 'eda complet', 'exploration complète',
                                   'analyse exploratoire', 'full eda', 'lancer eda']):
            return 'eda'
        if any(k in msg for k in ['analyser', 'analyse', 'analyze', 'explorer', 'explore']):
            return 'analyse'
        if any(k in msg for k in ['statistique', 'stats', 'statistic', 'describe', 'décrire']):
            return 'statistiques'
        if any(k in msg for k in ['resume', 'résumé', 'summary', 'aperçu', 'overview']):
            return 'resume'
        if any(k in msg for k in ['entrainer', 'entraîner', 'train', 'modele', 'modèle',
                                   'model', 'predi', 'classif', 'régress', 'regressio']):
            return 'entrainer'

        # Fallback LLM si disponible
        if is_available():
            try:
                prompt = (
                    f"Classe l'intention de cet utilisateur en UNE seule catégorie parmi : "
                    f"eda, analyse, statistiques, resume, entrainer, autre.\n"
                    f"Message : \"{user_message}\"\n"
                    f"Réponds uniquement avec le mot de la catégorie, sans explication."
                )
                result = call_llm(prompt=prompt).strip().lower()
                if result in ('eda', 'analyse', 'statistiques', 'resume', 'entrainer'):
                    return result
            except Exception:
                pass

        return 'autre'

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
