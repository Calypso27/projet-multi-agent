"""Agent Ingénieur Données - Version améliorée avec support multi-format"""
import pandas as pd
import io
from typing import Dict, Any, Optional
from ..models.message import Message, MessageType
from .base_agent import BaseAgent
from ..utils.file_detector import FileDetector
from ..utils.data_profiler import DataProfiler


class DataEngineerAgent(BaseAgent):
    """
    Agent Ingénieur Données
    - Upload multi-format (CSV, Excel, JSON, Parquet, etc.)
    - Validation automatique
    - Profiling intelligent
    """
    
    def __init__(self):
        super().__init__(name="DataEngineer", role="Ingénieur Données")
        self.current_dataset: Optional[pd.DataFrame] = None
        self.dataset_metadata: Dict[str, Any] = {}
        self.dataset_profile: Dict[str, Any] = {}
    
    def handle_message(self, message: Message):
        """Traite les messages"""
        
        if message.message_type == MessageType.DATA_UPLOAD:
            self._handle_data_upload(message)
        
        elif message.message_type == MessageType.TASK_REQUEST:
            task = message.content.get("task")
            if task == "hello":
                self._handle_hello(message)
            elif task == "validate":
                self._handle_validation(message)
    
    def _handle_data_upload(self, message: Message):
        """Traite l'upload avec support multi-format"""
        file_data = message.content.get('file_data')
        filename = message.content.get('filename')
        
        if not file_data or not filename:
            self._send_error(message.sender, "Données de fichier manquantes")
            return
        
        try:
            # Utiliser FileDetector pour charger automatiquement
            df, error = FileDetector.load_file(file_data, filename)
            
            if error:
                self._send_error(message.sender, error)
                return
            
            # Stocker le dataset
            self.current_dataset = df
            
            # Générer le profil automatique
            self.dataset_profile = DataProfiler.profile(df)
            
            # Métadonnées de base
            self.dataset_metadata = {
                'filename': filename,
                'format': FileDetector.detect_format(filename),
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'data_type': DataProfiler.get_data_type_description(df)
            }
            
            # Préparer le message de succès
            format_name = FileDetector.SUPPORTED_FORMATS.get(
                self.dataset_metadata['format'], 
                'Fichier'
            )
            
            success_message = (
                f"{format_name} chargé avec succès\n"
                f"Dimensions: {len(df)} lignes × {len(df.columns)} colonnes\n"
                f"Type: {self.dataset_metadata['data_type']}"
            )
            
            # Ajouter les avertissements si nécessaire
            warnings = []
            if self.dataset_profile['missing_values']['total'] > 0:
                warnings.append(
                    f"Attention: {self.dataset_profile['missing_values']['total']} valeurs manquantes"
                )
            if self.dataset_profile['duplicates'] > 0:
                warnings.append(
                    f"Attention: {self.dataset_profile['duplicates']} lignes dupliquées"
                )
            
            if warnings:
                success_message += "\n\n" + "\n".join(warnings)
            
            # Envoyer la réponse avec dataset, metadata et profile
            self.message_bus.send_message(Message(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.DATA_VALIDATION,
                content={
                    'valid': True,
                    'message': success_message,
                    'dataset': df,
                    'metadata': self.dataset_metadata,
                    'profile': self.dataset_profile
                }
            ))
            
        except Exception as e:
            self._send_error(message.sender, f"Erreur lors du chargement: {str(e)}")
    
    def _handle_validation(self, message: Message):
        """Valide la qualité des données actuelles"""
        if self.current_dataset is None:
            self._send_error(message.sender, "Aucun dataset chargé")
            return
        
        # Rapport de validation détaillé
        validation_report = {
            'valid': True,
            'metadata': self.dataset_metadata,
            'profile': self.dataset_profile,
            'quality_score': self._calculate_quality_score()
        }
        
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver=message.sender,
            message_type=MessageType.TASK_RESPONSE,
            content=validation_report
        ))
    
    def _calculate_quality_score(self) -> float:
        """Calcule un score de qualité des données (0-100)"""
        if not self.dataset_profile:
            return 0.0
        
        score = 100.0
        
        # Pénalité pour valeurs manquantes
        missing_pct = self.dataset_profile['missing_values']['percentage']
        score -= min(missing_pct, 30)
        
        # Pénalité pour duplicatas
        if self.current_dataset is not None:
            dup_pct = (self.dataset_profile['duplicates'] / len(self.current_dataset)) * 100
            score -= min(dup_pct * 2, 20)
        
        return max(0.0, score)
    
    def _handle_hello(self, message: Message):
        """Répond au message de test"""
        response = (
            "Agent Ingénieur Données opérationnel\n"
            f"Formats supportés: {FileDetector.get_supported_formats_string()}\n"
            "Prêt à recevoir vos fichiers"
        )
        
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver=message.sender,
            message_type=MessageType.TASK_RESPONSE,
            content={'message': response}
        ))
    
    def _send_error(self, recipient: str, error_message: str):
        """Envoie un message d'erreur"""
        self.message_bus.send_message(Message(
            sender=self.name,
            receiver=recipient,
            message_type=MessageType.ERROR,
            content={'error': error_message}
        ))
