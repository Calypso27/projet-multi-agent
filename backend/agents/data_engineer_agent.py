"""Agent Ingénieur Données - Version améliorée avec support multi-format et preprocessing"""
import pandas as pd
import io
from typing import Dict, Any, Optional
from ..models.message import Message, MessageType
from .base_agent import BaseAgent
from ..utils.file_detector import FileDetector
from ..utils.data_profiler import DataProfiler
from ..utils.data_preprocessor import DataPreprocessor
from ..utils.data_quality_scorer import DataQualityScorer


class DataEngineerAgent(BaseAgent):
    """
    Agent Ingénieur Données
    - Upload multi-format (CSV, Excel, JSON, Parquet, etc.)
    - Validation automatique
    - Profiling intelligent
    - Preprocessing (nettoyage, encodage, normalisation, train/test split)
    """

    def __init__(self):
        super().__init__(name="DataEngineer", role="Ingénieur Données")
        self.current_dataset: Optional[pd.DataFrame] = None
        self.dataset_metadata: Dict[str, Any] = {}
        self.dataset_profile: Dict[str, Any] = {}
        self.preprocessed_dataset: Optional[pd.DataFrame] = None
        self.preprocessing_report: Dict[str, Any] = {}
    
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
            elif task == "preprocess":
                self._handle_preprocessing(message)
            elif task == "clean":
                self._handle_cleaning(message)
            elif task == "encode":
                self._handle_encoding(message)
            elif task == "normalize":
                self._handle_normalization(message)
            elif task == "split":
                self._handle_split(message)
    
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

            # NOUVEAU: Évaluer la qualité des données
            quality_report = DataQualityScorer.evaluate(df)
            self.dataset_profile['quality_report'] = quality_report

            # Métadonnées de base
            self.dataset_metadata = {
                'filename': filename,
                'format': FileDetector.detect_format(filename),
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': df.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'data_type': DataProfiler.get_data_type_description(df),
                'quality_score': quality_report['overall_score'],
                'quality_grade': quality_report['grade']
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

    # ==================== PREPROCESSING METHODS ====================

    def _handle_preprocessing(self, message: Message):
        """Exécute le pipeline complet de preprocessing"""
        if self.current_dataset is None:
            self._send_error(message.sender, "Aucun dataset chargé")
            return

        try:
            # Récupérer les paramètres
            target_col = message.content.get('target_col', None)
            clean = message.content.get('clean', True)
            encode = message.content.get('encode', True)
            normalize = message.content.get('normalize', True)
            split = message.content.get('split', True)
            test_size = message.content.get('test_size', 0.2)

            # Exécuter le pipeline complet
            result = DataPreprocessor.preprocess_pipeline(
                self.current_dataset,
                target_col=target_col,
                clean=clean,
                encode=encode,
                normalize=normalize,
                split=split,
                test_size=test_size
            )

            # Stocker les résultats
            self.preprocessed_dataset = result['data_train']
            self.preprocessing_report = result['rapport']

            # Construire le message de réponse
            response_message = self._build_preprocessing_summary(result)

            # Envoyer la réponse
            self.message_bus.send_message(Message(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    'task': 'preprocess',
                    'success': True,
                    'message': response_message,
                    'data_train': result['data_train'],
                    'data_test': result['data_test'],
                    'rapport': result['rapport']
                }
            ))

        except Exception as e:
            self._send_error(message.sender, f"Erreur lors du preprocessing: {str(e)}")

    def _handle_cleaning(self, message: Message):
        """Exécute uniquement le nettoyage des données"""
        if self.current_dataset is None:
            self._send_error(message.sender, "Aucun dataset chargé")
            return

        try:
            remove_duplicates = message.content.get('remove_duplicates', True)
            handle_missing = message.content.get('handle_missing', 'auto')
            remove_outliers = message.content.get('remove_outliers', False)

            df_clean, rapport = DataPreprocessor.clean_data(
                self.current_dataset,
                remove_duplicates=remove_duplicates,
                handle_missing=handle_missing,
                remove_outliers=remove_outliers
            )

            self.preprocessed_dataset = df_clean

            # Construire le message de résumé
            summary = self._build_cleaning_summary(rapport)

            self.message_bus.send_message(Message(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    'task': 'clean',
                    'success': True,
                    'message': summary,
                    'dataset': df_clean,
                    'rapport': rapport
                }
            ))

        except Exception as e:
            self._send_error(message.sender, f"Erreur lors du nettoyage: {str(e)}")

    def _handle_encoding(self, message: Message):
        """Exécute uniquement l'encodage des variables catégorielles"""
        if self.current_dataset is None:
            self._send_error(message.sender, "Aucun dataset chargé")
            return

        try:
            max_categories = message.content.get('max_categories', 10)
            drop_first = message.content.get('drop_first', False)

            df_encoded, rapport = DataPreprocessor.encode_categorical(
                self.current_dataset,
                max_categories=max_categories,
                drop_first=drop_first
            )

            self.preprocessed_dataset = df_encoded

            # Construire le message de résumé
            summary = self._build_encoding_summary(rapport)

            self.message_bus.send_message(Message(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    'task': 'encode',
                    'success': True,
                    'message': summary,
                    'dataset': df_encoded,
                    'rapport': rapport
                }
            ))

        except Exception as e:
            self._send_error(message.sender, f"Erreur lors de l'encodage: {str(e)}")

    def _handle_normalization(self, message: Message):
        """Exécute uniquement la normalisation"""
        if self.current_dataset is None:
            self._send_error(message.sender, "Aucun dataset chargé")
            return

        try:
            exclude_cols = message.content.get('exclude_cols', [])

            df_normalized, rapport = DataPreprocessor.normalize_features(
                self.current_dataset,
                exclude_cols=exclude_cols
            )

            self.preprocessed_dataset = df_normalized

            # Construire le message de résumé
            summary = self._build_normalization_summary(rapport)

            self.message_bus.send_message(Message(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    'task': 'normalize',
                    'success': True,
                    'message': summary,
                    'dataset': df_normalized,
                    'rapport': rapport
                }
            ))

        except Exception as e:
            self._send_error(message.sender, f"Erreur lors de la normalisation: {str(e)}")

    def _handle_split(self, message: Message):
        """Exécute uniquement le train/test split"""
        if self.current_dataset is None:
            self._send_error(message.sender, "Aucun dataset chargé")
            return

        try:
            target_col = message.content.get('target_col', None)
            test_size = message.content.get('test_size', 0.2)
            random_state = message.content.get('random_state', 42)
            stratify = message.content.get('stratify', True)

            df_train, df_test, rapport = DataPreprocessor.train_test_split_data(
                self.current_dataset,
                target_col=target_col,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify
            )

            # Construire le message de résumé
            summary = self._build_split_summary(rapport)

            self.message_bus.send_message(Message(
                sender=self.name,
                receiver=message.sender,
                message_type=MessageType.TASK_RESPONSE,
                content={
                    'task': 'split',
                    'success': True,
                    'message': summary,
                    'data_train': df_train,
                    'data_test': df_test,
                    'rapport': rapport
                }
            ))

        except Exception as e:
            self._send_error(message.sender, f"Erreur lors du split: {str(e)}")

    # ==================== SUMMARY BUILDERS ====================

    def _build_preprocessing_summary(self, result: Dict[str, Any]) -> str:
        """Construit un résumé lisible du preprocessing complet"""
        rapport = result['rapport']
        lines = ["=== PREPROCESSING COMPLET ===\n"]

        # Étapes exécutées
        steps = rapport.get('steps_executed', [])
        lines.append(f"Étapes exécutées: {', '.join(steps)}\n")

        # Nettoyage
        if 'cleaning' in rapport and rapport['cleaning']:
            cleaning = rapport['cleaning']
            lines.append("\n[NETTOYAGE]")
            lines.append(f"- Lignes avant: {cleaning.get('rows_before', 0):,}")
            lines.append(f"- Lignes après: {cleaning.get('rows_after', 0):,}")
            if cleaning.get('duplicates_removed', 0) > 0:
                lines.append(f"- Doublons supprimés: {cleaning['duplicates_removed']}")
            if cleaning.get('missing_handled'):
                lines.append(f"- Valeurs manquantes traitées dans {len(cleaning['missing_handled'])} colonnes")
            if cleaning.get('outliers_removed', 0) > 0:
                lines.append(f"- Outliers supprimés: {cleaning['outliers_removed']}")

        # Encodage
        if 'encoding' in rapport and rapport['encoding']:
            encoding = rapport['encoding']
            lines.append("\n[ENCODAGE]")
            for col, method in encoding.items():
                lines.append(f"- {col}: {method}")

        # Normalisation
        if 'normalization' in rapport and rapport['normalization']:
            norm = rapport['normalization']
            cols = norm.get('columns_normalized', [])
            if cols:
                lines.append(f"\n[NORMALISATION]")
                lines.append(f"- {len(cols)} colonnes normalisées ({norm.get('scaler_used', 'StandardScaler')})")

        # Split
        if 'split' in rapport and rapport['split']:
            split_info = rapport['split']
            lines.append(f"\n[TRAIN/TEST SPLIT]")
            lines.append(f"- Train: {split_info.get('train_size', 0):,} lignes")
            lines.append(f"- Test: {split_info.get('test_size', 0):,} lignes")
            if split_info.get('stratified'):
                lines.append(f"- Stratification: activée sur '{split_info.get('target_column')}'")

        return "\n".join(lines)

    def _build_cleaning_summary(self, rapport: Dict[str, Any]) -> str:
        """Construit un résumé du nettoyage"""
        lines = ["=== NETTOYAGE DES DONNÉES ===\n"]
        lines.append(f"Lignes avant: {rapport.get('rows_before', 0):,}")
        lines.append(f"Lignes après: {rapport.get('rows_after', 0):,}")

        if rapport.get('duplicates_removed', 0) > 0:
            lines.append(f"\nDoublons supprimés: {rapport['duplicates_removed']}")

        if rapport.get('missing_handled'):
            lines.append(f"\nValeurs manquantes traitées:")
            for col, action in rapport['missing_handled'].items():
                lines.append(f"  - {col}: {action}")

        if rapport.get('outliers_removed', 0) > 0:
            lines.append(f"\nOutliers supprimés: {rapport['outliers_removed']}")

        return "\n".join(lines)

    def _build_encoding_summary(self, rapport: Dict[str, str]) -> str:
        """Construit un résumé de l'encodage"""
        lines = ["=== ENCODAGE DES VARIABLES CATÉGORIELLES ===\n"]

        if not rapport:
            lines.append("Aucune variable catégorielle à encoder")
        else:
            lines.append(f"{len(rapport)} colonnes encodées:\n")
            for col, method in rapport.items():
                lines.append(f"  - {col}: {method}")

        return "\n".join(lines)

    def _build_normalization_summary(self, rapport: Dict[str, Any]) -> str:
        """Construit un résumé de la normalisation"""
        lines = ["=== NORMALISATION DES FEATURES ===\n"]

        cols = rapport.get('columns_normalized', [])
        if not cols:
            lines.append("Aucune colonne à normaliser")
        else:
            lines.append(f"Scaler utilisé: {rapport.get('scaler_used', 'StandardScaler')}")
            lines.append(f"{len(cols)} colonnes normalisées:\n")
            for col in cols:
                lines.append(f"  - {col}")

        return "\n".join(lines)

    def _build_split_summary(self, rapport: Dict[str, Any]) -> str:
        """Construit un résumé du split"""
        lines = ["=== TRAIN/TEST SPLIT ===\n"]
        lines.append(f"Train: {rapport.get('train_size', 0):,} lignes ({(1-rapport.get('test_ratio', 0.2))*100:.0f}%)")
        lines.append(f"Test: {rapport.get('test_size', 0):,} lignes ({rapport.get('test_ratio', 0.2)*100:.0f}%)")

        if rapport.get('stratified'):
            lines.append(f"\nStratification activée sur: '{rapport.get('target_column')}'")
        else:
            lines.append("\nStratification: non utilisée")

        return "\n".join(lines)
