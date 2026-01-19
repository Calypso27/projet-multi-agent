"""Détecteur de format de fichier"""
import pandas as pd
import io
from typing import Tuple, Optional


class FileDetector:
    """Détecte et charge différents formats de fichiers"""
    
    SUPPORTED_FORMATS = {
        'csv': 'Fichiers CSV',
        'xlsx': 'Excel',
        'xls': 'Excel (ancien)',
        'json': 'JSON',
        'parquet': 'Parquet',
        'tsv': 'TSV (Tab-separated)',
        'txt': 'Texte délimité'
    }
    
    @staticmethod
    def detect_format(filename: str) -> Optional[str]:
        """Détecte le format d'après l'extension"""
        if '.' not in filename:
            return None
        
        extension = filename.split('.')[-1].lower()
        return extension if extension in FileDetector.SUPPORTED_FORMATS else None
    
    @staticmethod
    def load_file(file_data, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Charge un fichier et retourne le DataFrame et un message d'erreur si applicable
        
        Returns:
            (DataFrame, error_message)
        """
        format_type = FileDetector.detect_format(filename)
        
        if not format_type:
            return None, f"Format non supporté. Extensions acceptées: {', '.join(FileDetector.SUPPORTED_FORMATS.keys())}"
        
        try:
            if format_type == 'csv':
                # Essayer différents encodages
                df = FileDetector._read_csv_with_encoding(file_data)
            
            elif format_type in ['xlsx', 'xls']:
                df = pd.read_excel(file_data)
            
            elif format_type == 'json':
                df = pd.read_json(file_data)
            
            elif format_type == 'parquet':
                df = pd.read_parquet(file_data)
            
            elif format_type == 'tsv':
                df = FileDetector._read_csv_with_encoding(file_data, sep='\t')
            
            elif format_type == 'txt':
                df = FileDetector._load_delimited_text(file_data)
            
            else:
                return None, f"Format {format_type} non supporté"
            
            return df, None
            
        except Exception as e:
            return None, f"Erreur lors du chargement: {str(e)}"
    
    @staticmethod
    def _read_csv_with_encoding(file_data, sep=','):
        """Essaie de lire un CSV avec différents encodages"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        
        for encoding in encodings:
            try:
                file_data.seek(0)
                df = pd.read_csv(file_data, sep=sep, encoding=encoding)
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                # Si ce n'est pas une erreur d'encodage, on la propage
                if 'codec' not in str(e).lower():
                    raise
        
        # Si aucun encodage ne fonctionne
        file_data.seek(0)
        return pd.read_csv(file_data, sep=sep, encoding='latin-1', errors='ignore')
    
    @staticmethod
    def _load_delimited_text(file_data) -> pd.DataFrame:
        """Charge un fichier texte en détectant le délimiteur"""
        delimiters = [',', '\t', ';', '|']
        
        for delimiter in delimiters:
            try:
                file_data.seek(0)
                df = FileDetector._read_csv_with_encoding(file_data, sep=delimiter)
                if len(df.columns) > 1:
                    return df
            except:
                continue
        
        file_data.seek(0)
        return FileDetector._read_csv_with_encoding(file_data)
    
    @staticmethod
    def get_supported_formats_string() -> str:
        """Retourne une chaîne listant les formats supportés"""
        formats = [f"{ext.upper()}" for ext in FileDetector.SUPPORTED_FORMATS.keys()]
        return ", ".join(formats)
