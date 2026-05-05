# pipeline.py

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any, List

warnings.filterwarnings('ignore')  # Désactiver les warnings de numpy/pandas pour la clarté


# =======================================================================
# --- 1. CLASSE DE NETTOYAGE DES DONNÉES (MusicDataCleaner) ---
# =======================================================================

class MusicDataCleaner:
    """Classe de nettoyage des données, adaptée pour fonctionner en mémoire (avec DF en entrée)."""

    def __init__(self):
        # Initialisation légère sans chemin de fichier
        self.df: Optional[pd.DataFrame] = None
        self.initial_shape: Optional[tuple] = None
        self.removed_columns: List[tuple] = []

    def set_data(self, df_input: pd.DataFrame):
        """Définit le DataFrame à nettoyer, utilisé par l'application Streamlit."""
        self.df = df_input.copy()
        self.initial_shape = self.df.shape
        return self

    def remove_uninformative_columns(self, missing_threshold: float = 50, unique_threshold: int = 1):
        """Supprime les colonnes non informatives."""
        for col in self.df.columns.copy():
            missing_percent = (self.df[col].isnull().sum() / len(self.df)) * 100
            unique_count = self.df[col].nunique()

            should_remove = False

            if missing_percent > missing_threshold:
                should_remove = True
            elif unique_count <= unique_threshold:
                should_remove = True
            elif self.df[col].dtype == 'object' and self.df[col].astype(str).str.len().max() <= 1:
                should_remove = True

            if should_remove:
                self.df = self.df.drop(columns=[col])
        return self

    def remove_duplicates(self):
        """Supprime les doublons exacts (basé sur les caractéristiques musicales)."""
        duplicate_cols = [col for col in self.df.columns if col not in ['Song_ID', 'Track_ID']]
        self.df = self.df.drop_duplicates(subset=duplicate_cols)
        return self

    def handle_missing_values(self, categorical_fill: str = 'Unknown'):
        """Gestion intelligente des valeurs manquantes."""
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    fill_val = self.df[col].mode()[0] if not self.df[col].mode().empty else categorical_fill
                    self.df[col].fillna(fill_val, inplace=True)
        return self

    def clean_text_columns(self):
        """Nettoyage et standardisation des colonnes textuelles."""
        text_columns = self.df.select_dtypes(include=['object']).columns
        for col in text_columns:
            self.df[col] = self.df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.replace(
                r'[\n\t\r]', ' ', regex=True)
            if col in ['Artist', 'Genre', 'Sub_Genre']:
                self.df[col] = self.df[col].str.title().str.replace(r'[^\w\s]', '', regex=True)
        return self

    def validate_music_ranges(self):
        """Valide et corrige les plages des métriques musicales."""
        ranges = {
            'Tempo_BPM': (40, 200), 'Energy_Score': (0, 100), 'Danceability_Score': (0, 100),
            'Valence_Score': (0, 100), 'Acousticness_Score': (0, 100), 'Instrumentalness_Score': (0, 100),
            'Popularity_Score': (0, 100), 'Loudness_dB': (-60, 0)
        }
        for col, (min_val, max_val) in ranges.items():
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                self.df[col] = self.df[col].clip(min_val, max_val)
        return self

    def run_complete_cleaning(self, df_input: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Exécute tout le processus de nettoyage musical (méthode principale)."""
        self.set_data(df_input)
        try:
            (self.remove_uninformative_columns()
             .remove_duplicates()
             .handle_missing_values()
             .clean_text_columns()
             .validate_music_ranges())
            return self.df
        except Exception as e:
            print(f"Erreur durant le nettoyage : {e}")
            return None


# =======================================================================
# --- 2. CLASSE DE PRÉ-TRAITEMENT DES DONNÉES (MusicPreprocessor) ---
# =======================================================================

class MusicPreprocessor:
    """Classe de pré-traitement des données, adaptée pour fonctionner en mémoire."""

    def __init__(self):
        # Initialisation légère sans chemin de fichier
        self.df: Optional[pd.DataFrame] = None
        self.df_processed: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()

    def set_data(self, df_input: pd.DataFrame):
        """Définit le DataFrame nettoyé à pré-traiter."""
        self.df = df_input.copy()
        return self

    def create_audio_features(self) -> pd.DataFrame:
        """Créer et normaliser les features audio NUMÉRIQUES."""
        base_features = [
            'Tempo_BPM', 'Energy_Score', 'Danceability_Score', 'Valence_Score',
            'Acousticness_Score', 'Instrumentalness_Score', 'Loudness_dB',
            'Speechiness_Score', 'Liveness_Score', 'Popularity_Score'
        ]
        available_features = [col for col in base_features if col in self.df.columns]
        audio_df = self.df[available_features].copy()

        # Features dérivées numériques
        engineered_features: Dict[str, pd.Series] = {}
        if 'Energy_Score' in self.df.columns and 'Danceability_Score' in self.df.columns:
            engineered_features['Energy_Dance_Ratio'] = (
                        self.df['Energy_Score'] / (self.df['Danceability_Score'] + 0.1))
        if 'Energy_Score' in self.df.columns and 'Valence_Score' in self.df.columns:
            engineered_features['Energetic_Positivity'] = (self.df['Energy_Score'] * self.df['Valence_Score'] / 100)
        for name, feature in engineered_features.items():
            audio_df[name] = feature

        audio_scaled = self.scaler.fit_transform(audio_df)
        return pd.DataFrame(audio_scaled, columns=audio_df.columns)

    def create_categorical_features(self) -> pd.DataFrame:
        """Créer les features catégorielles (One-Hot Encoding)."""
        categorical_data: Dict[str, pd.DataFrame] = {}

        # Logique de création des dummies (Tempo, Genre, Mood, Sentiment, Language)
        if 'Tempo_BPM' in self.df.columns:
            tempo_categories = pd.cut(self.df['Tempo_BPM'], bins=[0, 80, 120, 160, 200],
                                      labels=['Tempo_Lent', 'Tempo_Moyen', 'Tempo_Rapide', 'Tempo_Tres_Rapide'])
            categorical_data['tempo'] = pd.get_dummies(tempo_categories, prefix='Tempo')
        if 'Genre' in self.df.columns:
            categorical_data['genres'] = pd.get_dummies(self.df['Genre'], prefix='Genre')
        if 'Mood' in self.df.columns:
            categorical_data['moods'] = pd.get_dummies(self.df['Mood'], prefix='Mood')
        if 'Sentiment_Label' in self.df.columns:
            categorical_data['sentiments'] = pd.get_dummies(self.df['Sentiment_Label'], prefix='Sentiment')
        if 'Language' in self.df.columns:
            categorical_data['languages'] = pd.get_dummies(self.df['Language'], prefix='Language')

        if categorical_data:
            return pd.concat(categorical_data.values(), axis=1)
        else:
            return pd.DataFrame()

    def create_artist_features(self) -> pd.DataFrame:
        """Créer des features basées sur les artistes (Fréquence + Top Artists OHE)."""
        if 'Artist' in self.df.columns:
            artist_features: Dict[str, Any] = {}
            artist_counts = self.df['Artist'].value_counts()
            artist_features['Artist_Frequency'] = self.df['Artist'].map(artist_counts)

            top_artists = artist_counts.head(15).index
            for artist in top_artists:
                clean_name = ''.join(c if c.isalnum() else '_' for c in artist)
                artist_features[f'Artist_{clean_name}'] = (self.df['Artist'] == artist).astype(int)

            return pd.DataFrame(artist_features)
        else:
            return pd.DataFrame()

    def run_complete_preprocessing(self, df_input: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Exécute tout le processus de pré-traitement et combine les features avec métadonnées."""
        self.set_data(df_input)
        try:
            # 1. Construction des features
            audio_df = self.create_audio_features()
            categorical_df = self.create_categorical_features()
            artist_df = self.create_artist_features()

            # 2. Combiner toutes les features numériques/encodées
            all_features = pd.concat([audio_df, categorical_df, artist_df], axis=1).fillna(0)

            # 3. Récupérer les métadonnées pour la recommandation
            metadata_cols = []
            for col in ['Song_ID', 'Song_Name', 'Artist', 'Genre', 'Mood', 'Sentiment_Label']:
                if col in self.df.columns:
                    metadata_cols.append(col)

            # 4. Concaténer les métadonnées et les features
            if metadata_cols:
                final_df = pd.concat([
                    self.df[metadata_cols].reset_index(drop=True),
                    all_features.reset_index(drop=True)
                ], axis=1)
            else:
                final_df = all_features

            self.df_processed = final_df
            return final_df

        except Exception as e:
            print(f"Erreur durant le pré-traitement : {e}")
            return None