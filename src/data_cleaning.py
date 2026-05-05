"""
data_cleaner.py
Classe de nettoyage des données pour musique - Version recommandation
"""
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class MusicDataCleaner:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.initial_shape = None
        self.removed_columns = []

    def load_data(self):
        """Charger les données avec gestion robuste des erreurs CSV"""
        print("📥 Chargement des données musicales...")

        if self.file_path.endswith('.csv'):
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

            for encoding in encodings:
                try:
                    self.df = pd.read_csv(
                        self.file_path,
                        encoding=encoding,
                        sep=None,  # Détection automatique
                        engine='python',
                        on_bad_lines='skip'
                    )
                    if not self.df.empty:
                        print(f"✅ Fichier chargé avec encodage: {encoding}")
                        break
                except Exception as e:
                    continue
            else:
                raise ValueError("❌ Impossible de lire le fichier CSV")
        else:
            self.df = pd.read_excel(self.file_path)
            print("✅ Fichier Excel chargé")

        self.initial_shape = self.df.shape
        print(f"✅ Données chargées : {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")
        return self

    def inspect_data(self):
        """Inspection détaillée des données musicales"""
        print("\n" + "="*60)
        print("📊 INSPECTION APPROFONDIE DES DONNÉES MUSICALES")
        print("="*60)

        print(f"\n📐 Dimensions : {self.df.shape}")

        print(f"\n📋 Types de données :")
        print(self.df.dtypes.value_counts())

        # Analyse détaillée par colonne
        print(f"\n🔍 ANALYSE PAR COLONNE :")
        print("-" * 50)

        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_percent = (missing_count / len(self.df)) * 100
            unique_count = self.df[col].nunique()

            print(f"\n📊 {col}:")
            print(f"   Type: {self.df[col].dtype}")
            print(f"   Valeurs manquantes: {missing_count} ({missing_percent:.1f}%)")
            print(f"   Valeurs uniques: {unique_count}")

            if self.df[col].dtype in ['object']:
                top_values = self.df[col].value_counts().head(3)
                print(f"   Top valeurs: {dict(top_values)}")
            elif self.df[col].dtype in ['int64', 'float64']:
                print(f"   Min: {self.df[col].min():.2f}, Max: {self.df[col].max():.2f}")
                print(f"   Moyenne: {self.df[col].mean():.2f}, Médiane: {self.df[col].median():.2f}")

        print(f"\n🔄 Doublons : {self.df.duplicated().sum()}")

        # Statistiques musicales spécifiques
        self._analyze_music_features()

        return self

    def _analyze_music_features(self):
        """Analyser les caractéristiques musicales spécifiques"""
        print(f"\n🎵 ANALYSE DES CARACTÉRISTIQUES MUSICALES :")
        print("-" * 40)

        # Colonnes typiques des datasets musicaux
        music_features = ['Tempo_BPM', 'Energy_Score', 'Danceability_Score', 'Valence_Score',
                         'Acousticness_Score', 'Instrumentalness_Score', 'Loudness_dB']

        for feature in music_features:
            if feature in self.df.columns:
                print(f"   {feature}: {self.df[feature].min():.1f}-{self.df[feature].max():.1f} "
                      f"(moy: {self.df[feature].mean():.1f})")

        # Analyse des genres
        if 'Genre' in self.df.columns:
            print(f"\n   Genres uniques: {self.df['Genre'].nunique()}")
            print(f"   Top 5 genres: {dict(self.df['Genre'].value_counts().head())}")

    def remove_uninformative_columns(self, missing_threshold=50, unique_threshold=1):
        """
        Supprimer SEULEMENT les colonnes vraiment non informatives
        Conserver toutes les features musicales pour la recommandation
        """
        print("\n" + "="*60)
        print("🗑️  SUPPRESSION DES COLONNES NON INFORMATIVES")
        print("="*60)

        initial_cols = len(self.df.columns)

        for col in self.df.columns.copy():
            missing_percent = (self.df[col].isnull().sum() / len(self.df)) * 100
            unique_count = self.df[col].nunique()

            # CRITÈRES STRICTS de suppression
            should_remove = False
            reason = ""

            # Seulement si vraiment inutile
            if missing_percent > missing_threshold:
                should_remove = True
                reason = f"trop de valeurs manquantes ({missing_percent:.1f}%)"
            elif unique_count <= unique_threshold:
                should_remove = True
                reason = f"une seule valeur unique ({unique_count})"
            elif self.df[col].dtype == 'object' and self.df[col].str.len().max() <= 1:
                should_remove = True
                reason = "colonnes texte vides"

            if should_remove:
                self.df = self.df.drop(columns=[col])
                self.removed_columns.append((col, reason))
                print(f"🗑️  Supprimé '{col}' : {reason}")

        final_cols = len(self.df.columns)
        print(f"\n✅ Colonnes supprimées : {initial_cols - final_cols}")
        print(f"📊 Colonnes restantes : {final_cols}")

        return self

    def remove_duplicates(self):
        """Supprimer les doublons exacts (basé sur les caractéristiques musicales)"""
        print("\n" + "="*60)
        print("🔄 SUPPRESSION DES DOUBLONS")
        print("="*60)

        initial_rows = len(self.df)

        # Colonnes pour identifier les doublons (exclure ID)
        duplicate_cols = [col for col in self.df.columns if col not in ['Song_ID', 'Track_ID']]
        self.df = self.df.drop_duplicates(subset=duplicate_cols)

        removed = initial_rows - len(self.df)

        if removed > 0:
            print(f"✅ {removed} doublons musicaux supprimés")
        else:
            print("✅ Aucun doublon trouvé")

        print(f"📊 Lignes restantes : {len(self.df)}")
        return self

    def handle_missing_values(self, categorical_fill='Unknown'):
        """Gestion intelligente des valeurs manquantes pour données musicales"""
        print("\n" + "="*60)
        print("❌ GESTION DES VALEURS MANQUANTES")
        print("="*60)

        initial_missing = self.df.isnull().sum().sum()

        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            if missing_count > 0:
                missing_percent = (missing_count / len(self.df)) * 100

                if self.df[col].dtype in ['int64', 'float64']:
                    # Pour les métriques musicales : médiane
                    fill_val = self.df[col].median()
                    self.df[col].fillna(fill_val, inplace=True)
                    print(f"🔢 '{col}' : {missing_count} valeurs → médiane ({fill_val:.2f})")
                else:
                    # Pour les catégorielles : mode ou valeur spécifique
                    if not self.df[col].mode().empty and len(self.df[col].mode()) > 0:
                        fill_val = self.df[col].mode()[0]
                    else:
                        fill_val = categorical_fill

                    self.df[col].fillna(fill_val, inplace=True)
                    print(f"📝 '{col}' : {missing_count} valeurs → '{fill_val}'")

        final_missing = self.df.isnull().sum().sum()
        print(f"\n✅ Valeurs manquantes traitées : {initial_missing} → {final_missing}")
        return self

    def clean_text_columns(self):
        """Nettoyage des colonnes textuelles pour la recommandation musicale"""
        print("\n" + "="*60)
        print("🧹 NETTOYAGE DES COLONNES TEXTUELLES")
        print("="*60)

        text_columns = self.df.select_dtypes(include=['object']).columns

        for col in text_columns:
            # Conversion en string
            self.df[col] = self.df[col].astype(str)

            # Nettoyage de base
            self.df[col] = self.df[col].str.strip()
            self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
            self.df[col] = self.df[col].str.replace(r'[\n\t\r]', ' ', regex=True)

            # Standardisation des noms d'artistes et genres
            if col in ['Artist', 'Genre', 'Sub_Genre']:
                self.df[col] = self.df[col].str.title()
                self.df[col] = self.df[col].str.replace(r'[^\w\s]', '', regex=True)

            print(f"✅ '{col}' nettoyé")

        return self

    def validate_music_ranges(self):
        """Valider les plages des métriques musicales"""
        print("\n" + "="*60)
        print("🎵 VALIDATION DES PLAGES MUSICALES")
        print("="*60)

        # Plages standards pour les métriques musicales
        ranges = {
            'Tempo_BPM': (40, 200),
            'Energy_Score': (0, 100),
            'Danceability_Score': (0, 100),
            'Valence_Score': (0, 100),
            'Acousticness_Score': (0, 100),
            'Instrumentalness_Score': (0, 100),
            'Popularity_Score': (0, 100),
            'Loudness_dB': (-60, 0)
        }

        for col, (min_val, max_val) in ranges.items():
            if col in self.df.columns:
                out_of_range = ((self.df[col] < min_val) | (self.df[col] > max_val)).sum()
                if out_of_range > 0:
                    print(f"⚠️  '{col}' : {out_of_range} valeurs hors plage [{min_val}-{max_val}]")
                    # Corriger les valeurs aberrantes
                    self.df[col] = self.df[col].clip(min_val, max_val)
                    print(f"   ✅ Valeurs corrigées")
                else:
                    print(f"✅ '{col}' : toutes les valeurs dans la plage [{min_val}-{max_val}]")

        return self

    def save_cleaned_data(self, output_path='../data/cleaned/music_cleaned.csv'):
        """Sauvegarder dans ../data/cleaned/"""
        print("\n" + "="*60)
        print("💾 SAUVEGARDE DES DONNÉES NETTOYÉES")
        print("="*60)

        # Créer le dossier ../data/cleaned s'il n'existe pas
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Sauvegarder
        self.df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"📂 Dossier : {os.path.dirname(output_path)}")
        print(f"💾 Fichier : {os.path.basename(output_path)}")
        print(f"📊 Dimensions finales : {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes")

        # Rapport final
        self._generate_final_report()

        return output_path

    def _generate_final_report(self):
        """Générer un rapport final spécifique musique"""
        print(f"\n📈 RAPPORT FINAL DE NETTOYAGE MUSICAL :")
        print("-" * 40)

        if self.initial_shape:
            print(f"   Lignes : {self.initial_shape[0]} → {self.df.shape[0]}")
            print(f"   Colonnes : {self.initial_shape[1]} → {self.df.shape[1]}")

        print(f"   Valeurs manquantes : {self.df.isnull().sum().sum()}")
        print(f"   Doublons : {self.df.duplicated().sum()}")

        # Statistiques musicales
        if 'Genre' in self.df.columns:
            print(f"   Genres uniques : {self.df['Genre'].nunique()}")
        if 'Artist' in self.df.columns:
            print(f"   Artistes uniques : {self.df['Artist'].nunique()}")

        if self.removed_columns:
            print(f"\n   Colonnes supprimées :")
            for col, reason in self.removed_columns:
                print(f"     - {col} : {reason}")
        else:
            print(f"\n   ✅ Aucune colonne informative supprimée")

    def run_complete_cleaning(self, output_path='../data/cleaned/music_cleaned.csv'):
        """Exécuter tout le processus de nettoyage musical"""
        print("🚀 DÉMARRAGE DU NETTOYAGE MUSICAL")
        print("="*60)

        try:
            (self.load_data()
               .inspect_data()
               .remove_uninformative_columns(missing_threshold=50, unique_threshold=1)
               .remove_duplicates()
               .handle_missing_values()
               .clean_text_columns()
               .validate_music_ranges()
               .save_cleaned_data(output_path))

            print("\n🎯 NETTOYAGE MUSICAL TERMINÉ AVEC SUCCÈS!")
            print("💡 Données optimisées pour la recommandation musicale")
            return self.df

        except Exception as e:
            print(f"\n❌ ERREUR : {str(e)}")
            raise

