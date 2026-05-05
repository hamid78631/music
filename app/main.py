# app.py (Application Web Streamlit - Ajout de la page Pipeline de Données)

import streamlit as st
import pandas as pd
import numpy as np
import io # Pour gérer les fichiers en mémoire
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ***************************************************************
# *** MODIFICATION CLÉ : IMPORTATION DES MODULES D'ANALYSE ***
# ***************************************************************
try:
    # 1. Importation de la classe d'analyse (PCA)
    from pca_analysis_module import PCAAnalysis3D

    # 2. Importation des classes de pipeline
    from data_cleaning_module import MusicDataCleaner
    from preprocessing_module import MusicPreprocessor

except ImportError as e:
    st.error(f"❌ ERREUR: Impossible d'importer un module nécessaire. {e}")
    st.error("Assurez-vous que les classes MusicDataCleaner, MusicPreprocessor, et PCAAnalysis3D sont disponibles dans des modules importables (ou copiez leur code ici).")
    st.stop()


# --- 1. CONFIGURATION ET FONCTIONS UTILES ---

# Configuration de la page
st.set_page_config(
    layout="wide",
    page_title="Recommandation Musicale IA",
    initial_sidebar_state="expanded"
)

# Utiliser le cache pour stocker les DataFrames chargés
@st.cache_data
def load_data_from_upload(uploaded_file):
    """Charge le DataFrame depuis le fichier téléversé."""
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.data_original = df
        st.success("✅ Fichier chargé en mémoire. Prêt pour le nettoyage.")
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
        return None

def run_cleaning_from_app(df_to_clean):
    """Exécute la logique de nettoyage via MusicDataCleaner."""
    st.info("🧹 Démarrage du nettoyage des données...")
    try:
        # Créer une instance de Cleaner, en utilisant le DataFrame en mémoire
        # NOTE: La classe MusicDataCleaner doit être adaptée pour prendre un DF directement
        # ou nous devons enregistrer le fichier temporairement.

        # Solution simple : Utiliser une approche fonctionnelle sur l'objet de classe
        cleaner = MusicDataCleaner(file_path=None) # file_path=None si l'on travaille en mémoire
        cleaner.df = df_to_clean.copy() # Assigner le DF chargé

        # Exécuter les étapes de nettoyage sans sauvegarde sur disque
        (cleaner
           .remove_uninformative_columns()
           .remove_duplicates()
           .handle_missing_values()
           .clean_text_columns()
           .validate_music_ranges())

        st.session_state.data_cleaned = cleaner.df
        st.success(f"✅ Nettoyage terminé. Lignes restantes : {cleaner.df.shape[0]}")
        return cleaner.df

    except Exception as e:
        st.error(f"❌ ERREUR lors du Nettoyage : {e}")
        return None

def run_preprocessing_from_app(df_cleaned):
    """Exécute la logique de préprocessing via MusicPreprocessor."""
    st.info("🔨 Démarrage du préprocessing (Feature Engineering)...")
    try:
        # Créer une instance de Preprocessor
        preprocessor = MusicPreprocessor(data_path=None) # Travail en mémoire
        preprocessor.df = df_cleaned.copy()

        # Exécuter les étapes de feature engineering et de scaling
        # (Nous utilisons la méthode de run_complete_preprocessing de la classe)

        # 1. Création de toutes les features SÉPARÉMENT
        audio_df = preprocessor.create_audio_features()
        categorical_df = preprocessor.create_categorical_features()
        artist_df = preprocessor.create_artist_features()

        # 2. Combiner toutes les features
        all_features = pd.concat([audio_df, categorical_df, artist_df], axis=1).fillna(0)

        st.session_state.data_processed = all_features
        st.success(f"✅ Préprocessing terminé. {all_features.shape[1]} features créées.")
        return all_features

    except Exception as e:
        st.error(f"❌ ERREUR lors du Préprocessing : {e}")
        return None

def run_pca_from_app(df_features):
    """Exécute l'analyse PCA et le Clustering."""
    st.info("🔍 Démarrage de l'analyse PCA & Clustering...")
    try:
        # 1. Initialiser l'analyse
        analysis = PCAAnalysis3D(n_components=3)

        # 2. Préparer les données pour la PCA (scaling et nettoyage sont gérés dans l'objet)
        # On passe le DF de features directement à un wrapper PCA (doit être adapté)

        # Solution: réappliquer le scaler pour être sûr (même si déjà fait dans preprocessing)
        analysis.data = df_features
        analysis.data_numeric = df_features.select_dtypes(include=[np.number]).fillna(0)
        analysis.X_scaled = analysis.scaler.fit_transform(analysis.data_numeric)

        # 3. Exécuter l'analyse complète
        analysis.perform_pca()
        optimal_k = analysis.find_optimal_clusters(max_k=8, plot_elbow=False) # Pas de plot ici
        analysis.perform_clustering(n_clusters=optimal_k)

        # Stocker les résultats
        st.session_state.pca_results = analysis
        st.success(f"✅ Analyse PCA terminée. Variance expliquée (3PC): {analysis.cumulative_variance[-1]:.1%}")
        st.success(f"✅ Clustering terminé. {optimal_k} clusters identifiés.")
        return analysis

    except Exception as e:
        st.error(f"❌ ERREUR lors de l'Analyse PCA/Clustering : {e}")
        return None

# --- Fonctions précédentes (load_data_and_analyze, get_recommendations, pages de visualisation) ---

# ***************************************************************
# *** NOUVELLE PAGE : PIPELINE INTERACTIF ***
# ***************************************************************

def pipeline_page():
    """Page dédiée à l'importation et au traitement des données."""
    st.title("⚙️ Pipeline de Données Interactif")
    st.subheader("Importez votre fichier, nettoyez-le et pré-traitez-le en direct.")
    st.markdown("---")

    # --- 1. IMPORTATION DU FICHIER CSV ---
    st.header("1. Téléversement du Fichier CSV")
    uploaded_file = st.file_uploader(
        "Sélectionnez un fichier CSV à analyser. Assurez-vous qu'il contient les colonnes musicales nécessaires (Tempo, Energy, Artist, etc.).",
        type="csv"
    )

    if uploaded_file is not None:
        df_original = load_data_from_upload(uploaded_file)

        if df_original is not None:
            st.markdown(f"**Fichier chargé :** {uploaded_file.name} ({df_original.shape[0]} lignes, {df_original.shape[1]} colonnes)")

            # Aperçu (style moderne)
            st.dataframe(df_original.head(), use_container_width=True)
            st.markdown("---")

            # --- 2. NETTOYAGE DES DONNÉES ---
            st.header("2. Nettoyage des Données")

            if st.button("🧹 Lancer le Nettoyage (MusicDataCleaner)", key="btn_clean"):
                df_cleaned = run_cleaning_from_app(df_original)
                if df_cleaned is not None:
                    st.success("Nettoyage réussi. Aperçu des données nettoyées :")
                    st.dataframe(df_cleaned.head(), use_container_width=True)

            # --- 3. PRÉTRAITEMENT & ENGINEERING ---
            if 'data_cleaned' in st.session_state:
                st.header("3. Pré-traitement (Feature Engineering)")

                if st.button("🔨 Lancer le Pré-traitement (MusicPreprocessor)", key="btn_preprocess"):
                    df_features = run_preprocessing_from_app(st.session_state.data_cleaned)
                    if df_features is not None:
                        st.success("Pré-traitement réussi. Aperçu des Features :")
                        st.dataframe(df_features.head(), use_container_width=True)
                        st.metric("Nouvelle Dimension", f"{df_features.shape[1]} Features")

                        # Bouton d'export (Téléchargement)
                        csv_export = df_features.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="⬇️ Exporter le fichier de FEATURES (CSV)",
                            data=csv_export,
                            file_name='music_features_processed.csv',
                            mime='text/csv',
                            key="export_features"
                        )

            # --- 4. ANALYSE PCA ET CLUSTERING ---
            if 'data_processed' in st.session_state:
                st.header("4. Analyse et Modélisation (PCA/KMeans)")

                if st.button("🔍 Lancer l'Analyse PCA & Clustering", key="btn_pca"):
                    analysis_results = run_pca_from_app(st.session_state.data_processed)
                    if analysis_results is not None:
                         st.success("Analyse terminée ! Vous pouvez maintenant basculer vers les visualisations et l'application.")
                         st.info("Note : Pour utiliser ces résultats dans les autres pages, vous devez recharger l'application ou adapter la mise en cache.")


# --- MAIN APP EXECUTION ---

def main():

    # Initialisation des sessions d'état pour le pipeline
    if 'data_original' not in st.session_state:
        st.session_state.data_original = None
    if 'data_cleaned' not in st.session_state:
        st.session_state.data_cleaned = None
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = None
    if 'pca_results' not in st.session_state:
        st.session_state.pca_results = None

    # --- BARRE LATÉRALE DE NAVIGATION (NAVBAR) ---
    st.sidebar.markdown("# 🎧 Menu")

    page_selection = st.sidebar.radio(
        "Choisissez votre vue",
        ('Pipeline de Données', 'Application', 'Visualisation 3D', 'Visualisation 2D')
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("🚀 Interface par Streamlit")

    # --- ROUTAGE DES PAGES ---
    if page_selection == 'Pipeline de Données':
        pipeline_page()
    else:
        # Pour les pages d'analyse, on charge les résultats stockés dans la session state
        # (Ou on charge les fichiers sauvegardés si l'on veut un fonctionnement permanent)
        if st.session_state.pca_results:
            analysis = st.session_state.pca_results
            # Le DataFrame doit être construit avec les clusters
            df_results = pd.DataFrame(analysis.X_pca, columns=[f'PC{i+1}' for i in range(analysis.pca.n_components_)])
            df_results['Cluster'] = analysis.clusters
            # Ajouter les métadonnées (Simplification)
            if 'Song_Name' in analysis.data_original.columns:
                df_results['Song_Name'] = analysis.data_original['Song_Name']
            if 'Artist' in analysis.data_original.columns:
                df_results['Artist'] = analysis.data_original['Artist']
            if 'Genre' in analysis.data_original.columns:
                df_results['Genre'] = analysis.data_original['Genre']

            if page_selection == 'Application':
                application_page(analysis, df_results)
            elif page_selection == 'Visualisation 3D':
                plot_3d_page(df_results, analysis.pca)
            elif page_selection == 'Visualisation 2D':
                plot_2d_page(df_results, analysis.pca)
        else:
            st.warning("⚠️ Veuillez d'abord exécuter les étapes 1 à 4 dans la page **'Pipeline de Données'**.")

if __name__ == "__main__":
    main()