# app.py (VERSION ULTRA-AMÉLIORÉE avec TOUTES les nouvelles fonctionnalités)

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from pathlib import Path
import json
from datetime import datetime

warnings.filterwarnings('ignore')


# =======================================================================
# --- 1. CLASSES DU PIPELINE INTÉGRÉES ---
# =======================================================================

class MusicDataCleaner:
    """Classe de nettoyage des données."""

    def __init__(self):
        self.file_path = None
        self.df = None
        self.initial_shape = None
        self.removed_columns = []

    def set_data(self, df_input):
        self.df = df_input.copy()
        self.initial_shape = self.df.shape
        return self

    def remove_uninformative_columns(self, missing_threshold=50, unique_threshold=1):
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
        duplicate_cols = [col for col in self.df.columns if col not in ['Song_ID', 'Track_ID']]
        self.df = self.df.drop_duplicates(subset=duplicate_cols)
        return self

    def handle_missing_values(self, categorical_fill='Unknown'):
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].dtype in ['int64', 'float64']:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    if not self.df[col].mode().empty and len(self.df[col].mode()) > 0:
                        fill_val = self.df[col].mode()[0]
                    else:
                        fill_val = categorical_fill
                    self.df[col].fillna(fill_val, inplace=True)
        return self

    def clean_text_columns(self):
        text_columns = self.df.select_dtypes(include=['object']).columns
        for col in text_columns:
            self.df[col] = self.df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.replace(
                r'[\n\t\r]', ' ', regex=True)
            if col in ['Artist', 'Genre', 'Sub_Genre']:
                self.df[col] = self.df[col].str.title().str.replace(r'[^\w\s]', '', regex=True)
        return self

    def validate_music_ranges(self):
        ranges = {'Tempo_BPM': (40, 200), 'Energy_Score': (0, 100), 'Danceability_Score': (0, 100),
                  'Valence_Score': (0, 100), 'Acousticness_Score': (0, 100), 'Instrumentalness_Score': (0, 100),
                  'Popularity_Score': (0, 100), 'Loudness_dB': (-60, 0)}
        for col, (min_val, max_val) in ranges.items():
            if col in self.df.columns and self.df[col].dtype in ['int64', 'float64']:
                self.df[col] = self.df[col].clip(min_val, max_val)
        return self

    def run_complete_cleaning(self, df_input):
        self.set_data(df_input)
        try:
            (self.remove_uninformative_columns()
             .remove_duplicates()
             .handle_missing_values()
             .clean_text_columns()
             .validate_music_ranges())
            return self.df
        except Exception as e:
            st.error(f"Erreur durant le nettoyage : {e}")
            return None


class MusicPreprocessor:
    """Classe de pré-traitement des données."""

    def __init__(self):
        self.data_path = None
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.feature_names = []

    def set_data(self, df_input):
        self.df = df_input.copy()
        return self

    def create_audio_features(self):
        base_features = ['Tempo_BPM', 'Energy_Score', 'Danceability_Score', 'Valence_Score', 'Acousticness_Score',
                         'Instrumentalness_Score', 'Loudness_dB', 'Speechiness_Score', 'Liveness_Score',
                         'Popularity_Score']
        available_features = [col for col in base_features if col in self.df.columns]
        audio_df = self.df[available_features].copy()
        engineered_features = {}
        if 'Energy_Score' in self.df.columns and 'Danceability_Score' in self.df.columns:
            engineered_features['Energy_Dance_Ratio'] = (
                        self.df['Energy_Score'] / (self.df['Danceability_Score'] + 0.1))
        if 'Energy_Score' in self.df.columns and 'Valence_Score' in self.df.columns:
            engineered_features['Energetic_Positivity'] = (self.df['Energy_Score'] * self.df['Valence_Score'] / 100)
        for name, feature in engineered_features.items():
            audio_df[name] = feature
        audio_scaled = self.scaler.fit_transform(audio_df)
        return pd.DataFrame(audio_scaled, columns=audio_df.columns)

    def create_categorical_features(self):
        categorical_data = {}
        if 'Tempo_BPM' in self.df.columns:
            categorical_data['tempo'] = pd.get_dummies(
                pd.cut(self.df['Tempo_BPM'], bins=[0, 80, 120, 160, 200],
                       labels=['Lent', 'Moyen', 'Rapide', 'Tres_Rapide']), prefix='Tempo')
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

    def create_artist_features(self):
        if 'Artist' in self.df.columns:
            artist_features = {}
            artist_counts = self.df['Artist'].value_counts()
            artist_features['Artist_Frequency'] = self.df['Artist'].map(artist_counts)
            top_artists = artist_counts.head(15).index
            for artist in top_artists:
                clean_name = ''.join(c if c.isalnum() else '_' for c in artist)
                artist_features[f'Artist_{clean_name}'] = (self.df['Artist'] == artist).astype(int)
            return pd.DataFrame(artist_features)
        else:
            return pd.DataFrame()

    def run_complete_preprocessing(self, df_input):
        self.set_data(df_input)
        try:
            audio_df = self.create_audio_features()
            categorical_df = self.create_categorical_features()
            artist_df = self.create_artist_features()
            all_features = pd.concat([audio_df, categorical_df, artist_df], axis=1).fillna(0)
            metadata_cols = []
            for col in ['Song_ID', 'Song_Name', 'Artist', 'Genre', 'Mood', 'Sentiment_Label']:
                if col in self.df.columns:
                    metadata_cols.append(col)
            if metadata_cols:
                final_df = pd.concat(
                    [self.df[metadata_cols].reset_index(drop=True), all_features.reset_index(drop=True)], axis=1)
            else:
                final_df = all_features
            self.df_processed = final_df
            return final_df
        except Exception as e:
            st.error(f"Erreur durant le pré-traitement : {e}")
            return None


class PCAAnalysis3D:
    """Classe d'analyse PCA et Clustering."""

    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.clusters = None
        self.silhouette_score = 0.0
        self.data_original = None
        self.data_numeric = None
        self.X_scaled = None
        self.X_pca = None
        self.cumulative_variance = None
        self.kmeans = None

    def load_and_preprocess_data(self, data):
        self.data_original = data.copy()
        self.data_numeric = data.select_dtypes(include=[np.number])
        if self.data_numeric.isnull().sum().sum() > 0:
            self.data_numeric = self.data_numeric.fillna(self.data_numeric.median())
        self.X_scaled = self.scaler.fit_transform(self.data_numeric)
        return self.X_scaled

    def perform_pca(self):
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        return self.X_pca

    def perform_clustering(self, n_clusters=4):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
        self.clusters = self.kmeans.fit_predict(self.X_pca)
        if len(np.unique(self.clusters)) > 1:
            self.silhouette_score = silhouette_score(self.X_pca, self.clusters)
        else:
            self.silhouette_score = 0.0
        return self.clusters

    def find_optimal_clusters(self, max_k=8):
        silhouette_scores = []
        possible_k_values = range(2, max_k + 1)
        for k in possible_k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(self.X_pca)
            if len(np.unique(clusters)) > 1:
                silhouette_scores.append(silhouette_score(self.X_pca, clusters))
            else:
                silhouette_scores.append(-1)
        if any(s != -1 for s in silhouette_scores):
            max_silhouette_idx = np.argmax([s if s != -1 else -2 for s in silhouette_scores])
            optimal_k = possible_k_values[max_silhouette_idx]
            return optimal_k
        else:
            return 4


@st.cache_data(show_spinner="Analyse PCA et Clustering en cours...")
def run_pca_clustering_cached(df_features_to_analyze):
    """Fonction CACHÉE pour exécuter la PCA et le Clustering sur 5000 lignes."""
    sample_size = min(5000, df_features_to_analyze.shape[0])
    df_sample = df_features_to_analyze.sample(n=sample_size, random_state=42)
    analysis = PCAAnalysis3D(n_components=3)
    analysis.load_and_preprocess_data(data=df_sample)
    analysis.perform_pca()
    optimal_k = analysis.find_optimal_clusters(max_k=8)
    analysis.perform_clustering(n_clusters=optimal_k)
    return analysis


# =======================================================================
# --- 2. FONCTIONS INTERACTIVES DU PIPELINE ---
# =======================================================================

@st.cache_data
def load_data_from_upload(uploaded_file):
    """Charge le DataFrame depuis le fichier téléversé."""
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', sep=None, engine='python', on_bad_lines='skip')
        st.session_state.data_original = df
        st.session_state.data_cleaned = None
        st.session_state.data_processed = None
        st.session_state.pca_results = None
        return df
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier CSV : {e}")
        return None


def run_cleaning_from_app(df_to_clean):
    """Exécute la logique de nettoyage via MusicDataCleaner."""
    if df_to_clean is None:
        return None
    with st.spinner("🧹 Nettoyage en cours..."):
        cleaner = MusicDataCleaner()
        df_cleaned = cleaner.run_complete_cleaning(df_to_clean)
        if df_cleaned is not None:
            st.session_state.data_cleaned = df_cleaned
            st.success(f"✅ Nettoyage terminé. Lignes restantes : {df_cleaned.shape[0]}")
        return df_cleaned


def run_preprocessing_from_app(df_cleaned):
    """Exécute la logique de preprocessing via MusicPreprocessor."""
    if df_cleaned is None:
        return None
    with st.spinner("🔨 Pré-traitement et Feature Engineering en cours..."):
        preprocessor = MusicPreprocessor()
        df_features = preprocessor.run_complete_preprocessing(df_cleaned)
        if df_features is not None:
            st.session_state.data_processed = df_features
            st.success(f"✅ Pré-traitement terminé. {df_features.shape[1]} features et métadonnées combinées.")
        return df_features


def run_pca_from_app(df_features):
    """Exécute l'analyse PCA et le Clustering."""
    analysis = run_pca_clustering_cached(df_features)
    st.session_state.pca_results = analysis
    st.success(
        f"✅ Analyse terminée (Basée sur 5k échantillons). Clusters: {analysis.clusters.max() + 1} (Score Silhouette: {analysis.silhouette_score:.3f}).")
    return analysis


# =======================================================================
# --- 3. LOGIQUE DE RECOMMANDATION ---
# =======================================================================

def get_recommendations(df_features, selected_song_index, n_recommendations=5):
    """Recommandation par similarité Cosinus dans l'espace PCA 3D."""
    feature_cols = [col for col in df_features.columns if col.startswith('PC')]
    X = df_features[feature_cols].values
    cosine_sim = cosine_similarity(X)
    sim_scores = list(enumerate(cosine_sim[selected_song_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n_recommendations + 1]
    song_indices = [i[0] for i in sim_scores]
    recommendations_df = df_features.iloc[song_indices].copy()
    recommendations_df['Similarity_Score'] = [f"{i[1] * 100:.2f}%" for i in sim_scores]
    display_cols = ['Song_Name', 'Artist', 'Genre', 'Cluster', 'Similarity_Score']
    return recommendations_df[[col for col in display_cols if col in recommendations_df.columns]]


# =======================================================================
# --- 4. NOUVELLES FONCTIONNALITÉS : PLAYLISTS THÉMATIQUES ---
# =======================================================================

def create_themed_playlists(df_cleaned):
    """Crée automatiquement des playlists thématiques basées sur les caractéristiques musicales."""
    playlists = {}
    required_cols = ['Song_Name', 'Artist']
    optional_cols = ['Energy_Score', 'Valence_Score', 'Tempo_BPM', 'Acousticness_Score',
                     'Instrumentalness_Score', 'Speechiness_Score', 'Danceability_Score', 'Genre']
    available_cols = [col for col in optional_cols if col in df_cleaned.columns]
    if not all(col in df_cleaned.columns for col in required_cols):
        st.warning("⚠️ Colonnes Song_Name ou Artist manquantes pour créer des playlists.")
        return playlists

    # Playlist 1: Morning Energy ☀️
    if all(col in df_cleaned.columns for col in ['Energy_Score', 'Valence_Score', 'Tempo_BPM']):
        morning = df_cleaned[
            (df_cleaned['Energy_Score'] > 70) &
            (df_cleaned['Valence_Score'] > 60) &
            (df_cleaned['Tempo_BPM'] > 120)
            ].head(20)
        if not morning.empty:
            playlists['☀️ Morning Energy'] = morning

    # Playlist 2: Focus & Concentration 🎯
    if all(col in df_cleaned.columns for col in ['Instrumentalness_Score', 'Speechiness_Score', 'Energy_Score']):
        focus = df_cleaned[
            (df_cleaned['Instrumentalness_Score'] > 50) &
            (df_cleaned['Speechiness_Score'] < 10) &
            (df_cleaned['Energy_Score'].between(30, 60))
            ].head(20)
        if not focus.empty:
            playlists['🎯 Focus & Concentration'] = focus

    # Playlist 3: Chill & Relax 😌
    if all(col in df_cleaned.columns for col in ['Acousticness_Score', 'Energy_Score', 'Tempo_BPM']):
        chill = df_cleaned[
            (df_cleaned['Acousticness_Score'] > 40) &
            (df_cleaned['Energy_Score'] < 50) &
            (df_cleaned['Tempo_BPM'] < 100)
            ].head(20)
        if not chill.empty:
            playlists['😌 Chill & Relax'] = chill

    # Playlist 4: Workout 💪
    if all(col in df_cleaned.columns for col in ['Energy_Score', 'Danceability_Score', 'Tempo_BPM']):
        workout = df_cleaned[
            (df_cleaned['Energy_Score'] > 80) &
            (df_cleaned['Danceability_Score'] > 70) &
            (df_cleaned['Tempo_BPM'] > 140)
            ].head(20)
        if not workout.empty:
            playlists['💪 Workout'] = workout

    # Playlist 5: Party Mode 🎉
    if all(col in df_cleaned.columns for col in ['Energy_Score', 'Danceability_Score', 'Valence_Score']):
        party = df_cleaned[
            (df_cleaned['Energy_Score'] > 75) &
            (df_cleaned['Danceability_Score'] > 75) &
            (df_cleaned['Valence_Score'] > 70)
            ].head(20)
        if not party.empty:
            playlists['🎉 Party Mode'] = party

    return playlists


# =======================================================================
# --- 5. NOUVELLES FONCTIONNALITÉS : GRAPHIQUE RADAR ---
# =======================================================================

def plot_radar_chart(song_data, song_name="Chanson"):
    """Crée un graphique radar pour visualiser les caractéristiques d'une chanson."""
    features_to_plot = ['Energy_Score', 'Danceability_Score', 'Valence_Score',
                        'Acousticness_Score', 'Instrumentalness_Score']
    available_features = [f for f in features_to_plot if f in song_data.index]
    if len(available_features) < 3:
        st.warning("⚠️ Pas assez de caractéristiques disponibles pour créer le graphique radar.")
        return None
    values = [song_data[f] for f in available_features]
    labels = [f.replace('_Score', '').replace('_', ' ') for f in available_features]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name=song_name,
        line=dict(color='#1DB954', width=2),
        fillcolor='rgba(29, 185, 84, 0.3)'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        ),
        showlegend=False,
        title={
            'text': f"🎵 Profil Musical: {song_name}",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        height=400
    )
    return fig


# =======================================================================
# --- 6. NOUVELLE FONCTIONNALITÉ : FILTRES AVANCÉS ---
# =======================================================================

def apply_advanced_filters(df_cleaned):
    """Applique des filtres avancés sur les données."""
    st.subheader("🔍 Filtres Avancés")

    filtered_df = df_cleaned.copy()

    col1, col2 = st.columns(2)

    with col1:
        if 'Energy_Score' in df_cleaned.columns:
            energy_range = st.slider(
                "Plage d'Énergie",
                0, 100, (0, 100),
                key='energy_filter'
            )
            filtered_df = filtered_df[
                (filtered_df['Energy_Score'] >= energy_range[0]) &
                (filtered_df['Energy_Score'] <= energy_range[1])
            ]

        if 'Tempo_BPM' in df_cleaned.columns:
            tempo_range = st.slider(
                "Plage de Tempo (BPM)",
                40, 200, (40, 200),
                key='tempo_filter'
            )
            filtered_df = filtered_df[
                (filtered_df['Tempo_BPM'] >= tempo_range[0]) &
                (filtered_df['Tempo_BPM'] <= tempo_range[1])
            ]

    with col2:
        if 'Valence_Score' in df_cleaned.columns:
            valence_range = st.slider(
                "Plage de Positivité (Valence)",
                0, 100, (0, 100),
                key='valence_filter'
            )
            filtered_df = filtered_df[
                (filtered_df['Valence_Score'] >= valence_range[0]) &
                (filtered_df['Valence_Score'] <= valence_range[1])
            ]

        if 'Genre' in df_cleaned.columns:
            genres = ['Tous'] + sorted(df_cleaned['Genre'].unique().tolist())
            selected_genre = st.selectbox(
                "Genre Musical",
                genres,
                key='genre_filter'
            )
            if selected_genre != 'Tous':
                filtered_df = filtered_df[filtered_df['Genre'] == selected_genre]

    st.info(f"📊 Résultats filtrés : {len(filtered_df)} chansons sur {len(df_cleaned)}")

    return filtered_df


# =======================================================================
# --- 7. NOUVELLE FONCTIONNALITÉ : COMPARAISON DE CHANSONS ---
# =======================================================================

def compare_songs(df_cleaned, df_results):
    """Compare 2-3 chansons côte à côte."""
    st.subheader("⚖️ Comparaison de Chansons")

    song_options = df_cleaned.apply(
        lambda row: f"{row.get('Song_Name', f'Musique {row.name}')} - {row.get('Artist', 'Artiste Inconnu')}",
        axis=1
    )

    col1, col2 = st.columns(2)

    with col1:
        song1 = st.selectbox("Première chanson", song_options, key='compare_song1')

    with col2:
        song2 = st.selectbox("Deuxième chanson", song_options, key='compare_song2')

    if st.button("🔍 Comparer", use_container_width=True):
        idx1 = df_cleaned[song_options == song1].index[0]
        idx2 = df_cleaned[song_options == song2].index[0]

        data1 = df_cleaned.iloc[idx1]
        data2 = df_cleaned.iloc[idx2]

        # Créer les radars côte à côte
        col_radar1, col_radar2 = st.columns(2)

        with col_radar1:
            radar1 = plot_radar_chart(data1, data1.get('Song_Name', 'Chanson 1'))
            if radar1:
                st.plotly_chart(radar1, use_container_width=True)

        with col_radar2:
            radar2 = plot_radar_chart(data2, data2.get('Song_Name', 'Chanson 2'))
            if radar2:
                st.plotly_chart(radar2, use_container_width=True)

        # Calcul de similarité
        feature_cols = [col for col in df_results.columns if col.startswith('PC')]
        if len(feature_cols) > 0:
            X = df_results[feature_cols].values
            cosine_sim = cosine_similarity([X[idx1]], [X[idx2]])[0][0]

            st.markdown("---")
            st.metric(
                "Score de Compatibilité",
                f"{cosine_sim * 100:.1f}%",
                help="Basé sur la similarité cosinus dans l'espace PCA"
            )


# =======================================================================
# --- 8. NOUVELLE FONCTIONNALITÉ : HISTORIQUE & FAVORIS ---
# =======================================================================

def initialize_history():
    """Initialise l'historique en session state."""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'favorites' not in st.session_state:
        st.session_state.favorites = []


def add_to_history(song_info):
    """Ajoute une chanson à l'historique."""
    initialize_history()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {**song_info, 'timestamp': timestamp}

    # Éviter les doublons récents
    if not st.session_state.history or st.session_state.history[-1].get('Song_Name') != song_info.get('Song_Name'):
        st.session_state.history.append(entry)

        # Limiter à 50 entrées
        if len(st.session_state.history) > 50:
            st.session_state.history = st.session_state.history[-50:]


def toggle_favorite(song_info):
    """Ajoute/retire une chanson des favoris."""
    initialize_history()
    song_name = song_info.get('Song_Name', '')

    # Vérifier si déjà dans les favoris
    existing = [f for f in st.session_state.favorites if f.get('Song_Name') == song_name]

    if existing:
        st.session_state.favorites = [f for f in st.session_state.favorites if f.get('Song_Name') != song_name]
        return False
    else:
        st.session_state.favorites.append(song_info)
        return True


def show_history_and_favorites():
    """Affiche l'historique et les favoris."""
    st.title("📜 Historique & Favoris")
    st.markdown("---")

    initialize_history()

    tab1, tab2 = st.tabs(["📜 Historique", "⭐ Favoris"])

    with tab1:
        if st.session_state.history:
            st.subheader(f"Dernières {len(st.session_state.history)} chansons consultées")

            # Inverser pour afficher les plus récentes en premier
            for entry in reversed(st.session_state.history):
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"🎵 **{entry.get('Song_Name', 'N/A')}**")
                    st.caption(f"🎤 {entry.get('Artist', 'N/A')}")
                with col2:
                    st.caption(f"🕐 {entry.get('timestamp', 'N/A')}")
                with col3:
                    genre = entry.get('Genre', 'N/A')
                    st.caption(f"🎸 {genre}")
                st.markdown("---")

            if st.button("🗑️ Vider l'historique", key='clear_history'):
                st.session_state.history = []
                st.rerun()
        else:
            st.info("📭 Votre historique est vide. Consultez des chansons pour les voir apparaître ici !")

    with tab2:
        if st.session_state.favorites:
            st.subheader(f"⭐ {len(st.session_state.favorites)} chansons favorites")

            for fav in st.session_state.favorites:
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.write(f"⭐ **{fav.get('Song_Name', 'N/A')}**")
                    st.caption(f"🎤 {fav.get('Artist', 'N/A')}")
                with col2:
                    genre = fav.get('Genre', 'N/A')
                    st.caption(f"🎸 {genre}")
                with col3:
                    if st.button("❌", key=f"remove_fav_{fav.get('Song_Name', '')}"):
                        toggle_favorite(fav)
                        st.rerun()
                st.markdown("---")

            # Export des favoris
            if st.button("💾 Exporter mes favoris (CSV)", use_container_width=True):
                fav_df = pd.DataFrame(st.session_state.favorites)
                csv_data = fav_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Télécharger",
                    data=csv_data,
                    file_name=f"mes_favoris_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )
        else:
            st.info("💫 Vous n'avez pas encore de favoris. Cliquez sur ⭐ dans l'application pour en ajouter !")


# =======================================================================
# --- 9. NOUVELLE FONCTIONNALITÉ : DASHBOARD STATISTIQUES ---
# =======================================================================

def show_statistics_dashboard(df_cleaned):
    """Affiche un dashboard complet de statistiques."""
    st.title("📊 Dashboard Statistiques")
    st.markdown("---")

    # Métriques globales
    st.subheader("📈 Vue d'ensemble")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Chansons", len(df_cleaned))
    with col2:
        if 'Artist' in df_cleaned.columns:
            st.metric("Artistes Uniques", df_cleaned['Artist'].nunique())
    with col3:
        if 'Genre' in df_cleaned.columns:
            st.metric("Genres", df_cleaned['Genre'].nunique())
    with col4:
        if 'Energy_Score' in df_cleaned.columns:
            avg_energy = df_cleaned['Energy_Score'].mean()
            st.metric("Énergie Moyenne", f"{avg_energy:.1f}")

    st.markdown("---")

    # Graphiques
    col_left, col_right = st.columns(2)

    with col_left:
        # Distribution des genres
        if 'Genre' in df_cleaned.columns:
            st.subheader("🎸 Distribution des Genres")
            genre_counts = df_cleaned['Genre'].value_counts().head(10)
            fig_genre = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                title="Top 10 Genres"
            )
            st.plotly_chart(fig_genre, use_container_width=True)

    with col_right:
        # Top artistes
        if 'Artist' in df_cleaned.columns:
            st.subheader("🎤 Top Artistes")
            artist_counts = df_cleaned['Artist'].value_counts().head(10)
            fig_artist = px.bar(
                x=artist_counts.values,
                y=artist_counts.index,
                orientation='h',
                title="Top 10 Artistes",
                labels={'x': 'Nombre de chansons', 'y': 'Artiste'}
            )
            st.plotly_chart(fig_artist, use_container_width=True)

    st.markdown("---")

    # Heatmap Energy vs Danceability
    if all(col in df_cleaned.columns for col in ['Energy_Score', 'Danceability_Score']):
        st.subheader("🔥 Matrice Énergie vs Danceabilité")

        # Créer des bins
        energy_bins = pd.cut(df_cleaned['Energy_Score'], bins=10, labels=False)
        dance_bins = pd.cut(df_cleaned['Danceability_Score'], bins=10, labels=False)

        # Créer la matrice de comptage
        heatmap_data = pd.crosstab(energy_bins, dance_bins)

        fig_heatmap = px.imshow(
            heatmap_data,
            labels=dict(x="Danceability", y="Energy", color="Nombre de chansons"),
            title="Distribution Énergie vs Danceabilité",
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("---")

    # Statistiques des caractéristiques audio
    if any(col in df_cleaned.columns for col in ['Energy_Score', 'Valence_Score', 'Tempo_BPM']):
        st.subheader("🎵 Statistiques des Caractéristiques Audio")

        audio_features = ['Energy_Score', 'Danceability_Score', 'Valence_Score',
                         'Acousticness_Score', 'Instrumentalness_Score']
        available = [col for col in audio_features if col in df_cleaned.columns]

        if available:
            stats_df = df_cleaned[available].describe().T
            stats_df['feature'] = stats_df.index

            fig_violin = go.Figure()
            for feature in available:
                fig_violin.add_trace(go.Violin(
                    y=df_cleaned[feature],
                    name=feature.replace('_Score', ''),
                    box_visible=True,
                    meanline_visible=True
                ))

            fig_violin.update_layout(
                title="Distribution des Caractéristiques Audio",
                yaxis_title="Score (0-100)",
                height=500
            )
            st.plotly_chart(fig_violin, use_container_width=True)


# =======================================================================
# --- 10. NOUVELLE FONCTIONNALITÉ : GÉNÉRATEUR DE MIX PERSONNALISÉ ---
# =======================================================================

def create_custom_mix(df_cleaned):
    """Générateur de mix personnalisé avec critères."""
    st.subheader("🎧 Créer un Mix Personnalisé")
    st.markdown("Définissez vos critères pour générer un mix sur mesure")

    col1, col2 = st.columns(2)

    with col1:
        if 'Energy_Score' in df_cleaned.columns:
            energy_min = st.slider("Énergie Minimale", 0, 100, 50, key='mix_energy')
        else:
            energy_min = 0

        if 'Tempo_BPM' in df_cleaned.columns:
            tempo_min = st.slider("Tempo Minimal (BPM)", 40, 200, 100, key='mix_tempo')
        else:
            tempo_min = 40

    with col2:
        if 'Valence_Score' in df_cleaned.columns:
            valence_min = st.slider("Positivité Minimale", 0, 100, 50, key='mix_valence')
        else:
            valence_min = 0

        n_songs = st.slider("Nombre de chansons", 5, 50, 20, key='mix_count')

    # Nom du mix
    mix_name = st.text_input("Nom de votre mix", "Mon Mix Personnalisé", key='mix_name')

    if st.button("🎵 Générer le Mix", use_container_width=True):
        filtered = df_cleaned.copy()

        if 'Energy_Score' in df_cleaned.columns:
            filtered = filtered[filtered['Energy_Score'] >= energy_min]
        if 'Valence_Score' in df_cleaned.columns:
            filtered = filtered[filtered['Valence_Score'] >= valence_min]
        if 'Tempo_BPM' in df_cleaned.columns:
            filtered = filtered[filtered['Tempo_BPM'] >= tempo_min]

        if len(filtered) > 0:
            # Échantillonner ou prendre les meilleures
            result = filtered.head(n_songs)

            st.success(f"✅ Mix '{mix_name}' créé avec {len(result)} chansons !")

            # Afficher le mix
            display_cols = ['Song_Name', 'Artist']
            if 'Genre' in result.columns:
                display_cols.append('Genre')
            if 'Energy_Score' in result.columns:
                display_cols.append('Energy_Score')
            if 'Tempo_BPM' in result.columns:
                display_cols.append('Tempo_BPM')

            st.dataframe(result[display_cols], use_container_width=True, hide_index=True)

            # Export
            csv_data = result[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"💾 Télécharger '{mix_name}'",
                data=csv_data,
                file_name=f"{mix_name.replace(' ', '_')}.csv",
                mime='text/csv',
                use_container_width=True
            )
        else:
            st.warning("⚠️ Aucune chanson ne correspond à ces critères. Essayez des valeurs moins restrictives.")


# =======================================================================
# --- 11. PLOTS (3D et 2D) ---
# =======================================================================

def plot_3d_pca(df, pca_model):
    """Affiche le graphique 3D interactif avec Plotly"""
    fig = px.scatter_3d(
        df,
        x='PC1', y='PC2', z='PC3', color='Cluster',
        title='🌌 Espace Musical Réduit (PCA-3D) & Clustering',
        labels={
            'PC1': f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%})',
            'PC2': f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%})',
            'PC3': f'PC3 ({pca_model.explained_variance_ratio_[2]:.1%})'
        },
        hover_data=['Song_Name', 'Artist', 'Cluster'] if 'Song_Name' in df.columns else ['Cluster'],
        size_max=10,
        opacity=0.8
    )
    fig.update_layout(height=700, margin=dict(l=0, r=0, b=0, t=50))
    st.plotly_chart(fig, use_container_width=True)


def plot_2d_page_content(df_results, pca_model):
    """Contenu de la page 2D."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmap_clusters = plt.cm.get_cmap('viridis', df_results['Cluster'].nunique())

    # Plot 1: PC1 vs PC2
    scatter1 = axes[0].scatter(df_results['PC1'], df_results['PC2'],
                               c=df_results['Cluster'], cmap=cmap_clusters, alpha=0.7, s=50)
    axes[0].set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('PC1 vs PC2 - Clusters')
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(scatter1, ax=axes[0], label='Cluster')

    # Plot 2: Variance expliquée
    explained_variance = pca_model.explained_variance_ratio_
    axes[1].bar(range(1, len(explained_variance) + 1), explained_variance, color='teal')
    axes[1].set_xlabel('Composantes Principales')
    axes[1].set_ylabel('Variance Expliquée')
    axes[1].set_title('Variance Expliquée par Composante')

    st.pyplot(fig)

    st.subheader("Statistiques PCA")
    st.metric("Variance Totale Expliquée par PC1-PC3", f"{pca_model.explained_variance_ratio_[:3].sum():.1%}")
    st.metric("Nombre de Clusters", df_results['Cluster'].nunique())


# =======================================================================
# --- 12. PAGES DE L'APPLICATION ---
# =======================================================================

def application_page(analysis, df_results):
    """Page principale de recommandation avec GRAPHIQUE RADAR."""
    st.title("🎶 Application de Recommandation IA")
    st.markdown("---")

    st.subheader("🔎 Recherche et Recommandation")

    song_options = df_results.apply(
        lambda row: f"{row.get('Song_Name', f'Musique {row.name}')} - {row.get('Artist', 'Artiste Inconnu')}",
        axis=1
    )

    selected_song_display = st.selectbox(
        "**Sélectionnez une chanson de base ou commencez à taper :**",
        options=song_options
    )

    selected_song_index = df_results[song_options == selected_song_display].index[0]
    selected_row = df_results.iloc[selected_song_index]

    st.markdown("---")

    # Affichage en 2 colonnes
    col_info, col_radar = st.columns([1, 1])

    with col_info:
        st.subheader("💿 Chanson Actuelle")
        st.markdown(f"**Titre :** {selected_row.get('Song_Name', 'N/A')}")
        st.markdown(f"**Artiste :** {selected_row.get('Artist', 'N/A')}")
        st.markdown(f"**Genre :** {selected_row.get('Genre', 'N/A')}")
        st.markdown(f"**Cluster :** {selected_row['Cluster']}")

        cluster_info = df_results[df_results['Cluster'] == selected_row['Cluster']]
        st.metric(label="Taille du Cluster", value=cluster_info.shape[0])
        st.metric("Score de Silhouette", f"{analysis.silhouette_score:.3f}")

        # Bouton favori
        song_info = {
            'Song_Name': selected_row.get('Song_Name', 'N/A'),
            'Artist': selected_row.get('Artist', 'N/A'),
            'Genre': selected_row.get('Genre', 'N/A')
        }

        is_fav = any(f.get('Song_Name') == song_info.get('Song_Name') for f in st.session_state.get('favorites', []))
        fav_label = "💔 Retirer des favoris" if is_fav else "⭐ Ajouter aux favoris"

        if st.button(fav_label, use_container_width=True):
            toggle_favorite(song_info)
            st.rerun()

        # Ajouter à l'historique
        add_to_history(song_info)

    with col_radar:
        st.subheader("📊 Profil Musical")
        original_data = analysis.data_original.iloc[selected_song_index]
        radar_fig = plot_radar_chart(original_data, selected_row.get('Song_Name', 'Chanson'))
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)

    st.markdown("---")

    n_reco = st.slider("Nombre de recommandations à afficher", 1, 10, 5, key='n_reco_app')

    if st.button("✨ Lancer la Recommandation", use_container_width=True):
        recommendations = get_recommendations(df_results, selected_song_index, n_reco)
        st.success(f"Top {n_reco} Recommandations trouvées :")
        st.dataframe(recommendations, use_container_width=True, hide_index=True)


def playlists_page():
    """Page : Playlists Thématiques"""
    st.title("🎵 Playlists Thématiques Automatiques")
    st.markdown("Découvrez des playlists créées automatiquement selon vos goûts musicaux !")
    st.markdown("---")

    if st.session_state.data_cleaned is None:
        st.warning("⚠️ Veuillez d'abord charger et nettoyer vos données dans la page 'Pipeline de Données'.")
        return

    with st.spinner("🎼 Création des playlists en cours..."):
        playlists = create_themed_playlists(st.session_state.data_cleaned)

    if not playlists:
        st.error("❌ Impossible de créer des playlists. Vérifiez que vos données contiennent les colonnes nécessaires.")
        return

    st.success(f"✅ {len(playlists)} playlist(s) créée(s) !")

    selected_playlist = st.selectbox(
        "🎧 Choisissez une playlist :",
        options=list(playlists.keys())
    )

    if selected_playlist:
        playlist_df = playlists[selected_playlist]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de chansons", len(playlist_df))
        with col2:
            if 'Energy_Score' in playlist_df.columns:
                avg_energy = playlist_df['Energy_Score'].mean()
                st.metric("Énergie Moyenne", f"{avg_energy:.1f}")
            else:
                st.metric("Énergie Moyenne", "N/A")
        with col3:
            if 'Tempo_BPM' in playlist_df.columns:
                avg_tempo = playlist_df['Tempo_BPM'].mean()
                st.metric("Tempo Moyen", f"{avg_tempo:.0f} BPM")
            else:
                st.metric("Tempo Moyen", "N/A")

        st.markdown("---")

        st.subheader(f"🎶 Contenu de la playlist : {selected_playlist}")

        display_cols = ['Song_Name', 'Artist']
        if 'Genre' in playlist_df.columns:
            display_cols.append('Genre')
        if 'Energy_Score' in playlist_df.columns:
            display_cols.append('Energy_Score')
        if 'Tempo_BPM' in playlist_df.columns:
            display_cols.append('Tempo_BPM')

        st.dataframe(
            playlist_df[display_cols],
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")
        csv_export = playlist_df[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"⬇️ Télécharger la playlist '{selected_playlist}'",
            data=csv_export,
            file_name=f"playlist_{selected_playlist.replace(' ', '_')}.csv",
            mime='text/csv',
            use_container_width=True
        )


def advanced_features_page():
    """Nouvelle page pour les fonctionnalités avancées."""
    st.title("🚀 Fonctionnalités Avancées")
    st.markdown("---")

    if st.session_state.data_cleaned is None:
        st.warning("⚠️ Veuillez d'abord charger et nettoyer vos données dans la page 'Pipeline de Données'.")
        return

    tab1, tab2, tab3 = st.tabs(["🔍 Filtres Avancés", "⚖️ Comparaison", "🎧 Mix Personnalisé"])

    with tab1:
        filtered_df = apply_advanced_filters(st.session_state.data_cleaned)

        if len(filtered_df) > 0:
            st.subheader("📋 Résultats Filtrés")
            display_cols = ['Song_Name', 'Artist']
            if 'Genre' in filtered_df.columns:
                display_cols.append('Genre')
            if 'Energy_Score' in filtered_df.columns:
                display_cols.append('Energy_Score')
            if 'Tempo_BPM' in filtered_df.columns:
                display_cols.append('Tempo_BPM')

            st.dataframe(filtered_df[display_cols].head(20), use_container_width=True, hide_index=True)

            csv_export = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="💾 Exporter les résultats filtrés",
                data=csv_export,
                file_name="resultats_filtres.csv",
                mime='text/csv',
                use_container_width=True
            )

    with tab2:
        if st.session_state.pca_results:
            analysis = st.session_state.pca_results
            df_results = pd.DataFrame(
                analysis.X_pca,
                columns=[f'PC{i + 1}' for i in range(analysis.pca.n_components_)]
            )
            df_results['Cluster'] = analysis.clusters
            metadata_cols = [col for col in analysis.data_original.columns if col not in analysis.data_numeric.columns]
            for col in metadata_cols:
                df_results[col] = analysis.data_original[col].reset_index(drop=True)

            compare_songs(st.session_state.data_cleaned, df_results)
        else:
            st.warning("⚠️ Veuillez d'abord exécuter l'analyse PCA dans le Pipeline de Données.")

    with tab3:
        create_custom_mix(st.session_state.data_cleaned)


def plot_3d_page(analysis, df_results):
    """Page de visualisation 3D interactive."""
    st.title("🌌 Visualisation 3D de l'Espace Musical PCA")
    st.subheader("Explorez les clusters par similarité de features.")
    st.markdown("---")
    plot_3d_pca(df_results, analysis.pca)


def plot_2d_page(analysis, df_results):
    """Page de visualisation 2D."""
    st.title("📈 Visualisation 2D & Métriques PCA")
    st.subheader("Analyse de la variance et des paires de composantes.")
    st.markdown("---")
    plot_2d_page_content(df_results, analysis.pca)


def pipeline_page():
    """Page dédiée à l'importation et au traitement des données."""
    st.title("⚙️ Pipeline de Données Interactif")
    st.subheader("Importez, nettoyez, pré-traitez et modélisez vos données.")
    st.markdown("---")

    st.header("1. Téléversement du Fichier CSV")
    uploaded_file = st.file_uploader("Sélectionnez un fichier CSV à analyser.", type="csv")

    if uploaded_file is not None:
        load_data_from_upload(uploaded_file)

    if st.session_state.data_original is not None:
        df_original = st.session_state.data_original
        st.info(
            f"Fichier chargé : {uploaded_file.name if uploaded_file else 'N/A'} ({df_original.shape[0]} lignes, {df_original.shape[1]} colonnes)")
        st.dataframe(df_original.head(), use_container_width=True)
        st.markdown("---")

        st.header("2. Nettoyage des Données (MusicDataCleaner)")
        if st.button("🧹 Lancer le Nettoyage", key="btn_clean_pipe"):
            run_cleaning_from_app(st.session_state.data_original)

        if st.session_state.data_cleaned is not None:
            st.header("3. Pré-traitement (MusicPreprocessor)")
            st.dataframe(st.session_state.data_cleaned.head(), use_container_width=True)

            if st.button("🔨 Lancer le Pré-traitement", key="btn_preprocess_pipe"):
                run_preprocessing_from_app(st.session_state.data_cleaned)

            if st.session_state.data_processed is not None:
                st.header("4. Analyse et Modélisation (PCA/KMeans)")

                if st.button("🔬 Lancer l'Analyse PCA & Clustering", key="btn_pca_pipe"):
                    run_pca_from_app(st.session_state.data_processed)

                csv_export = st.session_state.data_processed.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Exporter le fichier de FEATURES (CSV)",
                    data=csv_export,
                    file_name='music_features_processed.csv',
                    mime='text/csv',
                    key="export_features"
                )
    else:
        st.warning("→ Veuillez téléverser votre fichier CSV à l'étape 1 pour commencer le pipeline.")


# =======================================================================
# --- 13. EXÉCUTION PRINCIPALE ---
# =======================================================================

def main():
    st.set_page_config(
        layout="wide",
        page_title="Recommandation Musicale IA",
        page_icon="🎵",
        initial_sidebar_state="expanded"
    )

    # Initialisation des sessions d'état
    for key in ['data_original', 'data_cleaned', 'data_processed', 'pca_results']:
        if key not in st.session_state:
            st.session_state[key] = None

    initialize_history()

    # --- BARRE LATÉRALE DE NAVIGATION (NAVBAR) ---
    st.sidebar.markdown("# 🎧 Menu de Navigation")

    page_selection = st.sidebar.radio(
        "Choisissez votre vue",
        (
            '⚙️ Pipeline de Données',
            '🎶 Application',
            '🎵 Playlists Thématiques',
            '🚀 Fonctionnalités Avancées',
            '📊 Dashboard Statistiques',
            '📜 Historique & Favoris',
            '🌌 Visualisation 3D',
            '📈 Visualisation 2D'
        )
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("🚀 Interface par Streamlit")
    st.sidebar.caption("✨ Nouvelles fonctionnalités :")
    st.sidebar.info(
        "• Filtres avancés\n"
        "• Comparaison de chansons\n"
        "• Mix personnalisés\n"
        "• Dashboard statistiques\n"
        "• Historique & Favoris\n"
        "• Playlists automatiques\n"
        "• Graphiques radar"
    )

    # --- ROUTAGE DES PAGES ---
    if page_selection == '⚙️ Pipeline de Données':
        pipeline_page()

    elif page_selection == '🎵 Playlists Thématiques':
        playlists_page()

    elif page_selection == '🚀 Fonctionnalités Avancées':
        advanced_features_page()

    elif page_selection == '📊 Dashboard Statistiques':
        if st.session_state.data_cleaned is not None:
            show_statistics_dashboard(st.session_state.data_cleaned)
        else:
            st.warning("⚠️ Veuillez d'abord charger et nettoyer vos données dans la page 'Pipeline de Données'.")

    elif page_selection == '📜 Historique & Favoris':
        show_history_and_favorites()

    else:
        if st.session_state.pca_results:
            analysis = st.session_state.pca_results

            df_results = pd.DataFrame(
                analysis.X_pca,
                columns=[f'PC{i + 1}' for i in range(analysis.pca.n_components_)]
            )
            df_results['Cluster'] = analysis.clusters

            metadata_cols_for_display = [
                col for col in analysis.data_original.columns
                if col not in analysis.data_numeric.columns
            ]

            for col in metadata_cols_for_display:
                df_results[col] = analysis.data_original[col].reset_index(drop=True)

            if page_selection == '🎶 Application':
                application_page(analysis, df_results)
            elif page_selection == '🌌 Visualisation 3D':
                plot_3d_page(analysis, df_results)
            elif page_selection == '📈 Visualisation 2D':
                plot_2d_page(analysis, df_results)
        else:
            st.warning("⚠️ Veuillez d'abord exécuter les étapes 1 à 4 dans la page **'⚙️ Pipeline de Données'**.")
            st.info("👉 Allez dans 'Pipeline de Données' pour charger et analyser vos données.")


if __name__ == "__main__":
    main()