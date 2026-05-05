# pca_analysis.py (VERSION TURBO-CHARGÉE : PCA sur 5000 lignes pour la performance)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st  # NÉCESSAIRE POUR LE CACHING
import warnings

warnings.filterwarnings('ignore')


# =======================================================================
# --- CLASSE PCAAnalysis3D ---
# =======================================================================

class PCAAnalysis3D:
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
        """Prépare les données pour la PCA."""
        self.data_original = data.copy()
        self.data_numeric = data.select_dtypes(include=[np.number])

        if self.data_numeric.isnull().sum().sum() > 0:
            self.data_numeric = self.data_numeric.fillna(self.data_numeric.median())

        self.X_scaled = self.scaler.fit_transform(self.data_numeric)
        return self.X_scaled

    def perform_pca(self):
        """Effectue l'analyse PCA."""
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        self.cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        return self.X_pca

    def perform_clustering(self, n_clusters=4):
        """Effectue le clustering sur les composantes PCA."""
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
        self.clusters = self.kmeans.fit_predict(self.X_pca)
        if len(np.unique(self.clusters)) > 1:
            self.silhouette_score = silhouette_score(self.X_pca, self.clusters)
        else:
            self.silhouette_score = 0.0
        return self.clusters

    def find_optimal_clusters(self, max_k=8):
        """Trouve le k optimal."""
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
            return possible_k_values[max_silhouette_idx]
        else:
            return 4

    # --- Les méthodes de visualisation sont laissées ici pour le module ---
    def create_3d_plot_interactive(self):
        # ... (Logique de plot 3D inchangée) ...
        if self.clusters is None: self.perform_clustering()
        plot_df = pd.DataFrame(
            {'PC1': self.X_pca[:, 0], 'PC2': self.X_pca[:, 1], 'PC3': self.X_pca[:, 2], 'Cluster': self.clusters,
             'Taille': 20})
        fig = px.scatter_3d(plot_df, x='PC1', y='PC2', z='PC3', color='Cluster',
                            title='🎵 ANALYSE PCA 3D AVEC CLUSTERING', hover_data=None, size='Taille', size_max=15,
                            opacity=0.8)
        return fig

    def create_2d_plots(self):
        # ... (Logique de plot 2D inchangée) ...
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        return fig

    def get_summary_stats(self):
        if self.clusters is None: self.perform_clustering()
        stats = {
            'n_samples': self.X_pca.shape[0], 'variance_explained_total': self.pca.explained_variance_ratio_[:3].sum(),
            'n_clusters': len(np.unique(self.clusters)), 'silhouette_score': self.silhouette_score
        }
        return stats


# =======================================================================
# --- FONCTION CACHÉE DE L'ANALYSE (SOLUTION AU BLOCAGE) ---
# =======================================================================

@st.cache_data(
    show_spinner="🔄 Analyse PCA et Clustering en cours... (Échantillon de 5 000 lignes pour la rapidité - Patientez quelques secondes)")
def run_pca_clustering_cached(df_features_to_analyze):
    """
    Fonction CACHÉE pour exécuter la PCA et le Clustering sur un échantillon (5000 lignes).
    """

    # 1. ÉCHANTILLONNAGE POUR ACCÉLÉRER
    sample_size = 5000

    if df_features_to_analyze.shape[0] > sample_size:
        # Créer un échantillon aléatoire de 5000 lignes
        df_sample = df_features_to_analyze.sample(n=sample_size, random_state=42)
    else:
        # Utiliser l'ensemble des données si elles sont déjà petites
        df_sample = df_features_to_analyze.copy()

    # 2. Exécution de l'analyse sur l'ÉCHANTILLON
    analysis = PCAAnalysis3D(n_components=3)
    analysis.load_and_preprocess_data(data=df_sample)
    analysis.perform_pca()

    # 3. Clustering
    optimal_k = analysis.find_optimal_clusters(max_k=8)
    analysis.perform_clustering(n_clusters=optimal_k)

    return analysis