import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

class LightweightRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2", n_clusters=3):
        self.model       = SentenceTransformer(model_name)
        self.n_clusters  = n_clusters
        self.documents   = []
        self.embeddings  = None  
        self.centroids   = None  
        self.assignments = None   
        self.adj_matrix  = None  

    # Math Utilities
    def _l2_norm(self, v):
        return np.sqrt(np.sum(v ** 2))

    def _normalize(self, v):
        norm = self._l2_norm(v)
        return v / norm if norm > 0 else v

    def _cosine_similarity(self, a, b):
        norm_a = self._l2_norm(a)
        norm_b = self._l2_norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.sum(a * b) / (norm_a * norm_b)

    # K-Means Clustering
    def _kmeans(self, vectors, max_iters=100, tol=1e-4):
        n, d = vectors.shape
        k    = self.n_clusters

        np.random.seed(42)
        idx         = np.random.choice(n, k, replace=False)
        centroids   = vectors[idx].copy()
        assignments = np.zeros(n, dtype=int)

        for iteration in range(max_iters):
            distances = np.zeros((n, k))
            for j, centroid in enumerate(centroids):
                diff = vectors - centroid
                distances[:, j] = np.sqrt(np.sum(diff ** 2, axis=1))
            new_assignments = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                members = vectors[new_assignments == j]
                new_centroids[j] = (
                    np.mean(members, axis=0)
                    if len(members) > 0
                    else centroids[j]
                )

            shift       = np.sqrt(np.sum((new_centroids - centroids) ** 2))
            assignments = new_assignments
            centroids   = new_centroids

            if shift < tol:
                break

        return centroids, assignments

    # Graph Adjacency Builder
    def _build_adjacency(self, threshold=0.5):
        n = len(self.embeddings)
        adj = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(self.embeddings[i], self.embeddings[j])
                if sim >= threshold:
                    adj[i, j] = sim
                    adj[j, i] = sim
        return adj

    # Indexing
    def index(self, documents, adj_threshold=0.5):
        self.documents = documents
        raw_embeddings = self.model.encode(documents)
        
        # Normalize embeddings
        self.embeddings = np.array([self._normalize(emb) for emb in raw_embeddings])
        
        # K-Means
        self.centroids, self.assignments = self._kmeans(self.embeddings)
        
        # Adjacency Matrix (GraphRAG)
        self.adj_matrix = self._build_adjacency(threshold=adj_threshold)
        print(f"✅ {len(documents)} belge başarıyla indekslendi ve kümelendi!")

    # Search (IVF destekli)
    search_query = None # placeholder
    def search(self, query, top_k=2, use_ivf=True):
        q_emb = self._normalize(self.model.encode(query))
        
        if use_ivf and self.centroids is not None:
            # En yakın kümeyi bul
            cluster_dists = [np.sqrt(np.sum((q_emb - c) ** 2)) for c in self.centroids]
            best_cluster = np.argmin(cluster_dists)
            candidate_indices = np.where(self.assignments == best_cluster)[0]
            if len(candidate_indices) == 0:
                candidate_indices = range(len(self.documents))
        else:
            candidate_indices = range(len(self.documents))

        scores = []
        for idx in candidate_indices:
            score = self._cosine_similarity(q_emb, self.embeddings[idx])
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in scores[:top_k]]

    # Graph Search (GraphRAG Hop Traversal)
    def graph_search(self, query, top_k=2, hops=1):
        base_results = self.search(query, top_k=top_k, use_ivf=False)
        if not base_results:
            return []

        # En iyi belgenin indexini bul
        best_doc, best_score = base_results[0]
        start_idx = self.documents.index(best_doc)

        visited = {start_idx}
        current_frontier = {start_idx}

        for _ in range(hops):
            next_frontier = set()
            for idx in current_frontier:
                neighbors = np.where(self.adj_matrix[idx] > 0)[0]
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        next_frontier.add(n)
            current_frontier = next_frontier

        # Komşuları ve ana sonucu skorla
        expanded_indices = list(visited)
        q_emb = self._normalize(self.model.encode(query))
        scores = [(idx, self._cosine_similarity(q_emb, self.embeddings[idx])) for idx in expanded_indices]
        scores.sort(key=lambda x: x[1], reverse=True)

        return [(self.documents[idx], score) for idx, score in scores[:top_k]]

    # Visualisation (PCA)
    def visualise(self, query=None):
        # Manuel PCA 
        centered = self.embeddings - np.mean(self.embeddings, axis=0)
        cov = np.cov(centered, rowvar=False)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        idx = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:, idx]
        projection_matrix = eigenvecs[:, :2]
        
        projected = np.dot(centered, projection_matrix)

        plt.figure(figsize=(8, 6))
        plt.scatter(projected[:, 0], projected[:, 1], c=self.assignments, cmap='viridis', s=100, edgecolor='k')
        
        if self.centroids is not None:
            centered_centroids = self.centroids - np.mean(self.embeddings, axis=0)
            proj_centroids = np.dot(centered_centroids, projection_matrix)
            plt.scatter(proj_centroids[:, 0], proj_centroids[:, 1], marker='X', s=200, c='red', label='Centroids')

        plt.title("LightweightRAG - 2D PCA Vector Space & K-Means Clusters")
        plt.legend()
        plt.grid(True)
        plt.show()