"""
RAG Retriever with BGE Embedding for Text-to-SQL
Uses BAAI/bge-large-en-v1.5 for semantic similarity search.
"""
import numpy as np
import pickle
from pathlib import Path
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("WARNING: sentence-transformers not installed. Run: pip install sentence-transformers")

# Config
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_TOP_K = 10

class TextToSQLRetriever:
    def __init__(self, data_path: str = None, model_name: str = EMBEDDING_MODEL):
        self.data_path = Path(data_path) if data_path else None
        self.model_name = model_name
        self.model = None
        self.questions = []
        self.sqls = []
        self.embeddings = None
        
        if data_path:
            self._load_data()
        
    def _load_data(self):
        if not HAS_SBERT:
            raise RuntimeError("sentence-transformers not installed")
            
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        
        # Load model
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        # Load data
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["Transcription", "SQL Ground Truth"])
        
        self.questions = df["Transcription"].tolist()
        self.sqls = df["SQL Ground Truth"].tolist()
        
        # Encode all questions
        print(f"Encoding {len(self.questions)} questions...")
        # BGE models recommend adding instruction prefix for retrieval
        texts_with_instruction = [f"Represent this sentence for searching relevant SQL examples: {q}" for q in self.questions]
        self.embeddings = self.model.encode(texts_with_instruction, show_progress_bar=True, normalize_embeddings=True)
        print("Encoding complete!")
        
    def retrieve(self, query: str, k: int = DEFAULT_TOP_K) -> list[dict]:
        """Retrieve top-k similar examples."""
        if self.embeddings is None or self.model is None:
            return []
        
        # Encode query with instruction
        query_with_instruction = f"Represent this sentence for searching relevant SQL examples: {query}"
        query_emb = self.model.encode([query_with_instruction], normalize_embeddings=True)
        
        # Cosine similarity (embeddings are normalized)
        scores = np.dot(self.embeddings, query_emb.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": float(scores[idx]),
                "question": self.questions[idx],
                "sql": self.sqls[idx]
            })
        return results
    
    def save(self, save_dir: str):
        """Save embeddings and data index."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / "retriever_bge.pkl", "wb") as f:
            pickle.dump({
                "model_name": self.model_name,
                "questions": self.questions,
                "sqls": self.sqls,
                "embeddings": self.embeddings
            }, f)
        print(f"Retriever saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str):
        """Load retriever from disk."""
        if not HAS_SBERT:
            raise RuntimeError("sentence-transformers not installed")
            
        instance = cls.__new__(cls)
        load_dir = Path(load_dir)
        
        # Try BGE version first, fallback to TF-IDF version
        bge_path = load_dir / "retriever_bge.pkl"
        tfidf_path = load_dir / "retriever.pkl"
        
        if bge_path.exists():
            with open(bge_path, "rb") as f:
                data = pickle.load(f)
            instance.model_name = data["model_name"]
            instance.questions = data["questions"]
            instance.sqls = data["sqls"]
            instance.embeddings = data["embeddings"]
            print(f"Loading model: {instance.model_name}...")
            instance.model = SentenceTransformer(instance.model_name)
        elif tfidf_path.exists():
            # Fallback to TF-IDF (old format)
            print("WARNING: Loading old TF-IDF format. Consider rebuilding with BGE.")
            with open(tfidf_path, "rb") as f:
                data = pickle.load(f)
            instance.model_name = None
            instance.model = None
            instance.questions = data["questions"]
            instance.sqls = data["sqls"]
            instance.embeddings = None
            # Use TF-IDF for retrieval
            instance._tfidf_vectorizer = data["vectorizer"]
            instance._tfidf_vectors = data["vectors"]
        else:
            raise FileNotFoundError(f"No retriever found in {load_dir}")
            
        return instance

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="research_pipeline/datasets/train_merged.csv")
    parser.add_argument("--output", default="research_pipeline/rag_index")
    args = parser.parse_args()
    
    # Build index
    retriever = TextToSQLRetriever(args.data)
    retriever.save(args.output)
    
    # Test retrieve
    q = "Tính tổng doanh thu năm 2000"
    results = retriever.retrieve(q, k=5)
    print(f"\nQuery: {q}")
    for i, res in enumerate(results):
        print(f"{i+1}. [{res['score']:.4f}] {res['question'][:60]}...")
