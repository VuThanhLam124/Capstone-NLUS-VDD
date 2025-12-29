"""
RAG Retriever with BGE-M3 Multilingual Embedding for Text-to-SQL
Uses BAAI/bge-m3 for semantic similarity search (supports Vietnamese).
Hybrid retrieval: Semantic + BM25 for better accuracy.
"""
import numpy as np
import pickle
import re
from pathlib import Path
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("WARNING: sentence-transformers not installed. Run: pip install sentence-transformers")

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("INFO: rank_bm25 not installed. Using semantic-only retrieval.")

# Config - Using BGE-M3 for multilingual support (Vietnamese)
EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_TOP_K = 10
HYBRID_ALPHA = 0.7  # 70% semantic, 30% BM25


class TextToSQLRetriever:
    """
    Hybrid retriever using BGE-M3 (multilingual) + BM25.
    Optimized for Vietnamese Text-to-SQL.
    """
    
    def __init__(self, data_path: str = None, model_name: str = EMBEDDING_MODEL, use_hybrid: bool = True):
        self.data_path = Path(data_path) if data_path else None
        self.model_name = model_name
        self.model = None
        self.questions = []
        self.sqls = []
        self.embeddings = None
        self.use_hybrid = use_hybrid and HAS_BM25
        self.bm25 = None
        self.tokenized_questions = None
        
        if data_path:
            self._load_data()
    
    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer for Vietnamese/English."""
        return re.findall(r'\w+', text.lower())
        
    def _load_data(self):
        if not HAS_SBERT:
            raise RuntimeError("sentence-transformers not installed")
            
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        
        # Load model - BGE-M3 supports multilingual including Vietnamese
        print(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        
        # Load data
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["Transcription", "SQL Ground Truth"])
        
        self.questions = df["Transcription"].tolist()
        self.sqls = df["SQL Ground Truth"].tolist()
        
        # Encode all questions with BGE-M3
        print(f"Encoding {len(self.questions)} questions with BGE-M3...")
        self.embeddings = self.model.encode(
            self.questions,  # BGE-M3 doesn't need instruction prefix
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        print("Semantic encoding complete!")
        
        # Build BM25 index for hybrid retrieval
        if self.use_hybrid:
            print("Building BM25 index for hybrid retrieval...")
            self.tokenized_questions = [self._tokenize(q) for q in self.questions]
            self.bm25 = BM25Okapi(self.tokenized_questions)
            print("BM25 index ready!")
        
    def retrieve(self, query: str, k: int = DEFAULT_TOP_K, alpha: float = HYBRID_ALPHA) -> list[dict]:
        """
        Hybrid retrieval: alpha * semantic + (1-alpha) * BM25.
        
        Args:
            query: Vietnamese/English question
            k: Number of results
            alpha: Weight for semantic score (default 0.7)
        
        Returns:
            List of {score, semantic_score, bm25_score, question, sql}
        """
        if self.embeddings is None or self.model is None:
            return []
        
        # Semantic scores with BGE-M3
        query_emb = self.model.encode([query], normalize_embeddings=True)
        semantic_scores = np.dot(self.embeddings, query_emb.T).flatten()
        
        # Hybrid with BM25
        if self.use_hybrid and self.bm25 is not None:
            query_tokens = self._tokenize(query)
            bm25_scores = np.array(self.bm25.get_scores(query_tokens))
            
            # Normalize BM25 scores to [0, 1]
            bm25_min, bm25_max = bm25_scores.min(), bm25_scores.max()
            if bm25_max > bm25_min:
                bm25_scores = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
            else:
                bm25_scores = np.zeros_like(bm25_scores)
            
            # Combine scores
            final_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
        else:
            bm25_scores = np.zeros_like(semantic_scores)
            final_scores = semantic_scores
        
        # Get top-k indices
        top_indices = np.argsort(final_scores)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": float(final_scores[idx]),
                "semantic_score": float(semantic_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "question": self.questions[idx],
                "sql": self.sqls[idx]
            })
        return results
    
    def save(self, save_dir: str):
        """Save embeddings, BM25 index and data."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            "model_name": self.model_name,
            "questions": self.questions,
            "sqls": self.sqls,
            "embeddings": self.embeddings,
            "use_hybrid": self.use_hybrid,
            "tokenized_questions": self.tokenized_questions,
        }
        
        with open(save_dir / "retriever_bge_m3.pkl", "wb") as f:
            pickle.dump(save_data, f)
        print(f"Retriever saved to {save_dir}/retriever_bge_m3.pkl")
        print(f"  - Model: {self.model_name}")
        print(f"  - Samples: {len(self.questions)}")
        print(f"  - Hybrid: {self.use_hybrid}")

    @classmethod
    def load(cls, load_dir: str):
        """Load retriever from disk."""
        if not HAS_SBERT:
            raise RuntimeError("sentence-transformers not installed")
            
        instance = cls.__new__(cls)
        load_dir = Path(load_dir)
        
        # Try BGE-M3 version first, then old BGE, then TF-IDF
        bge_m3_path = load_dir / "retriever_bge_m3.pkl"
        bge_path = load_dir / "retriever_bge.pkl"
        tfidf_path = load_dir / "retriever.pkl"
        
        if bge_m3_path.exists():
            print(f"Loading BGE-M3 retriever from {bge_m3_path}...")
            with open(bge_m3_path, "rb") as f:
                data = pickle.load(f)
            instance.model_name = data["model_name"]
            instance.questions = data["questions"]
            instance.sqls = data["sqls"]
            instance.embeddings = data["embeddings"]
            instance.use_hybrid = data.get("use_hybrid", False)
            instance.tokenized_questions = data.get("tokenized_questions")
            
            print(f"Loading model: {instance.model_name}...")
            instance.model = SentenceTransformer(instance.model_name)
            
            # Rebuild BM25 if hybrid
            if instance.use_hybrid and instance.tokenized_questions and HAS_BM25:
                print("Rebuilding BM25 index...")
                instance.bm25 = BM25Okapi(instance.tokenized_questions)
            else:
                instance.bm25 = None
                instance.use_hybrid = False
                
        elif bge_path.exists():
            # Old BGE format - migrate to new
            print(f"Loading old BGE retriever from {bge_path}...")
            with open(bge_path, "rb") as f:
                data = pickle.load(f)
            instance.model_name = data["model_name"]
            instance.questions = data["questions"]
            instance.sqls = data["sqls"]
            instance.embeddings = data["embeddings"]
            instance.use_hybrid = False
            instance.bm25 = None
            instance.tokenized_questions = None
            print(f"Loading model: {instance.model_name}...")
            instance.model = SentenceTransformer(instance.model_name)
            print("WARNING: Old format loaded. Consider rebuilding with: python rag_retriever.py")
            
        elif tfidf_path.exists():
            # Fallback to TF-IDF (very old format)
            print("WARNING: Loading old TF-IDF format. Rebuild with: python rag_retriever.py")
            with open(tfidf_path, "rb") as f:
                data = pickle.load(f)
            instance.model_name = EMBEDDING_MODEL
            instance.model = None
            instance.questions = data["questions"]
            instance.sqls = data["sqls"]
            instance.embeddings = None
            instance.use_hybrid = False
            instance.bm25 = None
            instance.tokenized_questions = None
        else:
            raise FileNotFoundError(f"No retriever found in {load_dir}")
        
        print(f"Loaded {len(instance.questions)} examples, hybrid={instance.use_hybrid}")
        return instance


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build RAG index with BGE-M3 + BM25 hybrid")
    parser.add_argument("--data", default="research_pipeline/datasets/train_clean.csv", 
                        help="Path to training data CSV")
    parser.add_argument("--output", default="research_pipeline/rag_index",
                        help="Output directory for index")
    parser.add_argument("--no-hybrid", action="store_true", 
                        help="Disable BM25 hybrid (semantic only)")
    args = parser.parse_args()
    
    print("="*50)
    print("Building RAG Index with BGE-M3 (Multilingual)")
    print("="*50)
    
    # Build index
    retriever = TextToSQLRetriever(
        args.data, 
        model_name=EMBEDDING_MODEL,
        use_hybrid=not args.no_hybrid
    )
    retriever.save(args.output)
    
    # Test retrieval
    test_queries = [
        "Tính tổng doanh thu năm 2000",
        "Top 10 sản phẩm bán chạy nhất",
        "Khách hàng ở California mua nhiều nhất",
    ]
    
    print("\n" + "="*50)
    print("Testing Retrieval")
    print("="*50)
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retriever.retrieve(q, k=3)
        for i, res in enumerate(results):
            print(f"  {i+1}. [score={res['score']:.3f}, sem={res['semantic_score']:.3f}, bm25={res['bm25_score']:.3f}]")
            print(f"     Q: {res['question'][:60]}...")
    
    print("\nDone! Index saved to:", args.output)
