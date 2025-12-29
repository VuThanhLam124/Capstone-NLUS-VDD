import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextToSQLRetriever:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
        self.questions = []
        self.sqls = []
        self.vectors = None
        self._load_data()
        
    def _load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        df = df.dropna(subset=["Transcription", "SQL Ground Truth"])
        
        self.questions = df["Transcription"].tolist()
        self.sqls = df["SQL Ground Truth"].tolist()
        
        print(f"Fitting TF-IDF on {len(self.questions)} samples...")
        self.vectors = self.vectorizer.fit_transform(self.questions)
        
    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Retrieve top-k similar examples."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.vectors).flatten()
        
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
        """Save vectorizer and data index."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_dir / "retriever.pkl", "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "questions": self.questions,
                "sqls": self.sqls,
                "vectors": self.vectors
            }, f)
        print(f"Retriever saved to {save_dir}")

    @classmethod
    def load(cls, load_dir: str):
        """Load retriever from disk."""
        instance = cls.__new__(cls)
        load_dir = Path(load_dir)
        
        with open(load_dir / "retriever.pkl", "rb") as f:
            data = pickle.load(f)
            
        instance.vectorizer = data["vectorizer"]
        instance.questions = data["questions"]
        instance.sqls = data["sqls"]
        instance.vectors = data["vectors"]
        return instance

if __name__ == "__main__":
    # Test build
    retriever = TextToSQLRetriever("research_pipeline/datasets/train_merged.csv")
    retriever.save("research_pipeline/rag_index")
    
    # Test retrieve
    q = "Tính tổng doanh thu năm 2000"
    results = retriever.retrieve(q)
    print(f"\nQuery: {q}")
    for i, res in enumerate(results):
        print(f"{i+1}. [{res['score']:.4f}] {res['question']}")
