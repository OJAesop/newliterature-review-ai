# vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict

class SimpleVectorStore:
    def __init__(self, collection_name: str = "research_papers"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)
        
        # Free sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_papers(self, papers: List[Dict]):
        """Add papers to vector database"""
        documents = []
        metadatas = []
        ids = []
        
        for paper in papers:
            # Combine title and abstract for embedding
            text = f"{paper['title']} {paper['abstract']}"
            documents.append(text)
            
            metadatas.append({
                'title': paper['title'],
                'authors': json.dumps(paper['authors']),
                'journal': paper['journal'],
                'year': paper['year'],
                'pmid': paper['pmid']
            })
            
            ids.append(f"pmid_{paper['pmid']}")
        
        # Generate embeddings
        embeddings = self.model.encode(documents).tolist()
        
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_papers(self, query: str, n_results: int = 10) -> List[Dict]:
        """Search for relevant papers"""
        query_embedding = self.model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        return results

# Usage
vector_store = SimpleVectorStore()
vector_store.add_papers(papers)
results = vector_store.search_papers("tau protein aggregation")
