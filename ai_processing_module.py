# Literature Review AI - Week 3-4: AI Processing Module
# Uses completely free models: Hugging Face Transformers, sentence-transformers

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json
import pickle
from pathlib import Path
import re

# Free ML libraries
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import chromadb
from chromadb.config import Settings

class LiteratureAI:
    """
    AI processing for literature review using completely free models
    - Sentence transformers for embeddings (free)
    - BART/T5 for summarization (free)
    - ChromaDB for vector storage (free)
    """
    
    def __init__(self, models_cache_dir: str = "./models"):
        self.models_cache_dir = Path(models_cache_dir)
        self.models_cache_dir.mkdir(exist_ok=True)
        
        # Initialize free models
        print("Loading free AI models...")
        self._load_models()
        
        # Initialize vector database (free)
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="literature_papers",
            metadata={"description": "Academic papers for literature review"}
        )
    
    def _load_models(self):
        """Load all free models"""
        try:
            # 1. Sentence transformer for embeddings (free, ~500MB)
            print("Loading sentence transformer...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
            
            # 2. Summarization model (free, ~1.2GB)
            print("Loading summarization model...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",  # Good for academic text
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            
            # 3. Question answering model (free, ~400MB)
            print("Loading QA model...")
            self.qa_model = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Make sure you have internet connection for first-time download")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts using free sentence transformer"""
        try:
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return np.array([])
    
    def summarize_papers(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize abstracts using free BART model"""
        summaries = []
        
        print("Generating summaries...")
        for idx, row in papers_df.iterrows():
            try:
                abstract = row['abstract']
                
                # Skip if no abstract
                if not abstract or len(abstract.strip()) < 100:
                    summaries.append("No abstract available")
                    continue
                
                # BART works best with 500-1000 tokens
                if len(abstract) > 1000:
                    abstract = abstract[:1000] + "..."
                
                # Generate summary
                summary = self.summarizer(
                    abstract,
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )[0]['summary_text']
                
                summaries.append(summary)
                
            except Exception as e:
                print(f"Error summarizing paper {idx}: {e}")
                summaries.append("Summary generation failed")
        
        papers_df['ai_summary'] = summaries
        return papers_df
    
    def extract_key_concepts(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """Extract key concepts using NER and keyword extraction"""
        try:
            # Use NER pipeline (free)
            ner = pipeline("ner", 
                          model="dbmdz/bert-large-cased-finetuned-conll03-english",
                          aggregation_strategy="simple")
            
            all_concepts = []
            
            for idx, row in papers_df.iterrows():
                text = f"{row['title']} {row['abstract']}"
                
                # Extract named entities
                entities = ner(text)
                concepts = [entity['word'] for entity in entities 
                           if entity['score'] > 0.9]  # High confidence only
                
                # Simple keyword extraction (free)
                keywords = self._extract_keywords(text)
                concepts.extend(keywords)
                
                # Remove duplicates and clean
                concepts = list(set([c.strip() for c in concepts if len(c.strip()) > 2]))
                all_concepts.append(concepts[:10])  # Top 10 concepts
            
            papers_df['key_concepts'] = all_concepts
            return papers_df
            
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            papers_df['key_concepts'] = [[] for _ in range(len(papers_df))]
            return papers_df
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Simple keyword extraction using frequency and importance"""
        # Remove common academic words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'we', 'our', 'study', 'research',
            'paper', 'article', 'results', 'conclusion', 'method', 'approach'
        }
        
        # Extract potential keywords (2-4 words, capitalize)
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        words = [w for w in words if w not in stop_words]
        
        # Simple frequency-based extraction
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [kw[0] for kw in keywords[:max_keywords]]
    
    def store_embeddings(self, papers_df: pd.DataFrame):
        """Store paper embeddings in ChromaDB (free vector database)"""
        try:
            # Generate embeddings for titles + abstracts
            texts = []
            metadatas = []
            ids = []
            
            for idx, row in papers_df.iterrows():
                text = f"{row['title']} {row['abstract']}"
                texts.append(text)
                
                metadata = {
                    'title': row['title'],
                    'authors': ', '.join(row['authors']) if row['authors'] else '',
                    'year': str(row['year']),
                    'source': row['source'],
                    'journal': row['journal'],
                    'url': row['url']
                }
                metadatas.append(metadata)
                ids.append(f"paper_{idx}")
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Stored {len(texts)} papers in vector database")
            
        except Exception as e:
            print(f"Error storing embeddings: {e}")
    
    def semantic_search(self, query: str, n_results: int = 10) -> List[Dict]:
        """Semantic search using embeddings"""
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'title': results['metadatas'][0][i]['title'],
                    'authors': results['metadatas'][0][i]['authors'],
                    'year': results['metadatas'][0][i]['year'],
                    'source': results['metadatas'][0][i]['source'],
                    'url': results['metadatas'][0][i]['url'],
                    'similarity_score': results['distances'][0][i],
                    'document_excerpt': results['documents'][0][i][:300] + "..."
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def answer_question(self, question: str, context_papers: List[Dict] = None) -> Dict:
        """Answer questions about the literature using QA model"""
        try:
            if not context_papers:
                # Search for relevant papers first
                context_papers = self.semantic_search(question, n_results=5)
            
            # Combine contexts
            context = " ".join([paper['document_excerpt'] for paper in context_papers])
            
            # Limit context length for the model
            if len(context) > 4000:
                context = context[:4000] + "..."
            
            # Generate answer
            answer = self.qa_model(question=question, context=context)
            
            return {
                'question': question,
                'answer': answer['answer'],
                'confidence': answer['score'],
                'context_papers': [p['title'] for p in context_papers],
                'sources': [p['url'] for p in context_papers if p['url']]
            }
            
        except Exception as e:
            print(f"Error answering question: {e}")
            return {
                'question': question,
                'answer': "Could not generate answer",
                'confidence': 0.0,
                'context_papers': [],
                'sources': []
            }
    
    def find_similar_papers(self, paper_title: str, n_results: int = 5) -> List[Dict]:
        """Find papers similar to a given paper"""
        return self.semantic_search(paper_title, n_results)
    
    def cluster_papers(self, papers_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """Cluster papers by topic using embeddings"""
        try:
            # Generate embeddings
            texts = [f"{row['title']} {row['abstract']}" for _, row in papers_df.iterrows()]
            embeddings = self.generate_embeddings(texts)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            papers_df['cluster'] = clusters
            
            # Generate cluster summaries
            cluster_summaries = {}
            for cluster_id in range(n_clusters):
                cluster_papers = papers_df[papers_df['cluster'] == cluster_id]
                cluster_titles = ' '.join(cluster_papers['title'].tolist())
                
                # Simple cluster naming based on common words
                common_words = self._extract_keywords(cluster_titles, max_keywords=3)
                cluster_summaries[cluster_id] = f"Cluster {cluster_id}: {', '.join(common_words)}"
            
            papers_df['cluster_summary'] = papers_df['cluster'].map(cluster_summaries)
            
            return papers_df
            
        except Exception as e:
            print(f"Error clustering papers: {e}")
            papers_df['cluster'] = 0
            papers_df['cluster_summary'] = "Clustering failed"
            return papers_df
    
    def generate_literature_summary(self, papers_df: pd.DataFrame) -> Dict:
        """Generate overall literature summary"""
        try:
            total_papers = len(papers_df)
            
            # Extract date range
            years = papers_df['year'].dropna().astype(str)
            years = [y for y in years if y.isdigit() and 1900 <= int(y) <= 2024]
            year_range = f"{min(years)}-{max(years)}" if years else "Unknown"
            
            # Top sources
            top_sources = papers_df['source'].value_counts().head().to_dict()
            
            # Most cited papers (if available)
            if 'citation_count' in papers_df.columns:
                top_cited = papers_df.nlargest(5, 'citation_count')[['title', 'citation_count']].to_dict('records')
            else:
                top_cited = []
            
            # Common concepts
            all_concepts = []
            if 'key_concepts' in papers_df.columns:
                for concepts in papers_df['key_concepts']:
                    all_concepts.extend(concepts)
            
            concept_freq = {}
            for concept in all_concepts:
                concept_freq[concept] = concept_freq.get(concept, 0) + 1
            
            top_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'total_papers': total_papers,
                'year_range': year_range,
                'top_sources': top_sources,
                'top_cited_papers': top_cited,
                'top_concepts': top_concepts,
                'summary_generated': True
            }
            
        except Exception as e:
            print(f"Error generating literature summary: {e}")
            return {'error': str(e), 'summary_generated': False}

# Example usage combining data pipeline + AI processing
if __name__ == "__main__":
    # Initialize AI processor
    ai = LiteratureAI()
    
    # Load previously collected papers (from data pipeline)
    try:
        papers_df = pd.read_json("literature_search_20241201_120000.json")  # Adjust filename
        print(f"Loaded {len(papers_df)} papers")
    except FileNotFoundError:
        print("No previous search results found. Run the data pipeline first.")
        exit()
    
    # Process papers with AI
    print("Processing papers with AI...")
    
    # 1. Generate summaries
    papers_df = ai.summarize_papers(papers_df.head(10))  # Start with 10 papers
    
    # 2. Extract key concepts
    papers_df = ai.extract_key_concepts(papers_df)
    
    # 3. Store embeddings for semantic search
    ai.store_embeddings(papers_df)
    
    # 4. Cluster papers by topic
    papers_df = ai.cluster_papers(papers_df)
    
    # 5. Generate overall summary
    lit_summary = ai.generate_literature_summary(papers_df)
    print("\nLiterature Summary:")
    print(json.dumps(lit_summary, indent=2))
    
    # 6. Example semantic search
    print("\nExample semantic search:")
    search_results = ai.semantic_search("machine learning applications", n_results=3)
    for i, result in enumerate(search_results):
        print(f"\n{i+1}. {result['title']}")
        print(f"   Authors: {result['authors']}")
        print(f"   Similarity: {result['similarity_score']:.3f}")
    
    # 7. Example question answering
    print("\nExample question answering:")
    qa_result = ai.answer_question("What are the main applications of machine learning in healthcare?")
    print(f"Q: {qa_result['question']}")
    print(f"A: {qa_result['answer']}")
    print(f"Confidence: {qa_result['confidence']:.3f}")
    
    # Save processed results
    papers_df.to_json("processed_literature.json", orient='records', indent=2)
    print("\nProcessed results saved to processed_literature.json")
