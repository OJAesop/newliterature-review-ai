# Literature Review AI - Week 1-2: Data Pipeline Implementation

import requests
import pandas as pd
from typing import List, Dict, Optional
import time
import json
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

class LiteraturePipeline:
    """
    Free data pipeline for academic literature collection
    Uses: PubMed, arXiv, Semantic Scholar APIs (all free)
    """
    
    def __init__(self):
        self.pubmed_base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.arxiv_base = "http://export.arxiv.org/api/query"
        self.semantic_scholar_base = "https://api.semanticscholar.org/graph/v1/"
        self.papers = []
        
    def search_pubmed(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search PubMed for papers - completely free"""
        try:
            # Step 1: Search for paper IDs
            search_url = f"{self.pubmed_base}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json'
            }
            
            response = requests.get(search_url, params=search_params)
            search_data = response.json()
            
            if 'esearchresult' not in search_data:
                return []
                
            ids = search_data['esearchresult'].get('idlist', [])
            
            if not ids:
                return []
            
            # Step 2: Fetch paper details
            fetch_url = f"{self.pubmed_base}efetch.fcgi"
            fetch_params = {
                'db': 'pubmed',
                'id': ','.join(ids),
                'retmode': 'xml'
            }
            
            response = requests.get(fetch_url, params=fetch_params)
            
            # Parse XML response
            root = ET.fromstring(response.content)
            papers = []
            
            for article in root.findall('.//PubmedArticle'):
                paper = self._parse_pubmed_article(article)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"PubMed search error: {e}")
            return []
    
    def search_arxiv(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search arXiv for papers - completely free"""
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(self.arxiv_base, params=params)
            
            # Parse XML response
            root = ET.fromstring(response.content)
            papers = []
            
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper = self._parse_arxiv_entry(entry)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []
    
    def search_semantic_scholar(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search Semantic Scholar - free with rate limits"""
        try:
            url = f"{self.semantic_scholar_base}paper/search"
            params = {
                'query': query,
                'limit': min(max_results, 100),  # API limit
                'fields': 'title,abstract,authors,year,citationCount,url,venue'
            }
            
            # Add delay to respect rate limits
            time.sleep(1)
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"Semantic Scholar API error: {response.status_code}")
                return []
            
            data = response.json()
            papers = []
            
            for paper_data in data.get('data', []):
                paper = self._parse_semantic_scholar_paper(paper_data)
                if paper:
                    papers.append(paper)
            
            return papers
            
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []
    
    def _parse_pubmed_article(self, article) -> Optional[Dict]:
        """Parse PubMed XML article"""
        try:
            title_elem = article.find('.//ArticleTitle')
            abstract_elem = article.find('.//AbstractText')
            
            title = title_elem.text if title_elem is not None else ""
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in article.findall('.//Author'):
                first_name = author.find('.//ForeName')
                last_name = author.find('.//LastName')
                if first_name is not None and last_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
            
            # Extract publication year
            year_elem = article.find('.//PubDate/Year')
            year = year_elem.text if year_elem is not None else ""
            
            # Extract journal
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            return {
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'year': year,
                'journal': journal,
                'source': 'PubMed',
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.find('.//PMID').text}/" if article.find('.//PMID') is not None else ""
            }
            
        except Exception as e:
            print(f"Error parsing PubMed article: {e}")
            return None
    
    def _parse_arxiv_entry(self, entry) -> Optional[Dict]:
        """Parse arXiv XML entry"""
        try:
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            title = entry.find('atom:title', ns).text.strip()
            summary = entry.find('atom:summary', ns).text.strip()
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', ns):
                name = author.find('atom:name', ns)
                if name is not None:
                    authors.append(name.text)
            
            # Extract publication date
            published = entry.find('atom:published', ns).text
            year = published.split('-')[0] if published else ""
            
            # Extract arXiv ID and URL
            arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
            url = f"https://arxiv.org/abs/{arxiv_id}"
            
            return {
                'title': title,
                'abstract': summary,
                'authors': authors,
                'year': year,
                'journal': 'arXiv',
                'source': 'arXiv',
                'url': url,
                'arxiv_id': arxiv_id
            }
            
        except Exception as e:
            print(f"Error parsing arXiv entry: {e}")
            return None
    
    def _parse_semantic_scholar_paper(self, paper_data) -> Optional[Dict]:
        """Parse Semantic Scholar paper data"""
        try:
            authors = [author.get('name', '') for author in paper_data.get('authors', [])]
            
            return {
                'title': paper_data.get('title', ''),
                'abstract': paper_data.get('abstract', ''),
                'authors': authors,
                'year': str(paper_data.get('year', '')),
                'journal': paper_data.get('venue', ''),
                'source': 'Semantic Scholar',
                'url': paper_data.get('url', ''),
                'citation_count': paper_data.get('citationCount', 0)
            }
            
        except Exception as e:
            print(f"Error parsing Semantic Scholar paper: {e}")
            return None
    
    def comprehensive_search(self, query: str, max_per_source: int = 50) -> pd.DataFrame:
        """Search all sources and combine results"""
        print(f"Searching for: '{query}'")
        all_papers = []
        
        # Search PubMed
        print("Searching PubMed...")
        pubmed_papers = self.search_pubmed(query, max_per_source)
        all_papers.extend(pubmed_papers)
        print(f"Found {len(pubmed_papers)} papers from PubMed")
        
        # Search arXiv
        print("Searching arXiv...")
        arxiv_papers = self.search_arxiv(query, max_per_source)
        all_papers.extend(arxiv_papers)
        print(f"Found {len(arxiv_papers)} papers from arXiv")
        
        # Search Semantic Scholar
        print("Searching Semantic Scholar...")
        ss_papers = self.search_semantic_scholar(query, max_per_source)
        all_papers.extend(ss_papers)
        print(f"Found {len(ss_papers)} papers from Semantic Scholar")
        
        # Convert to DataFrame and remove duplicates
        df = pd.DataFrame(all_papers)
        
        if not df.empty:
            # Remove duplicate papers based on title similarity
            df = df.drop_duplicates(subset=['title'], keep='first')
            print(f"Total unique papers: {len(df)}")
        
        return df
    
    def save_results(self, df: pd.DataFrame, filename: str = None):
        """Save results to CSV and JSON"""
        if filename is None:
            filename = f"literature_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save as CSV
        df.to_csv(f"{filename}.csv", index=False)
        
        # Save as JSON for easier programmatic access
        df.to_json(f"{filename}.json", orient='records', indent=2)
        
        print(f"Results saved as {filename}.csv and {filename}.json")

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = LiteraturePipeline()
    
    # Example search
    query = "machine learning healthcare"
    results_df = pipeline.comprehensive_search(query, max_per_source=20)
    
    # Display sample results
    if not results_df.empty:
        print(f"\nSample of {len(results_df)} papers found:")
        for idx, paper in results_df.head().iterrows():
            print(f"\nTitle: {paper['title']}")
            print(f"Authors: {', '.join(paper['authors']) if paper['authors'] else 'N/A'}")
            print(f"Year: {paper['year']}")
            print(f"Source: {paper['source']}")
            print(f"Abstract: {paper['abstract'][:200]}...")
    
    # Save results
    pipeline.save_results(results_df)
