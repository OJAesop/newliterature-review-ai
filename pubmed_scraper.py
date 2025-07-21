# pubmed_scraper.py
import requests
import xml.etree.ElementTree as ET
import json
import time
from typing import List, Dict

class PubMedScraper:
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def search_papers(self, query: str, max_results: int = 100) -> List[str]:
        """Search PubMed and return list of PMIDs"""
        search_url = f"{self.base_url}esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml'
        }
        
        response = requests.get(search_url, params=params)
        root = ET.fromstring(response.content)
        
        pmids = []
        for id_elem in root.findall('.//Id'):
            pmids.append(id_elem.text)
        
        return pmids
    
    def fetch_paper_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed information for papers"""
        papers = []
        
        # Process in batches to respect API limits
        batch_size = 20
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            
            fetch_url = f"{self.base_url}efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(batch),
                'retmode': 'xml'
            }
            
            response = requests.get(fetch_url, params=params)
            root = ET.fromstring(response.content)
            
            for article in root.findall('.//PubmedArticle'):
                paper = self.extract_paper_info(article)
                if paper:
                    papers.append(paper)
            
            # Be respectful to API
            time.sleep(0.5)
        
        return papers
    
    def extract_paper_info(self, article) -> Dict:
        """Extract relevant information from XML"""
        try:
            title_elem = article.find('.//ArticleTitle')
            abstract_elem = article.find('.//AbstractText')
            
            title = title_elem.text if title_elem is not None else ""
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in article.findall('.//Author'):
                last_name = author.find('LastName')
                first_name = author.find('ForeName')
                if last_name is not None and first_name is not None:
                    authors.append(f"{first_name.text} {last_name.text}")
            
            # Extract journal and date
            journal_elem = article.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            year_elem = article.find('.//PubDate/Year')
            year = year_elem.text if year_elem is not None else ""
            
            return {
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'journal': journal,
                'year': year,
                'pmid': article.find('.//PMID').text
            }
        except:
            return None

# Usage example
scraper = PubMedScraper()
pmids = scraper.search_papers("Alzheimer's disease treatment", max_results=50)
papers = scraper.fetch_paper_details(pmids)
