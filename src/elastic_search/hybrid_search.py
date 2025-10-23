"""
Feature 4: Hybrid Search Implementation
Combines BM25 keyword search with kNN vector search
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import numpy as np

# For generating query embeddings
import google.generativeai as genai

# Load environment variables
load_dotenv('config/.env')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Index configuration
INDEX_NAME = "arxiv_papers"


def get_elasticsearch_client():
    """
    Create and return Elasticsearch client.
    
    Returns:
        Elasticsearch: Connected ES client
    """
    
    url = os.getenv('ELASTIC_URL')
    api_key = os.getenv('ELASTIC_API_KEY')
    
    if url and api_key:
        es = Elasticsearch(
            url,
            api_key=api_key,
            request_timeout=30
        )
    else:
        cloud_id = os.getenv('ELASTIC_CLOUD_ID')
        es = Elasticsearch(
            cloud_id=cloud_id,
            api_key=api_key,
            request_timeout=30
        )
    
    return es


def initialize_gemini():
    """
    Initialize Gemini API for generating query embeddings.
    """
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("⚠️  GEMINI_API_KEY not found. Using Vertex AI instead...")
        return False
    
    genai.configure(api_key=api_key)
    return True


def generate_query_embedding(query_text):
    """
    Generate embedding for search query.
    
    Args:
        query_text (str): Search query
    
    Returns:
        list: Query embedding vector
    """
    
    # Try Gemini first
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=query_text,
            task_type="retrieval_query"  # Different task type for queries!
        )
        return result['embedding']
    except:
        pass
    
    # Fallback to Vertex AI
    try:
        from vertexai.language_models import TextEmbeddingModel
        from google.cloud import aiplatform
        
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
        aiplatform.init(project=project_id, location='us-central1')
        
        model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        embeddings = model.get_embeddings([query_text])
        return embeddings[0].values
    except Exception as e:
        print(f"❌ Error generating query embedding: {e}")
        return None


def keyword_search(es, query_text, size=10, index_name=INDEX_NAME):
    """
    Perform BM25 keyword search.
    
    Args:
        es: Elasticsearch client
        query_text (str): Search query
        size (int): Number of results
        index_name (str): Index name
    
    Returns:
        dict: Search results
    """
    
    search_body = {
        "size": size,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "title^3",  # Title is 3x more important
                    "abstract^2",  # Abstract is 2x more important
                    "title_abstract_combined"
                ],
                "type": "best_fields"
            }
        }
    }
    
    return es.search(index=index_name, body=search_body)


def vector_search(es, query_embedding, size=10, index_name=INDEX_NAME):
    """
    Perform kNN vector search.
    
    Args:
        es: Elasticsearch client
        query_embedding (list): Query embedding vector
        size (int): Number of results
        index_name (str): Index name
    
    Returns:
        dict: Search results
    """
    
    search_body = {
        "size": size,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_embedding}
                }
            }
        }
    }
    
    return es.search(index=index_name, body=search_body)


def reciprocal_rank_fusion(keyword_results, vector_results, k=60):
    """
    Combine keyword and vector search results using RRF.
    
    Args:
        keyword_results (dict): BM25 search results
        vector_results (dict): kNN search results
        k (int): RRF constant (default 60)
    
    Returns:
        list: Fused and ranked results
    """
    
    # Extract paper IDs and scores
    keyword_docs = {}
    for rank, hit in enumerate(keyword_results['hits']['hits'], 1):
        paper_id = hit['_id']
        keyword_docs[paper_id] = {
            'rank': rank,
            'score': hit['_score'],
            'source': hit['_source']
        }
    
    vector_docs = {}
    for rank, hit in enumerate(vector_results['hits']['hits'], 1):
        paper_id = hit['_id']
        vector_docs[paper_id] = {
            'rank': rank,
            'score': hit['_score'],
            'source': hit['_source']
        }
    
    # Calculate RRF scores
    all_paper_ids = set(keyword_docs.keys()) | set(vector_docs.keys())
    rrf_scores = {}
    
    for paper_id in all_paper_ids:
        rrf_score = 0
        
        # Add keyword contribution
        if paper_id in keyword_docs:
            rrf_score += 1 / (k + keyword_docs[paper_id]['rank'])
        
        # Add vector contribution
        if paper_id in vector_docs:
            rrf_score += 1 / (k + vector_docs[paper_id]['rank'])
        
        rrf_scores[paper_id] = {
            'rrf_score': rrf_score,
            'keyword_rank': keyword_docs.get(paper_id, {}).get('rank', None),
            'vector_rank': vector_docs.get(paper_id, {}).get('rank', None),
            'keyword_score': keyword_docs.get(paper_id, {}).get('score', 0),
            'vector_score': vector_docs.get(paper_id, {}).get('score', 0),
            'source': keyword_docs.get(paper_id, {}).get('source') or vector_docs.get(paper_id, {}).get('source')
        }
    
    # Sort by RRF score
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1]['rrf_score'],
        reverse=True
    )
    
    return sorted_results


def hybrid_search(es, query_text, top_k=10, index_name=INDEX_NAME):
    """
    Perform hybrid search combining keyword and vector search.
    
    Args:
        es: Elasticsearch client
        query_text (str): Search query
        top_k (int): Number of results to return
        index_name (str): Index name
    
    Returns:
        list: Top-k search results with RRF scores
    """
    
    print(f"\n🔍 Searching for: '{query_text}'")
    print("="*60)
    
    # Generate query embedding
    print("🧠 Generating query embedding...")
    query_embedding = generate_query_embedding(query_text)
    
    if query_embedding is None:
        print("⚠️  Could not generate embedding. Using keyword search only.")
        keyword_results = keyword_search(es, query_text, size=top_k, index_name=index_name)
        return format_keyword_results(keyword_results)
    
    print("✅ Query embedding generated")
    
    # Perform keyword search
    print("📝 Performing keyword search (BM25)...")
    keyword_results = keyword_search(es, query_text, size=20, index_name=index_name)
    print(f"✅ Found {len(keyword_results['hits']['hits'])} keyword matches")
    
    # Perform vector search
    print("🧠 Performing vector search (kNN)...")
    vector_results = vector_search(es, query_embedding, size=20, index_name=index_name)
    print(f"✅ Found {len(vector_results['hits']['hits'])} semantic matches")
    
    # Fuse results
    print("⚖️  Combining results with RRF...")
    fused_results = reciprocal_rank_fusion(keyword_results, vector_results)
    
    # Return top-k
    top_results = fused_results[:top_k]
    print(f"✅ Returning top {len(top_results)} results")
    
    return top_results


def format_keyword_results(results):
    """Format keyword-only results to match hybrid output."""
    formatted = []
    for rank, hit in enumerate(results['hits']['hits'], 1):
        formatted.append((
            hit['_id'],
            {
                'rrf_score': hit['_score'],
                'keyword_rank': rank,
                'vector_rank': None,
                'keyword_score': hit['_score'],
                'vector_score': 0,
                'source': hit['_source']
            }
        ))
    return formatted


def display_results(results):
    """
    Display search results in a readable format.
    
    Args:
        results (list): Search results from hybrid_search
    """
    
    print("\n" + "="*60)
    print("📊 SEARCH RESULTS")
    print("="*60)
    
    for i, (paper_id, data) in enumerate(results, 1):
        paper = data['source']
        
        print(f"\n{i}. {paper['title']}")
        print(f"   arXiv ID: {paper['arxiv_id']}")
        print(f"   Authors: {', '.join(paper['authors_short'])}")
        print(f"   Year: {paper['year']} | Category: {paper['primary_category']}")
        print(f"   Abstract: {paper['abstract'][:200]}...")
        print(f"\n   📈 Scores:")
        print(f"      RRF Score: {data['rrf_score']:.4f}")
        print(f"      Keyword Rank: {data['keyword_rank'] or 'N/A'} (score: {data['keyword_score']:.2f})")
        print(f"      Vector Rank: {data['vector_rank'] or 'N/A'} (score: {data['vector_score']:.2f})")
        print(f"   🔗 PDF: {paper['pdf_url']}")
        print("-"*60)


def test_searches():
    """
    Run several test searches to demonstrate hybrid search.
    """
    
    print("="*60)
    print("🧪 HYBRID SEARCH TEST SUITE")
    print("="*60)
    
    # Connect to Elasticsearch
    es = get_elasticsearch_client()
    
    # Initialize embedding generation
    initialize_gemini()
    
    # Test queries
    test_queries = [
        "transformer attention mechanisms",
        "federated learning privacy",
        "efficient neural networks",
        "reinforcement learning robotics",
        "image generation diffusion models"
    ]
    
    for query in test_queries:
        results = hybrid_search(es, query, top_k=5)
        display_results(results)
        print("\n" + "="*60 + "\n")
        input("Press Enter to see next query results...")


def interactive_search():
    """
    Interactive search mode.
    """
    
    print("="*60)
    print("🔍 INTERACTIVE HYBRID SEARCH")
    print("="*60)
    print("Type your search queries. Type 'quit' to exit.")
    print()
    
    # Connect to Elasticsearch
    es = get_elasticsearch_client()
    
    # Initialize embedding generation
    initialize_gemini()
    
    while True:
        query = input("\n🔎 Enter search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            results = hybrid_search(es, query, top_k=5)
            display_results(results)
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """
    Main function - choose mode.
    """
    
    print("="*60)
    print("🔍 Hybrid Search System")
    print("="*60)
    print("\nChoose mode:")
    print("1. Interactive search")
    print("2. Run test queries")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        interactive_search()
    elif choice == "2":
        test_searches()
    else:
        print("Invalid choice. Running interactive mode...")
        interactive_search()


if __name__ == "__main__":
    main()