"""
Feature 4: Elasticsearch Index Setup (Serverless Compatible)
Creates index with hybrid search capabilities (BM25 + kNN)
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from tqdm import tqdm

# Load environment variables
load_dotenv('config/.env')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Index configuration
INDEX_NAME = "arxiv_papers"


def get_elasticsearch_client():
    """
    Create and return Elasticsearch client.
    
    Returns:
        Elasticsearch: Connected ES client
    """
    
    # Try URL method first (since it works for you)
    url = os.getenv('ELASTIC_URL')
    api_key = os.getenv('ELASTIC_API_KEY')
    
    if url and api_key:
        es = Elasticsearch(
            url,
            api_key=api_key,
            request_timeout=30
        )
    else:
        # Fallback to Cloud ID
        cloud_id = os.getenv('ELASTIC_CLOUD_ID')
        es = Elasticsearch(
            cloud_id=cloud_id,
            api_key=api_key,
            request_timeout=30
        )
    
    return es


def create_index_mapping():
    """
    Define the index mapping for hybrid search (serverless compatible).
    
    Returns:
        dict: Index mapping configuration
    """
    
    # Simplified mapping for serverless (no shard/replica settings)
    mapping = {
        "mappings": {
            "properties": {
                # Text fields for keyword search
                "title": {
                    "type": "text",
                    "analyzer": "english"
                },
                "abstract": {
                    "type": "text",
                    "analyzer": "english"
                },
                "title_abstract_combined": {
                    "type": "text",
                    "analyzer": "english"
                },
                
                # Vector field for semantic search
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine"
                },
                
                # Metadata fields
                "arxiv_id": {
                    "type": "keyword"
                },
                "authors": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    }
                },
                "authors_short": {
                    "type": "keyword"
                },
                "published": {
                    "type": "date"
                },
                "year": {
                    "type": "integer"
                },
                "categories": {
                    "type": "keyword"
                },
                "primary_category": {
                    "type": "keyword"
                },
                "pdf_url": {
                    "type": "keyword"
                },
                "abstract_word_count": {
                    "type": "integer"
                },
                "num_authors": {
                    "type": "integer"
                }
            }
        }
    }
    
    return mapping


def create_index(es, index_name=INDEX_NAME, delete_if_exists=False):
    """
    Create Elasticsearch index.
    
    Args:
        es: Elasticsearch client
        index_name (str): Name of the index
        delete_if_exists (bool): Whether to delete existing index
    
    Returns:
        bool: True if successful
    """
    
    print(f"\n📊 Creating index: {index_name}")
    
    # Check if index exists
    if es.indices.exists(index=index_name):
        if delete_if_exists:
            print(f"⚠️  Index '{index_name}' already exists. Deleting...")
            es.indices.delete(index=index_name)
        else:
            print(f"✅ Index '{index_name}' already exists.")
            return True
    
    # Create index with mapping
    mapping = create_index_mapping()
    
    try:
        es.indices.create(index=index_name, body=mapping)
        print(f"✅ Index '{index_name}' created successfully!")
        return True
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        # If index already exists, that's okay
        if "resource_already_exists_exception" in str(e):
            print(f"✅ Index '{index_name}' already exists (continuing...)")
            return True
        return False


def prepare_document(paper):
    """
    Prepare paper document for indexing.
    
    Args:
        paper (dict): Paper data with embedding
    
    Returns:
        dict: Document ready for Elasticsearch
    """
    
    # Create a copy to avoid modifying original
    doc = {
        "arxiv_id": paper.get("arxiv_id"),
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "title_abstract_combined": paper.get("title_abstract_combined"),
        "embedding": paper.get("embedding"),
        "authors": paper.get("authors", []),
        "authors_short": paper.get("authors_short", []),
        "published": paper.get("published"),
        "year": paper.get("year"),
        "categories": paper.get("categories", []),
        "primary_category": paper.get("primary_category"),
        "pdf_url": paper.get("pdf_url"),
        "abstract_word_count": paper.get("abstract_word_count"),
        "num_authors": paper.get("num_authors")
    }
    
    return doc


def index_papers(es, papers, index_name=INDEX_NAME, batch_size=50):
    """
    Index papers into Elasticsearch using bulk API.
    
    Args:
        es: Elasticsearch client
        papers (list): List of papers with embeddings
        index_name (str): Name of the index
        batch_size (int): Number of documents per batch
    
    Returns:
        dict: Indexing statistics
    """
    
    print(f"\n📤 Indexing {len(papers)} papers...")
    
    from elasticsearch.helpers import bulk
    
    # Prepare documents for bulk indexing
    actions = []
    
    for paper in tqdm(papers, desc="Preparing documents"):
        doc = prepare_document(paper)
        
        action = {
            "_index": index_name,
            "_id": paper["arxiv_id"],
            "_source": doc
        }
        
        actions.append(action)
    
    # Bulk index
    print(f"\n⏳ Uploading to Elasticsearch...")
    
    try:
        success, failed = bulk(es, actions, chunk_size=batch_size, raise_on_error=False)
        
        print(f"✅ Successfully indexed: {success}")
        if failed:
            print(f"❌ Failed: {len(failed)}")
        
        return {
            "success": success,
            "failed": len(failed) if failed else 0
        }
    
    except Exception as e:
        print(f"❌ Error during bulk indexing: {e}")
        return {"success": 0, "failed": len(papers)}


def verify_index(es, index_name=INDEX_NAME):
    """
    Verify that index was created correctly (serverless compatible).
    
    Args:
        es: Elasticsearch client
        index_name (str): Name of the index
    """
    
    print(f"\n🔍 Verifying index: {index_name}")
    
    # Get document count using count API
    try:
        count_result = es.count(index=index_name)
        doc_count = count_result["count"]
        print(f"✅ Total documents: {doc_count}")
    except Exception as e:
        print(f"⚠️  Could not get document count: {e}")
    
    # Get a sample document
    try:
        result = es.search(
            index=index_name,
            body={
                "size": 1,
                "query": {"match_all": {}}
            }
        )
        
        if result["hits"]["hits"]:
            sample = result["hits"]["hits"][0]["_source"]
            print(f"✅ Sample document ID: {sample['arxiv_id']}")
            print(f"✅ Sample title: {sample['title'][:60]}...")
            print(f"✅ Has embedding: {'embedding' in sample}")
            if 'embedding' in sample:
                print(f"✅ Embedding dimension: {len(sample['embedding'])}")
    except Exception as e:
        print(f"⚠️  Could not retrieve sample: {e}")


def main():
    """
    Main function to set up Elasticsearch index.
    """
    
    print("="*60)
    print("🔍 Elasticsearch Index Setup")
    print("="*60)
    
    # Connect to Elasticsearch
    print("\n🔌 Connecting to Elasticsearch...")
    es = get_elasticsearch_client()
    
    # Test connection
    info = es.info()
    print(f"✅ Connected to cluster: {info['cluster_name']}")
    print(f"✅ Elasticsearch version: {info['version']['number']}")
    
    # Create index
    create_index(es, delete_if_exists=False)
    
    # Load papers with embeddings
    papers_path = PROCESSED_DATA_DIR / "arxiv_papers_with_embeddings.json"
    
    print(f"\n📖 Loading papers from: {papers_path}")
    
    try:
        with open(papers_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {papers_path}")
        print("💡 Make sure you ran Feature 3 first!")
        return
    
    papers = data.get('papers', [])
    print(f"✅ Loaded {len(papers)} papers")
    
    # Index papers
    result = index_papers(es, papers)
    
    # Verify
    verify_index(es)
    
    print("\n" + "="*60)
    print("✅ Feature 4 Setup Complete!")
    print(f"✅ {result['success']} papers indexed successfully")
    print(f"✅ Index name: {INDEX_NAME}")
    print("="*60)
    
    print("\n💡 Next: Run hybrid_search.py to test searching!")


if __name__ == "__main__":
    main()