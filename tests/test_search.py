"""
Test hybrid search functionality
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.elastic_search.hybrid_search import get_elasticsearch_client, hybrid_search

def test_hybrid_search():
    print('Testing Hybrid Search...')
    print('='*50)
    
    # Connect
    es = get_elasticsearch_client()
    print('✅ Connected to Elasticsearch')
    
    # Test search
    query = "transformer models"
    results = hybrid_search(es, query, top_k=3)
    
    print(f'\n✅ Search completed')
    print(f'✅ Found {len(results)} results')
    
    # Check results structure
    if results:
        paper_id, data = results[0]
        assert 'rrf_score' in data
        assert 'source' in data
        assert 'title' in data['source']
        print('✅ Results have correct structure')
    
    print('\n🎉 All tests passed!')

if __name__ == '__main__':
    test_hybrid_search()
