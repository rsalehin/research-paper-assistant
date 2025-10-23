"""
Test for Feature 3: Vector Embeddings
"""
import json
import numpy as np
from pathlib import Path

def test_embeddings_file_exists():
    """Test if embeddings file exists"""
    json_path = Path('data/processed/arxiv_papers_with_embeddings.json')
    npy_path = Path('data/embeddings/paper_embeddings.npy')
    
    assert json_path.exists(), 'Embeddings JSON file not found'
    assert npy_path.exists(), 'Embeddings numpy file not found'
    
    print('✅ Embeddings files exist')

def test_embeddings_structure():
    """Test if embeddings have correct structure"""
    with open('data/processed/arxiv_papers_with_embeddings.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert 'metadata' in data, 'Missing metadata'
    assert 'papers' in data, 'Missing papers'
    assert data['metadata']['embedding_dimension'] == 768, 'Wrong embedding dimension'
    
    print(f'✅ Embeddings structure valid')
    print(f'✅ Embedding dimension: {data["metadata"]["embedding_dimension"]}')

def test_paper_has_embedding():
    """Test if papers have embedding vectors"""
    with open('data/processed/arxiv_papers_with_embeddings.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    paper = data['papers'][0]
    
    assert 'embedding' in paper, 'Paper missing embedding'
    assert len(paper['embedding']) == 768, 'Wrong embedding length'
    assert isinstance(paper['embedding'][0], (int, float)), 'Embedding values not numeric'
    
    print('✅ Papers have valid embeddings')

def test_numpy_array():
    """Test numpy embeddings array"""
    embeddings = np.load('data/embeddings/paper_embeddings.npy')
    
    assert embeddings.shape[1] == 768, 'Wrong embedding dimension in numpy array'
    assert embeddings.dtype == np.float64 or embeddings.dtype == np.float32, 'Wrong data type'
    
    print(f'✅ Numpy array valid: shape {embeddings.shape}')

def test_embedding_similarity():
    """Test if embeddings capture similarity"""
    with open('data/processed/arxiv_papers_with_embeddings.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    papers = data['papers']
    
    if len(papers) >= 2:
        emb1 = np.array(papers[0]['embedding'])
        emb2 = np.array(papers[1]['embedding'])
        
        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Similarity should be between -1 and 1
        assert -1 <= similarity <= 1, 'Similarity out of range'
        
        print(f'✅ Similarity calculation works: {similarity:.4f}')

if __name__ == '__main__':
    print('Testing Feature 3: Vector Embeddings')
    print('='*50)
    test_embeddings_file_exists()
    test_embeddings_structure()
    test_paper_has_embedding()
    test_numpy_array()
    test_embedding_similarity()
    print('='*50)
    print('✅ All Feature 3 tests passed!')
