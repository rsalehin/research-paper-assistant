"""
Run all tests for all features
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_all_tests():
    """Run all feature tests"""
    
    print('🧪 Running All Tests')
    print('='*60)
    
    # Test Feature 1
    print('\n📦 Feature 1: Data Collection')
    print('-'*60)
    try:
        from tests.test_fetch_arxiv import (
            test_arxiv_data_exists,
            test_arxiv_data_structure,
            test_paper_fields
        )
        test_arxiv_data_exists()
        test_arxiv_data_structure()
        test_paper_fields()
        print('✅ Feature 1: PASSED')
    except Exception as e:
        print(f'❌ Feature 1: FAILED - {e}')
    
    # Test Feature 2
    print('\n🧹 Feature 2: Preprocessing')
    print('-'*60)
    try:
        from tests.test_preprocess import (
            test_processed_data_exists,
            test_processed_data_structure,
            test_processed_fields,
            test_text_cleaning
        )
        test_processed_data_exists()
        test_processed_data_structure()
        test_processed_fields()
        test_text_cleaning()
        print('✅ Feature 2: PASSED')
    except Exception as e:
        print(f'❌ Feature 2: FAILED - {e}')
    
    # Test Feature 3
    print('\n🧠 Feature 3: Embeddings')
    print('-'*60)
    try:
        from tests.test_embeddings import (
            test_embeddings_file_exists,
            test_embeddings_structure,
            test_paper_has_embedding,
            test_numpy_array,
            test_embedding_similarity
        )
        test_embeddings_file_exists()
        test_embeddings_structure()
        test_paper_has_embedding()
        test_numpy_array()
        test_embedding_similarity()
        print('✅ Feature 3: PASSED')
    except Exception as e:
        print(f'❌ Feature 3: FAILED - {e}')
    
    print('\n' + '='*60)
    print('✅ All Tests Complete!')
    print('='*60)

if __name__ == '__main__':
    run_all_tests()
