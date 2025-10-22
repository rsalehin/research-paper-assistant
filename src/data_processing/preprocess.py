"""
Feature 2: Data Preprocessing
Cleans and prepares arXiv papers for indexing and search
"""
"""
cs.LG - Machine Learning
cs.AI - AI
cs.CL - Computation and Language
cs.CV - Computer Vision 
stat.ML - Statistics machine learning
"""

import json
import re
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm


# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text (str): Raw text to clean
    
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove newlines and extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    return text


def extract_year(date_string):
    """
    Extract year from date string.
    
    Args:
        date_string (str): Date in format 'YYYY-MM-DD'
    
    Returns:
        int: Year
    """
    try:
        return int(date_string.split('-')[0])
    except:
        return None


def calculate_abstract_stats(abstract):
    """
    Calculate statistics about the abstract.
    
    Args:
        abstract (str): Paper abstract
    
    Returns:
        dict: Statistics (word_count, char_count)
    """
    words = abstract.split()
    
    return {
        'word_count': len(words),
        'char_count': len(abstract)
    }


def extract_features(paper):
    """
    Extract additional features from paper data.
    
    Args:
        paper (dict): Raw paper data
    
    Returns:
        dict: Paper with additional features
    """
    
    # Clean title and abstract
    paper['title'] = clean_text(paper.get('title', ''))
    paper['abstract'] = clean_text(paper.get('abstract', ''))
    
    # Combine title and abstract for search
    paper['title_abstract_combined'] = f"{paper['title']}. {paper['abstract']}"
    
    # Extract year
    paper['year'] = extract_year(paper.get('published', ''))
    
    # Calculate abstract statistics
    stats = calculate_abstract_stats(paper['abstract'])
    paper['abstract_word_count'] = stats['word_count']
    paper['abstract_char_count'] = stats['char_count']
    
    # Count authors
    paper['num_authors'] = len(paper.get('authors', []))
    
    # Create a shorter author list (first 3 authors + "et al" if more)
    authors = paper.get('authors', [])
    if len(authors) > 3:
        paper['authors_short'] = authors[:3] + ['et al.']
    else:
        paper['authors_short'] = authors
    
    # Clean comment and journal_ref
    paper['comment'] = clean_text(paper.get('comment', ''))
    paper['journal_ref'] = clean_text(paper.get('journal_ref', ''))
    
    return paper


def validate_paper(paper):
    """
    Validate that paper has all required fields.
    
    Args:
        paper (dict): Paper data
    
    Returns:
        tuple: (is_valid, reason)
    """
    
    # Required fields
    required_fields = ['arxiv_id', 'title', 'abstract', 'authors']
    
    for field in required_fields:
        if field not in paper or not paper[field]:
            return False, f"Missing {field}"
    
    # Check minimum abstract length (at least 50 words)
    if paper.get('abstract_word_count', 0) < 50:
        return False, "Abstract too short"
    
    # Check title length (at least 3 words)
    if len(paper.get('title', '').split()) < 3:
        return False, "Title too short"
    
    return True, "Valid"


def process_papers(input_file="arxiv_papers.json", output_file="arxiv_papers_processed.json"):
    """
    Process all papers from raw to cleaned format.
    
    Args:
        input_file (str): Input filename in raw data directory
        output_file (str): Output filename in processed data directory
    
    Returns:
        dict: Processing statistics
    """
    
    print("="*60)
    print(" Starting Data Preprocessing")
    print("="*60)
    
    # Load raw data
    input_path = RAW_DATA_DIR / input_file
    
    print(f"\n Loading data from: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f" Error: File not found: {input_path}")
        print(" Make sure you ran Feature 1 first!")
        return None
    
    papers = raw_data.get('papers', [])
    print(f" Loaded {len(papers)} papers")
    
    # Process papers
    processed_papers = []
    skipped_papers = []
    
    print("\n Processing papers...")
    for paper in tqdm(papers, desc="Progress"):
        try:
            # Extract features
            processed_paper = extract_features(paper)
            
            # Validate
            is_valid, reason = validate_paper(processed_paper)
            
            if is_valid:
                processed_papers.append(processed_paper)
            else:
                skipped_papers.append({
                    'arxiv_id': paper.get('arxiv_id', 'unknown'),
                    'reason': reason
                })
        
        except Exception as e:
            skipped_papers.append({
                'arxiv_id': paper.get('arxiv_id', 'unknown'),
                'reason': f"Error: {str(e)}"
            })
    
    # Create output data package
    output_data = {
        'metadata': {
            'total_papers': len(processed_papers),
            'skipped_papers': len(skipped_papers),
            'process_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source_file': input_file,
            'categories': list(set(cat for paper in processed_papers for cat in paper.get('categories', [])))
        },
        'papers': processed_papers
    }
    
    # Save processed data
    output_path = PROCESSED_DATA_DIR / output_file
    
    print(f"\n Saving processed data to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Display statistics
    display_statistics(processed_papers, skipped_papers)
    
    # Save skipped papers log if any
    if skipped_papers:
        skipped_path = PROCESSED_DATA_DIR / "skipped_papers.json"
        with open(skipped_path, 'w', encoding='utf-8') as f:
            json.dump(skipped_papers, f, indent=2)
        print(f"\n  Skipped papers log saved to: {skipped_path}")
    
    print("\n" + "="*60)
    print(" Feature 2 Complete!")
    print(f" {len(processed_papers)} papers successfully processed")
    print(f" Data saved to: {output_path}")
    print("="*60)
    
    return {
        'total_processed': len(processed_papers),
        'total_skipped': len(skipped_papers),
        'output_path': output_path
    }


def display_statistics(papers, skipped_papers):
    """
    Display processing statistics.
    
    Args:
        papers (list): Processed papers
        skipped_papers (list): Skipped papers with reasons
    """
    
    if not papers:
        print("\n  No papers were processed!")
        return
    
    df = pd.DataFrame(papers)
    
    print("\n Processing Statistics:")
    print("="*60)
    print(f" Successfully processed: {len(papers)}")
    print(f"  Skipped: {len(skipped_papers)}")
    print(f" Success rate: {len(papers)/(len(papers)+len(skipped_papers))*100:.1f}%")
    
    print("\n Content Statistics:")
    print(f"   Average abstract length: {df['abstract_word_count'].mean():.0f} words")
    print(f"   Average title length: {df['title'].str.split().str.len().mean():.1f} words")
    print(f"   Average authors per paper: {df['num_authors'].mean():.1f}")
    
    print("\n Year Distribution:")
    year_counts = df['year'].value_counts().sort_index(ascending=False).head()
    for year, count in year_counts.items():
        print(f"   {year}: {count} papers")
    
    print("\n  Top Categories:")
    all_categories = [cat for cats in df['categories'] for cat in cats]
    category_counts = pd.Series(all_categories).value_counts().head(5)
    for cat, count in category_counts.items():
        print(f"   {cat}: {count} papers")
    
    if skipped_papers:
        print("\n  Skipped Papers Reasons:")
        skip_reasons = pd.Series([s['reason'] for s in skipped_papers]).value_counts()
        for reason, count in skip_reasons.items():
            print(f"   {reason}: {count} papers")


def display_sample(processed_file="arxiv_papers_processed.json", n=3):
    """
    Display a sample of processed papers.
    
    Args:
        processed_file (str): Processed data filename
        n (int): Number of papers to display
    """
    
    file_path = PROCESSED_DATA_DIR / processed_file
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    papers = data['papers']
    
    print("\n Sample Processed Papers:")
    print("="*60)
    
    for i, paper in enumerate(papers[:n], 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   ID: {paper['arxiv_id']}")
        print(f"   Authors: {', '.join(paper['authors_short'])}")
        print(f"   Year: {paper['year']}")
        print(f"   Categories: {', '.join(paper['categories'][:3])}")
        print(f"   Abstract: {paper['abstract'][:150]}...")
        print(f"   Stats: {paper['abstract_word_count']} words, {paper['num_authors']} authors")


def main():
    """
    Main function to run preprocessing.
    """
    
    # Process papers
    result = process_papers()
    
    if result:
        # Display sample
        display_sample(n=3)


if __name__ == "__main__":
    main()