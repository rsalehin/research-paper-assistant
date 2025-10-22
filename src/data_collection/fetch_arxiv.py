import arxiv
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# We have manually already created the data folder. Here we
# are doing it again. So, if we hadn't created the folder
# this line would have created it with the intermediate
# parent folders (data). The second argument specifies
# that if the folders already exist, don't raise FileExistsError
def fetch_arxiv_papers(max_results=100, categories = None):
    """
    Fetch papers from arxiv based on categories. 
    
    Args:
        max_results (int): Maximum number of papers to fetch
        categories (list): List of arxiv categories (e.g., ['cs.AI', 'cs.LG'])
    
    Returns:
    list: list of paper dictionaries
    """
    if categories is None:
        categories = ["cs.AI", "cs.LG", "cs.CL"]
    #Build query
    query = " OR ".join([f"cat:{cat}" for cat in categories])
    
    print(f"Searching arxiv for papers in: {', '.join(categories)}")
    print(f"Fetching up to {max_results} papers...")
    
    # Create search object
    search = arxiv.Search(
        query = query,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.SubmittedDate, 
        sort_order = arxiv.SortOrder.Descending
    )
    
    # Fetch papers with tqdm download bar
    
    print("\n Downloading papers...")
    papers = []
    for result in tqdm(search.results(), total=max_results, desc="Progress"):
        try:
            paper_data ={
                'arxiv_id': result.entry_id.split('/')[-1],
                'title': result.title.strip(),
                'abstract': result.summary.strip().replace('\n', ' '),
                'authors': [author.name for author in result.authors],
                'published': result.updated.strftime('%Y-%m-%d'),
                'updated': result.updated.strftime('%Y-%m-%d'),
                'categories': result.categories, 
                'primary_category': result.primary_category, 
                'pdf_url': result.pdf_url, 
                'comment': result.comment if result.comment else "",
                'journal_ref': result.journal_ref if result.journal_ref else ""
            }
            papers.append(paper_data)
        except Exception as e:
            print(f"\n Error processing paper: {e}")
            continue 
    return papers

def save_papers_to_json(papers, filename="arxiv_papers.json"):
    """
    Save papers to JSON file. 
    Args:
        papers (list): List of paper dictionary
        filename (str): Output filename
    """
    output_path = DATA_DIR / filename
    data_package = {
        'metadata':{
            'total_papers': len(papers),
            'fetch_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'categories': list(set(cat for paper in papers for cat in paper['categories']))
        },
        'papers':papers 
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_package, f, indent=2, ensure_ascii=False)
    print(f"\n Saved to: {output_path}")
    return output_path

def display_sample(papers, n=5):
    """
    Display sample of papers in a nice table. 
    Args:
        papers (list): List of paper dictionaries
        n(int): Number of papers to display
    """
    df = pd.DataFrame(papers)
    display_columns = ['arxiv_id', 'title', 'primary_category', 'published']
    
    if len(df) > 0:
        print(f"\n Sample of {min(n, len(df))} papers: \n")
        print(df[display_columns].head(n).to_string(index=False))
        
        # Show Statistics
        print(f"\n Statistics: ")
        print(f"   Total papers: {len(df)}")
        print(f"   Date range: {df['published'].min()} to {df['published'].max()}")
        print(f"   Categories: {df['primary_category'].nunique} unique")
        print(f"   Average authors per paper: {df['authors'].apply(len).mean():.1f}")
    else:
        print("\n No papers fetched!")
        
def main():
    """
    Main function to run the data collection. 
    """
    
    print("="*60)
    print("Research Paper Intelligence System")
    print("Feature 1: arXiv Data Collection")
    print("="*60)
    
    # Fetch papers
    papers = fetch_arxiv_papers(max_results=100, categories=['cs.AI', 'cs.LG', 'cs.CL'])
    # Save to JSON
    if papers:
        output_path = save_papers_to_json(papers)
        # Display sample
        display_sample(papers, n=5)
        print("="*60)
        print(f"{len(papers)} papers successfully downloaded")
        print(f"Data saved to: {output_path}")
        print("="*60)
    else:
        print("\n No papers were fetched. Check your internet connection.")
        
if __name__=="__main__":
    main()