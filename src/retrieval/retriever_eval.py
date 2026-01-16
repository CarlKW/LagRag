import json
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def normalize_law(law_str):
    """Normalize law format for comparison"""
    if not law_str:
        return ""
    # Remove "sfs-" prefix, convert to lowercase
    normalized = law_str.lower().replace("sfs-", "").replace("sfs ", "")
    # Handle both "2025:1255" and "2025-1255" formats
    normalized = normalized.replace(":", "-")
    return normalized

golden_file = Path("data/golden_paragraphs.jsonl")
golden_entries = []

with open(golden_file,"r", encoding = "utf-8") as f:
    current_json = ""
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            if current_json:
                try:
                    entry = json.loads(current_json)
                    golden_entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error on line {e}")
                    print(f"Current json; {current_json[:200]}..")
                current_json = ""
            continue 
            
        current_json += line + " "

    if current_json:
        try:
            entry = json.loads(current_json)
            golden_entries.append(entry)
        except json.JSONDecodeError as e:
            print(f"Error parsing last JSON: {e}")


print(f"\nLoaded {len(golden_entries)} entries")
if golden_entries:
    print("\nFirst entry:")
    print(golden_entries[0])

from src.retrieval.retriever import RerankingRetriever
retriever = RerankingRetriever()

stored_chunks = []
for entry in golden_entries:
    query = entry["query"]
    golden_p = entry["gold_paragraphs"]
    law = entry["law"]

    # normalize the law(sfs) text for correct scoring
    normalized_gold = []
    for para in golden_p:
        if "," in para:  # If it contains commas, split it
            # Split by comma and strip whitespace
            split_paras = [p.strip() for p in para.split(",")]
            normalized_gold.extend(split_paras)
        else:
            normalized_gold.append(para)  # Already a single paragraph
    
    golden_p = normalized_gold 


    # Retreival
    results = retriever.retrieve(query)

    # First, filter chunks to only those from the correct law
    correct_law_chunks = []
    # Normalize the golden law ONCE (outside the loop)
    normalized_golden_law = normalize_law(law)
    
    # Single loop to check each candidate
    for candidate in results["initial_candidates"]:
        candidate_law = candidate.get("sfs_nr", "")
        normalized_candidate_law = normalize_law(candidate_law)
        
        # Check INSIDE the loop
        if normalized_golden_law == normalized_candidate_law:
            correct_law_chunks.append(candidate)
    
    print(f"Found {len(correct_law_chunks)} chunks from correct law out of {len(results['initial_candidates'])} total")
    if len(correct_law_chunks) == 0:
        print(f"Did not find the correct law for '{query}' ")
        

    # Extract paragraphs from all candidate results
    retrieved_paragraphs = []
    for candidate in results["initial_candidates"]:
        if "paragraf" in candidate: 
            retrieved_paragraphs.append(candidate["paragraf"])
    
    # Print top retrieved chunks for inspection
    
   # for i, candidate in enumerate(results["initial_candidates"][:50], 1):  # Top 10
   #     print(f"\n--- Chunk {i} ---")
   #     print(f"Law (SFS): {candidate.get("sfs_nr")}")
   #     print(f"Law/Title: {candidate.get('titel', candidate.get('title', 'N/A'))}")
   ##     print(f"Paragraph: {candidate.get('paragraf', 'N/A')}")
   #     print(f"Retrieval Score: {candidate.get('score_retrieval', 'N/A'):.4f}")
   #     print(f"Rerank Score: {candidate.get('score_rerank', 'N/A'):.4f}")
   #     print(f"Text preview:")
   #     text = candidate.get('text', 'N/A')
   #     print(f"  {text[:300]}..." if len(text) > 300 else f"  {text}")

    found_paragraphs = []
    for gold_para in golden_p:
        if gold_para in retrieved_paragraphs:
            found_paragraphs.append(gold_para)

    missing_paragraphs = []
    for gold_para in golden_p:
        if gold_para not in retrieved_paragraphs:
            missing_paragraphs.append(gold_para)
    

        
    # Print results for this query
    print(f"\n{'_'*80}")
    print(f"Query: {query}")
    print(f"Law: {law}")
    print(f"Gold paragraphs: {golden_p}")
    print(f"Found: {found_paragraphs}")
    print(f"Missing: {missing_paragraphs}")
    print(f"Score: {len(found_paragraphs)}/{len(golden_p)}")

