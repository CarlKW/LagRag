import json
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def normalize_law(law_str):
    # Normalize law format for comparison
    if not law_str:
        return ""
    normalized = law_str.lower().replace("sfs-", "").replace("sfs ", "")
    normalized = normalized.replace(":", "-")
    return normalized

def normalize_paragraph(golden_paras):
    # Normalize a list of paras for comparison
    normalized_golden_paragraphs = []
    for para in golden_paras:
        if "," in para:  
            split_paras = [p.strip() for p in para.split(",")]
            normalized_golden_paragraphs.extend(split_paras)
        else:
            normalized_golden_paragraphs.append(para)
    
    return normalized_golden_paragraphs


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
                    print(f"Error on line {line_num}: {e}")
                    print(f"Current json{current_json[:200]}")
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

for entry in golden_entries:
    query = entry["query"]
    golden_p = entry["gold_paragraphs"]
    law = entry["law"]
    results = retriever.retrieve(query)

    correct_law_chunks = []
    normalized_golden_law = normalize_law(law)
    normalized_golden_para = normalize_paragraph(golden_p) 
    
    for candidate in results["initial_candidates"]:
        candidate_law = candidate.get("sfs_nr", "")
        normalized_candidate_law = normalize_law(candidate_law)
        
        if normalized_golden_law == normalized_candidate_law:
            correct_law_chunks.append(candidate)


    print(f"\n{'='*80}")
    print(f"Number of chunks with correct law in inital search: {len(correct_law_chunks)}/{len(results['initial_candidates'])}")
    print(f"\n{'='*80}")
        
   # retrieved_paragraphs = []                              We dont check the paragraphs for the intial 50 candidates, onlu the laws 
    #for candidate in results["initial_candidates"]:
    #    if "paragraf" in candidate: 
     #       retrieved_paragraphs.append(candidate["paragraf"])
    
    # prnt top retrieved chunks for inspection
    
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

   # found_paragraphs = []
   # for gold_para in normalized_golden_para:
    #    if gold_para in retrieved_paragraphs:
     #       found_paragraphs.append(gold_para)

   # missing_paragraphs = []
   # for gold_para in normalized_golden_para:
   #     if gold_para not in retrieved_paragraphs:
   #         missing_paragraphs.append(gold_para)
    
    # Evaluate top 5 reranked chunks
    print(f"\n{'='*80}")
    print(f"TOP 10 RERANKED CHUNKS EVALUATION")
    
    reranked_results = results.get("reranked_results", [])
    top_reranked = reranked_results[:10]  # Get top reranked results
    
    if not top_reranked:
        print("No reranked results available.")
    else:
        correct_law_count = 0
        correct_paragraph_count = 0
        
        for i, chunk in enumerate(top_reranked, 1):
            rerank_score = chunk.get("score_rerank")
            retrieval_score = chunk.get("score_retrieval")
            chunk_law = chunk.get("sfs_nr", "")
            chunk_paragraphs = []
            chunk_paragraph = chunk.get("paragraf", "N/A")

            # Add the main chunk paragraph if it exists
            if chunk_paragraph != "N/A":
                chunk_paragraphs.append(chunk_paragraph)

            # Check if the retrieved chunk has surrounding context paragraphs and add those do the list
            if chunk.get("has_surrounding_context"):
                context_paragraphs = chunk.get("context_paragraphs")
                if context_paragraphs:
                    context_list = [p.strip() for p in context_paragraphs.split(",")]
                    chunk_paragraphs.extend(context_list)

            
            # Normalize for comparison
            normalized_chunk_para = normalize_paragraph(chunk_paragraphs)
            normalized_chunk_law = normalize_law(chunk_law)


            is_correct_law = normalized_chunk_law == normalized_golden_law
            nr_correct_paragraphs = sum(para in normalized_golden_para for para in normalized_chunk_para)
            
            if is_correct_law:
                correct_law_count += 1
            

            
            print(f"\n--- Reranked Chunk {i} ---")
            print(f"Law (SFS): {chunk_law} {'(CORRECT)' if is_correct_law else '(WRONG)'}")
            print(f"Fraction of correct paragraphs in chunk {nr_correct_paragraphs}/{len(normalized_golden_para)}")
            if rerank_score is not None:
                print(f"Rerank Score: {rerank_score:.4f}")
            if retrieval_score is not None:
                print(f"Retrieval Score: {retrieval_score:.4f}")
            print(f"Title: {chunk.get('titel', chunk.get('title', 'N/A'))[:80]}...")
            text_preview = chunk.get('text', 'N/A')
            print(f"Text preview: {text_preview[:200]}..." if len(text_preview) > 100 else f"Text preview: {text_preview}")
        
