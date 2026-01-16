#  convert filtered SFS JSONL file into langchain Documents
import json
from pathlib import Path
from typing import Iterator, List

from langchain_core.documents import Document


# determine document is a 'lag' or 'förordning' 
def determine_type(titel: str) -> str:
    titel_lower = titel.lower()
    if "lag" in titel_lower and "förordning" not in titel_lower:
        return "lag"
    elif "förordning" in titel_lower:
        return "förordning"
    else:
        return "lag"


def load_sfs_documents(jsonl_path: str) -> List[Document]:
    documents = []
    jsonl_file = Path(jsonl_path)
    
    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                fulltext = record.get("fulltext", "")
                if not fulltext:
                    continue  
                
                titel = record.get("titel", "")
                dok_id = record.get("dok_id", "")
                beteckning = record.get("beteckning", "")  # This is the SFS number
                
                metadata_dict = record.get("metadata", {})
                dokument_url_html = metadata_dict.get("dokument_url_html", "")
                organ = metadata_dict.get("organ", "")
                
                doc_type = determine_type(titel)
                
                doc_metadata = {
                    "sfs_nr": beteckning,
                    "titel": titel.strip() if titel else "",
                    "typ": doc_type,
                    "dok_id": dok_id,
                    "dokument_url_html": dokument_url_html,
                    "organ": organ,
                }
                
                # Create LangChain Document
                doc = Document(
                    page_content=fulltext,
                    metadata=doc_metadata
                )
                
                documents.append(doc)
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    return documents


# lazy loader for SFS documents 
def load_sfs_documents_lazy(jsonl_path: str) -> Iterator[Document]:
    jsonl_file = Path(jsonl_path)
    
    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                
                fulltext = record.get("fulltext", "")
                if not fulltext:
                    continue  
                
                titel = record.get("titel", "")
                dok_id = record.get("dok_id", "")
                beteckning = record.get("beteckning", "")  # This is the SFS number
                
                metadata_dict = record.get("metadata", {})
                dokument_url_html = metadata_dict.get("dokument_url_html", "")
                organ = metadata_dict.get("organ", "")
                
                doc_type = determine_type(titel)
                
                doc_metadata = {
                    "sfs_nr": beteckning,
                    "titel": titel.strip() if titel else "",
                    "typ": doc_type,
                    "dok_id": dok_id,
                    "dokument_url_html": dokument_url_html,
                    "organ": organ,
                }
                
                doc = Document(
                    page_content=fulltext,
                    metadata=doc_metadata
                )
                
                yield doc
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue


if __name__ == "__main__":
    # Example 
    project_root = Path(__file__).parent.parent.parent
    jsonl_file = project_root / "data" / "sfs_lagboken_1990plus_filtered.jsonl"
    
    print(f"Loading documents from {jsonl_file}...")
    documents = load_sfs_documents(str(jsonl_file))
    print(f"Loaded {len(documents)} documents")
    
    #first document as example
    if documents:
        print("\nFirst document example:")
        print(f"  Type: {documents[0].metadata['typ']}")
        print(f"  SFS nr: {documents[0].metadata['sfs_nr']}")
        print(f"  Title: {documents[0].metadata['titel'][:100]}...")
        print(f"  Page content length: {len(documents[0].page_content)} characters")
        print(f"  Organ: {documents[0].metadata['organ']}")

