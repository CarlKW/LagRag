import json
import os
import re
from pathlib import Path


def clean_text(text: str) -> str:

    if not text:
        return text
    
    # Normalize line breaks: \r\n -> \n, then \r -> \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Convert tabs to spaces
    text = text.replace('\t', ' ')
    
    # Collapse multiple consecutive newlines to max 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Collapse multiple consecutive dashes (3+ to single)
    text = re.sub(r'-{3,}', '-', text)
    
    # Collapse multiple consecutive spaces to single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Collapse multiple consecutive underscores (3+ to single)
    text = re.sub(r'_{3,}', '_', text)
    
    # Trim whitespace from start and end
    text = text.strip()
    
    return text


def filter_and_clean_jsonl(input_file: str, output_file: str = None):

    # Get absolute paths
    input_path = Path(input_file)
    
    if output_file is None:
        output_file = str(input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}")
    
    output_path = Path(output_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    total_rows = 0
    filtered_rows = 0
    
    with open(input_path, "r", encoding="utf-8") as f_in:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                total_rows += 1
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    titel = record.get("titel", "")
                    
                    if "Lag" not in titel and "Förordning" not in titel:
                        continue
                    
                    if "/r1/" in titel:
                        titel = titel.replace("/r1/", "")
                        record["titel"] = titel
                    
                    if "metadata" in record and isinstance(record["metadata"], dict):
                        metadata_titel = record["metadata"].get("titel", "")
                        if "/r1/" in metadata_titel:
                            record["metadata"]["titel"] = metadata_titel.replace("/r1/", "")
                    
                    if "fulltext" in record:
                        record["fulltext"] = clean_text(record["fulltext"])
                    
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    filtered_rows += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON on line {total_rows}: {e}")
                    continue
    
    print(f"Processing complete:")
    print(f"  Total rows processed: {total_rows}")
    print(f"  Rows kept (with 'Lag' or 'Förordning'): {filtered_rows}")
    print(f"  Rows removed: {total_rows - filtered_rows}")
    print(f"  Output file: {output_path}")


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    input_file = project_root / "data" / "sfs_lagboken_1990plus.jsonl"
    output_file = project_root / "data" / "sfs_lagboken_1990plus_filtered.jsonl"
    
    # input_file = project_root / "data" / "test.jsonl"
    # output_file = project_root / "data" / "test_out.jsonl"
    
    filter_and_clean_jsonl(str(input_file), str(output_file))


if __name__ == "__main__":
    main()

