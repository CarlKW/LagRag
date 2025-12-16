
import re
from typing import List

from langchain_core.documents import Document


def find_paragraph_starts(text: str) -> List[tuple[int, str]]:
    """
    Find all paragraph start positions in the text.
    
    Paragraphs start either:
    1. At the beginning of the document, or
    2. After two newline characters "\n\n",
    followed by:
       - a number (one or more digits), optionally followed by a single lowercase letter (e.g. "31 a"),
       - a space,
       - and a SINGLE "§" character.
    
    Args:
        text: The full text to search
        
    Returns:
        List of tuples: (start_position, paragraph_label)
        e.g., [(0, "1 §"), (150, "31 a §")]
    """
    # Normalize newlines
    text = text.replace("\r\n", "\n")
    
    # Pattern: (start of string OR \n\n) + number + optional letter + space + single §
    # We use negative lookahead to ensure we don't match "§§"
    # Match at start of string or after \n\n, then capture the paragraph number
    pattern = r'(?:^|\n\n)(\d+(?:\s+[a-z])?)\s+§(?!§)'
    
    matches = []
    for match in re.finditer(pattern, text, re.MULTILINE):
        start_pos = match.start()
        paragraph_label = match.group(1) + " §"
        
        # If match starts with \n\n, adjust position to after the newlines
        # The match.start() gives us the position of the \n\n or start
        if start_pos > 0 and text[start_pos:start_pos+2] == "\n\n":
            start_pos += 2
        
        matches.append((start_pos, paragraph_label))
    
    return matches


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences for chunking long paragraphs.
    Improved version that handles Swedish abbreviations and edge cases.
    """
    # Common Swedish abbreviations that shouldn't end sentences
    swedish_abbreviations = {
        't.ex.', 'dvs.', 'm.fl.', 'm.m.', 'osv.', 'bl.a.', 's.k.', 'f.n.',
        't.o.m.', 'fr.o.m.', 'ca.', 'kr.', 'st.', 's.', 'nr.', 'kap.',
        '§', '§§', 'art.', 'p.', 's.', 'st.', 'jfr.', 'jf.', 'ev.',
        'resp.', 'inkl.', 'exkl.', 'etc.', 'prof.', 'dr.', 'mrs.', 'mr.'
    }
    
    # Pattern: sentence ending (. ! ?) followed by space or newline
    # But we need to check if it's actually an abbreviation
    sentences = []
    last_end = 0
    
    # Find all potential sentence endings
    for match in re.finditer(r'([.!?])(\s+|$)', text):
        end_pos = match.end()
        punct = match.group(1)
        
        # Check if this is likely an abbreviation
        # Look back up to 10 characters to find word boundaries
        lookback_start = max(0, match.start() - 15)
        lookback_text = text[lookback_start:match.start() + 1].lower()
        
        # Check if any abbreviation pattern matches
        is_abbreviation = False
        for abbrev in swedish_abbreviations:
            if lookback_text.endswith(abbrev.lower()):
                is_abbreviation = True
                break
        
        # Also check for decimal numbers (e.g., "3.5", "12.3")
        if punct == '.':
            # Check if there's a digit before and after
            before_char = text[match.start() - 1] if match.start() > 0 else ''
            after_match = re.search(r'\d', text[match.end():match.end() + 3])
            if before_char.isdigit() and after_match:
                is_abbreviation = True
        
        # If it's not an abbreviation, it's a sentence boundary
        if not is_abbreviation:
            sentence = text[last_end:end_pos].strip()
            if sentence:
                sentences.append(sentence)
            last_end = end_pos
    
    # Add remaining text
    if last_end < len(text):
        remaining = text[last_end:].strip()
        if remaining:
            sentences.append(remaining)
    
    # If no sentences were found (e.g., no punctuation), return whole text as one sentence
    if not sentences:
        return [text.strip()] if text.strip() else []
    
    return sentences


def count_words(text: str) -> int:
    return len(text.split())


def validate_chunks(chunks: List[Document], min_words: int, max_words: int) -> dict:
    """
    Validate chunk quality and return statistics.
    
    Args:
        chunks: List of chunk Documents
        min_words: Expected minimum words per chunk
        max_words: Expected maximum words per chunk
        
    Returns:
        Dictionary with validation statistics
    """
    stats = {
        'total_chunks': len(chunks),
        'too_short': 0,
        'too_long': 0,
        'empty_chunks': 0,
        'word_counts': [],
        'avg_words': 0,
        'min_words_found': float('inf'),
        'max_words_found': 0
    }
    
    for chunk in chunks:
        word_count = count_words(chunk.page_content)
        
        if not chunk.page_content.strip():
            stats['empty_chunks'] += 1
        elif word_count < min_words:
            stats['too_short'] += 1
        elif word_count > max_words:
            stats['too_long'] += 1
        
        if word_count > 0:
            stats['word_counts'].append(word_count)
            stats['min_words_found'] = min(stats['min_words_found'], word_count)
            stats['max_words_found'] = max(stats['max_words_found'], word_count)
    
    if stats['word_counts']:
        stats['avg_words'] = sum(stats['word_counts']) / len(stats['word_counts'])
    
    return stats


def chunk_documents(
    docs: List[Document], 
    min_words: int = 75, 
    max_words: int = 500, 
    overlap_sentences: int = 2,
    include_surrounding_paragraphs: bool = True,
    short_document_threshold: int = 100
) -> List[Document]:
    """
    Split documents into paragraph-level chunks with optimized settings for legal documents.
    
    Args:
        docs: List of raw Documents (full law/regulation text)
        min_words: Minimum words per chunk (merge with next if shorter)
        max_words: Maximum words per chunk (split if longer)
        overlap_sentences: Number of sentences to overlap when splitting long paragraphs
        include_surrounding_paragraphs: Whether to include previous/next paragraphs for context
        short_document_threshold: Documents with fewer words than this are kept as single chunks
        
    Returns:
        List of Documents, each representing a paragraph chunk
    """
    all_chunks = []
    
    for doc in docs:
        text = doc.page_content
        if not text:
            # Create empty chunk with metadata (minimum guarantee)
            chunk = Document(
                page_content="",
                metadata={**doc.metadata, "paragraf": "empty_document", "is_short_document": True, "total_words": 0}
            )
            all_chunks.append(chunk)
            continue
        
        # Normalize newlines
        text = text.replace("\r\n", "\n")
        
        # Check if document is short - keep as single chunk
        total_words = count_words(text)
        if total_words < short_document_threshold:
            # Keep entire document as single chunk
            chunk = Document(
                page_content=text.strip(),
                metadata={
                    **doc.metadata,
                    "paragraf": "complete_document",
                    "is_short_document": True,
                    "total_words": total_words
                }
            )
            all_chunks.append(chunk)
            continue  # Skip paragraph-level processing
        
        # Find all paragraph starts
        para_starts = find_paragraph_starts(text)
        
        if not para_starts:
            # No paragraphs found, treat entire document as one chunk
            chunk = Document(
                page_content=text.strip(),
                metadata={**doc.metadata, "paragraf": "unknown", "is_short_document": False}
            )
            all_chunks.append(chunk)
            continue
        
        # Process each paragraph
        i = 0
        while i < len(para_starts):
            start_pos, para_label = para_starts[i]
            
            # Determine end position (start of next paragraph or end of text)
            if i + 1 < len(para_starts):
                end_pos = para_starts[i + 1][0]
            else:
                end_pos = len(text)
            
            # Extract paragraph text
            para_text = text[start_pos:end_pos].strip()
            
            # Build chunk with surrounding paragraphs if enabled
            chunk_text = para_text
            context_paragraphs = [para_label]
            has_surrounding_context = False
            included_next_paragraph = False
            
            if include_surrounding_paragraphs:
                # Try to include previous paragraph for context
                if i > 0:
                    prev_start, prev_label = para_starts[i - 1]
                    prev_end = start_pos
                    prev_para_text = text[prev_start:prev_end].strip()
                    prev_word_count = count_words(prev_para_text)
                    current_word_count = count_words(para_text)
                    
                    # Include previous paragraph if it fits within max_words
                    if prev_word_count + current_word_count <= max_words:
                        chunk_text = prev_para_text + "\n\n" + chunk_text
                        context_paragraphs.insert(0, prev_label)
                        has_surrounding_context = True
                
                # Try to include next paragraph for forward context
                if i + 1 < len(para_starts):
                    next_start, next_label = para_starts[i + 1]
                    next_end = para_starts[i + 2][0] if i + 2 < len(para_starts) else len(text)
                    next_para_text = text[next_start:next_end].strip()
                    next_word_count = count_words(next_para_text)
                    current_total = count_words(chunk_text)
                    
                    # Include next paragraph if it fits within max_words
                    if current_total + next_word_count <= max_words:
                        chunk_text = chunk_text + "\n\n" + next_para_text
                        context_paragraphs.append(next_label)
                        has_surrounding_context = True
                        included_next_paragraph = True
            
            # Check if paragraph needs to be split (too long)
            word_count = count_words(chunk_text)
            
            if word_count > max_words:
                # Split into multiple chunks
                sentences = split_into_sentences(chunk_text)
                
                if not sentences:
                    # Fallback: if sentence splitting failed, split by words
                    words = chunk_text.split()
                    sentences = []
                    for j in range(0, len(words), 50):  # Roughly 50 words per sentence chunk
                        sentences.append(" ".join(words[j:j+50]))
                
                current_chunk = []
                current_words = 0
                chunk_index = 0
                split_chunks = []  # Collect all split chunks first
                
                for sentence in sentences:
                    sent_words = count_words(sentence)
                    
                    if current_words + sent_words > max_words and current_chunk:
                        # Save current chunk
                        split_chunk_text = " ".join(current_chunk).strip()
                        if split_chunk_text:  # Only add non-empty chunks
                            split_chunks.append({
                                'text': split_chunk_text,
                                'words': current_words,
                                'index': chunk_index
                            })
                        
                        # Start new chunk with overlap
                        overlap = current_chunk[-overlap_sentences:] if len(current_chunk) >= overlap_sentences else current_chunk
                        current_chunk = overlap + [sentence]
                        current_words = sum(count_words(s) for s in current_chunk)
                        chunk_index += 1
                    else:
                        current_chunk.append(sentence)
                        current_words += sent_words
                
                # Add final chunk
                if current_chunk:
                    split_chunk_text = " ".join(current_chunk).strip()
                    if split_chunk_text:
                        split_chunks.append({
                            'text': split_chunk_text,
                            'words': current_words,
                            'index': chunk_index
                        })
                
                # Merge chunks that are too short (below min_words)
                # Merge forward: if a chunk is too short, merge with next
                merged_split_chunks = []
                j = 0
                while j < len(split_chunks):
                    chunk_data = split_chunks[j]
                    
                    # If chunk is too short and not the last one, try to merge with next
                    if chunk_data['words'] < min_words and j + 1 < len(split_chunks):
                        next_chunk = split_chunks[j + 1]
                        merged_words = chunk_data['words'] + next_chunk['words']
                        
                        # Only merge if total doesn't exceed max_words too much (allow 20% overage)
                        if merged_words <= max_words * 1.2:
                            merged_text = chunk_data['text'] + " " + next_chunk['text']
                            merged_split_chunks.append({
                                'text': merged_text,
                                'words': merged_words,
                                'index': chunk_data['index']
                            })
                            j += 2  # Skip next chunk since we merged it
                            continue
                    
                    # Keep chunk as-is
                    merged_split_chunks.append(chunk_data)
                    j += 1
                
                # Create Document objects for all split chunks
                for chunk_data in merged_split_chunks:
                    chunk = Document(
                        page_content=chunk_data['text'],
                        metadata={
                            **doc.metadata,
                            "paragraf": para_label,
                            "subchunk_index": chunk_data['index'],
                            "has_surrounding_context": has_surrounding_context,
                            "context_paragraphs": ", ".join(context_paragraphs) if context_paragraphs else None,
                            "is_short_document": False,
                            "word_count": chunk_data['words']  # Add word count for debugging
                        }
                    )
                    all_chunks.append(chunk)
                
                # Skip next paragraph if we included it in surrounding context
                if included_next_paragraph:
                    i += 1  # Skip the next paragraph since we already included it
                i += 1
            else:
                # Paragraph is within acceptable size range
                # Check if paragraph is too short (merge with next)
                if word_count < min_words and i + 1 < len(para_starts):
                    # If we already included next paragraph as context, we're done
                    if included_next_paragraph:
                        # Already has next paragraph, just use current chunk
                        chunk = Document(
                            page_content=chunk_text.strip(),
                            metadata={
                                **doc.metadata,
                                "paragraf": para_label,
                                "has_surrounding_context": has_surrounding_context,
                                "context_paragraphs": ", ".join(context_paragraphs) if context_paragraphs else None,
                                "is_short_document": False,
                                "word_count": word_count
                            }
                        )
                        all_chunks.append(chunk)
                        # Skip next paragraph since we already included it
                        i += 2
                    else:
                        # Try to merge with next paragraph
                        next_start, next_label = para_starts[i + 1]
                        next_end = para_starts[i + 2][0] if i + 2 < len(para_starts) else len(text)
                        next_para_text = text[next_start:next_end].strip()
                        next_word_count = count_words(next_para_text)
                        merged_word_count = word_count + next_word_count
                        
                        # Only merge if total doesn't exceed max_words too much
                        if merged_word_count <= max_words * 1.2:
                            merged_text = chunk_text + "\n\n" + next_para_text
                            merged_context = context_paragraphs + [next_label]
                            
                            chunk = Document(
                                page_content=merged_text.strip(),
                                metadata={
                                    **doc.metadata,
                                    "paragraf": para_label,
                                    "merged_paragraphs": ", ".join([para_label, next_label]),
                                    "has_surrounding_context": len(merged_context) > 1,
                                    "context_paragraphs": ", ".join(merged_context) if len(merged_context) > 1 else None,
                                    "is_short_document": False,
                                    "word_count": merged_word_count
                                }
                            )
                            all_chunks.append(chunk)
                            i += 2  # Skip next paragraph since we merged it
                        else:
                            # Can't merge (would be too long), just use current chunk
                            chunk = Document(
                                page_content=chunk_text.strip(),
                                metadata={
                                    **doc.metadata,
                                    "paragraf": para_label,
                                    "has_surrounding_context": has_surrounding_context,
                                    "context_paragraphs": ", ".join(context_paragraphs) if context_paragraphs else None,
                                    "is_short_document": False,
                                    "word_count": word_count
                                }
                            )
                            all_chunks.append(chunk)
                            i += 1
                else:
                    # Normal-sized paragraph (within min_words and max_words)
                    chunk = Document(
                        page_content=chunk_text.strip(),
                        metadata={
                            **doc.metadata,
                            "paragraf": para_label,
                            "has_surrounding_context": has_surrounding_context,
                            "context_paragraphs": ", ".join(context_paragraphs) if context_paragraphs and len(context_paragraphs) > 1 else None,
                            "is_short_document": False,
                            "word_count": word_count
                        }
                    )
                    all_chunks.append(chunk)
                    # Skip next paragraph if we included it in surrounding context
                    if included_next_paragraph:
                        i += 1  # Skip the next paragraph since we already included it
                    i += 1
    
    # Validate chunks (optional - can be removed in production)
    validation_stats = validate_chunks(all_chunks, min_words, max_words)
    if validation_stats['too_short'] > 0 or validation_stats['too_long'] > 0:
        print(f"Warning: {validation_stats['too_short']} chunks below min_words, "
              f"{validation_stats['too_long']} chunks above max_words")
    
    return all_chunks


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add project root to Python path for imports
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    from src.ingestion.loader import load_sfs_documents
    jsonl_file = project_root / "data" / "sfs_lagboken_1990plus_filtered.jsonl"
    
    print("Loading documents...")
    documents = load_sfs_documents(str(jsonl_file))
    print(f"Loaded {len(documents)} documents")
    
    print("\nChunking documents...")
    chunks = chunk_documents(documents[:40])  # Test 
    print(f"Created {len(chunks)} chunks")
    
    if chunks:
        # Write chunks to a readable text file
        output_file = project_root / "data" / "chunks_output.txt"
        print(f"\nWriting chunks to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"CHUNKER TEST OUTPUT\n")
            f.write(f"Total chunks: {len(chunks)}\n")
            f.write(f"Source documents: {len(documents[:40])}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, chunk in enumerate(chunks, 1):
                f.write("-" * 80 + "\n")
                f.write(f"CHUNK {i} / {len(chunks)}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Paragraph: {chunk.metadata.get('paragraf', 'N/A')}\n")
                f.write(f"SFS nr: {chunk.metadata.get('sfs_nr', 'N/A')}\n")
                f.write(f"Title: {chunk.metadata.get('titel', 'N/A')}\n")
                
                # Write additional metadata if present
                if 'subchunk_index' in chunk.metadata:
                    f.write(f"Subchunk index: {chunk.metadata['subchunk_index']}\n")
                if 'merged_paragraphs' in chunk.metadata:
                    f.write(f"Merged paragraphs: {chunk.metadata['merged_paragraphs']}\n")
                
                # Write word count
                word_count = count_words(chunk.page_content)
                f.write(f"Word count: {word_count}\n")
                
                f.write("\n" + "-" * 80 + "\n")
                f.write("CONTENT:\n")
                f.write("-" * 80 + "\n")
                f.write(chunk.page_content)
                f.write("\n\n")
        
        print(f"✓ Chunks written to {output_file}")
        
        # Also print first chunk example to console
        print("\nFirst chunk example:")
        print(f"  Paragraph: {chunks[0].metadata.get('paragraf', 'N/A')}")
        print(f"  SFS nr: {chunks[0].metadata.get('sfs_nr', 'N/A')}")
        print(f"  Title: {chunks[0].metadata.get('titel', 'N/A')[:80]}...")
        print(f"  Content preview: {chunks[0].page_content[:200]}...")

