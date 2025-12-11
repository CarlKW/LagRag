"""
Document inspection tool for analyzing SFS document structure and content.

This script provides comprehensive analysis of retrieved documents including
metadata structure, content statistics, and structural patterns.
"""

import argparse
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from src.ingestion.loader import load_sfs_documents, load_sfs_documents_lazy


def estimate_tokens(text: str, model: str = "gpt-4") -> Optional[int]:
    """Estimate token count using tiktoken if available."""
    if not TIKTOKEN_AVAILABLE:
        return None
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        return None


def detect_sections(text: str) -> List[Dict[str, str]]:
    """
    Detect common Swedish legal document section patterns.
    
    Returns list of detected sections with their labels and positions.
    """
    sections = []
    
    # Common patterns: "1 kap.", "Kapitel 1", "1 §", "Artikel 1", etc.
    patterns = [
        (r'(\d+)\s*kap\.', 'kapitel'),
        (r'Kapitel\s+(\d+)', 'kapitel'),
        (r'(\d+)\s*§', 'paragraf'),
        (r'Artikel\s+(\d+)', 'artikel'),
        (r'(\d+)\s*st\.', 'stycke'),
    ]
    
    for pattern, section_type in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            sections.append({
                'type': section_type,
                'number': match.group(1),
                'position': match.start(),
                'label': match.group(0)
            })
    
    return sorted(sections, key=lambda x: x['position'])


def analyze_text_structure(text: str) -> Dict[str, any]:
    """Analyze the structural properties of document text."""
    lines = text.split('\n')
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    # Line length distribution
    line_lengths = [len(line) for line in lines if line.strip()]
    
    # Word count (Swedish-aware: split on whitespace)
    words = text.split()
    word_count = len(words)
    
    # Sentence count (approximate: count sentence-ending punctuation)
    sentences = re.split(r'[.!?]+\s+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Character statistics
    char_count = len(text)
    char_count_no_ws = len(text.replace(' ', '').replace('\n', '').replace('\t', ''))
    
    # Whitespace analysis
    whitespace_chars = len([c for c in text if c.isspace()])
    tab_count = text.count('\t')
    newline_count = text.count('\n')
    
    return {
        'total_characters': char_count,
        'characters_no_whitespace': char_count_no_ws,
        'whitespace_characters': whitespace_chars,
        'tab_count': tab_count,
        'newline_count': newline_count,
        'line_count': len(lines),
        'non_empty_line_count': len([l for l in lines if l.strip()]),
        'paragraph_count': len(paragraphs),
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_words_per_sentence': word_count / sentence_count if sentence_count > 0 else 0,
        'avg_line_length': statistics.mean(line_lengths) if line_lengths else 0,
        'max_line_length': max(line_lengths) if line_lengths else 0,
        'min_line_length': min(line_lengths) if line_lengths else 0,
        'median_line_length': statistics.median(line_lengths) if line_lengths else 0,
    }


def format_metadata(metadata: Dict[str, any]) -> str:
    """Format metadata dictionary for display."""
    lines = []
    for key, value in sorted(metadata.items()):
        if value is None:
            value_str = "None"
        elif isinstance(value, str):
            value_str = value if len(value) < 80 else value[:77] + "..."
        else:
            value_str = str(value)
        lines.append(f"  {key:25s}: {value_str}")
    return "\n".join(lines)


def inspect_single_document(doc, index: Optional[int] = None) -> None:
    """Inspect and display analysis of a single document."""
    print("=" * 80)
    print("DOCUMENT INSPECTION REPORT")
    print("=" * 80)
    
    if index is not None:
        print(f"\nDocument Index: {index}")
    
    # Metadata section
    print("\n" + "-" * 80)
    print("METADATA")
    print("-" * 80)
    print(format_metadata(doc.metadata))
    
    # Content statistics
    print("\n" + "-" * 80)
    print("CONTENT STATISTICS")
    print("-" * 80)
    
    stats = analyze_text_structure(doc.page_content)
    
    print(f"  Total Characters:           {stats['total_characters']:,}")
    print(f"  Characters (no whitespace): {stats['characters_no_whitespace']:,}")
    print(f"  Whitespace Characters:      {stats['whitespace_characters']:,}")
    print(f"  Tabs:                       {stats['tab_count']:,}")
    print(f"  Newlines:                   {stats['newline_count']:,}")
    print(f"  Total Lines:                {stats['line_count']:,}")
    print(f"  Non-empty Lines:            {stats['non_empty_line_count']:,}")
    print(f"  Paragraphs:                 {stats['paragraph_count']:,}")
    print(f"  Words:                      {stats['word_count']:,}")
    print(f"  Sentences:                  {stats['sentence_count']:,}")
    print(f"  Avg Words per Sentence:     {stats['avg_words_per_sentence']:.2f}")
    print(f"  Avg Line Length:            {stats['avg_line_length']:.1f}")
    print(f"  Median Line Length:         {stats['median_line_length']:.1f}")
    print(f"  Max Line Length:            {stats['max_line_length']:,}")
    print(f"  Min Line Length:            {stats['min_line_length']:,}")
    
    # Token estimation
    if TIKTOKEN_AVAILABLE:
        token_count = estimate_tokens(doc.page_content)
        if token_count:
            print(f"  Estimated Tokens (GPT-4):    {token_count:,}")
    else:
        print(f"  Estimated Tokens:            (tiktoken not available)")
    
    # Section detection
    print("\n" + "-" * 80)
    print("STRUCTURAL ANALYSIS")
    print("-" * 80)
    
    sections = detect_sections(doc.page_content)
    if sections:
        section_types = Counter([s['type'] for s in sections])
        print(f"  Detected Sections:          {len(sections)}")
        print(f"  Section Types:")
        for sec_type, count in section_types.most_common():
            print(f"    - {sec_type}: {count}")
        
        print(f"\n  First 10 Sections:")
        for i, section in enumerate(sections[:10], 1):
            print(f"    {i}. {section['label']} ({section['type']}) at position {section['position']}")
    else:
        print("  No standard section markers detected")
    
    # Full content
    print("\n" + "-" * 80)
    print("FULL DOCUMENT CONTENT")
    print("-" * 80)
    
    content = doc.page_content
    
    print(f"\nFull document content ({len(content):,} characters):")
    print("-" * 80)
    print(content)
    
    print("\n" + "=" * 80)


def inspect_batch(documents: List, limit: Optional[int] = None) -> None:
    """Analyze and display statistics for a batch of documents."""
    docs_to_analyze = documents[:limit] if limit else documents
    
    print("=" * 80)
    print("BATCH DOCUMENT ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing {len(docs_to_analyze)} documents")
    if limit and len(documents) > limit:
        print(f"(limited to first {limit} documents from {len(documents)} total)")
    
    # Collect statistics
    all_stats = []
    metadata_keys = set()
    doc_types = Counter()
    
    for doc in docs_to_analyze:
        stats = analyze_text_structure(doc.page_content)
        all_stats.append(stats)
        metadata_keys.update(doc.metadata.keys())
        doc_types[doc.metadata.get('typ', 'unknown')] += 1
    
    # Aggregate statistics
    print("\n" + "-" * 80)
    print("AGGREGATE STATISTICS")
    print("-" * 80)
    
    print(f"\nDocument Types:")
    for doc_type, count in doc_types.most_common():
        print(f"  {doc_type}: {count}")
    
    print(f"\nMetadata Fields Found: {len(metadata_keys)}")
    print(f"  {', '.join(sorted(metadata_keys))}")
    
    # Calculate aggregate metrics
    total_chars = sum(s['total_characters'] for s in all_stats)
    total_words = sum(s['word_count'] for s in all_stats)
    avg_chars = statistics.mean([s['total_characters'] for s in all_stats])
    avg_words = statistics.mean([s['word_count'] for s in all_stats])
    median_chars = statistics.median([s['total_characters'] for s in all_stats])
    median_words = statistics.median([s['word_count'] for s in all_stats])
    
    print(f"\nContent Metrics:")
    print(f"  Total Characters (all docs): {total_chars:,}")
    print(f"  Total Words (all docs):     {total_words:,}")
    print(f"  Avg Characters per Doc:     {avg_chars:,.0f}")
    print(f"  Median Characters per Doc:  {median_chars:,.0f}")
    print(f"  Avg Words per Doc:          {avg_words:,.0f}")
    print(f"  Median Words per Doc:       {median_words:,.0f}")
    
    # Distribution statistics
    char_lengths = [s['total_characters'] for s in all_stats]
    word_lengths = [s['word_count'] for s in all_stats]
    
    print(f"\nCharacter Count Distribution:")
    print(f"  Min:    {min(char_lengths):,}")
    print(f"  Max:    {max(char_lengths):,}")
    print(f"  Mean:   {statistics.mean(char_lengths):,.0f}")
    print(f"  Median: {statistics.median(char_lengths):,.0f}")
    if len(char_lengths) > 1:
        print(f"  StdDev: {statistics.stdev(char_lengths):,.0f}")
    
    print(f"\nWord Count Distribution:")
    print(f"  Min:    {min(word_lengths):,}")
    print(f"  Max:    {max(word_lengths):,}")
    print(f"  Mean:   {statistics.mean(word_lengths):,.0f}")
    print(f"  Median: {statistics.median(word_lengths):,.0f}")
    if len(word_lengths) > 1:
        print(f"  StdDev: {statistics.stdev(word_lengths):,.0f}")
    
    print("\n" + "=" * 80)


def find_document_by_sfs(documents: List, sfs_number: str) -> Optional[Tuple[int, any]]:
    """Find a document by its SFS number."""
    for idx, doc in enumerate(documents):
        if doc.metadata.get('sfs_nr') == sfs_number:
            return (idx, doc)
    return None


def main():
    """Main entry point for the inspection script."""
    parser = argparse.ArgumentParser(
        description="Inspect SFS documents for structure and content analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Path to JSONL file (default: data/sfs_lagboken_1990plus_filtered.jsonl)'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--index',
        type=int,
        help='Inspect document at this index (0-based)'
    )
    group.add_argument(
        '--sfs',
        type=str,
        help='Inspect document with this SFS number (e.g., "1991:1867")'
    )
    group.add_argument(
        '--batch',
        action='store_true',
        help='Analyze batch statistics for all documents'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents for batch analysis'
    )
    
    args = parser.parse_args()
    
    # Determine data file path
    if args.data_file:
        data_file = Path(args.data_file)
    else:
        project_root = Path(__file__).parent.parent.parent
        data_file = project_root / "data" / "sfs_lagboken_1990plus_filtered.jsonl"
    
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return 1
    
    # Load documents
    print(f"Loading documents from: {data_file}")
    
    if args.batch or args.index is not None:
        # Need full list for batch or index access
        documents = load_sfs_documents(str(data_file))
        print(f"Loaded {len(documents)} documents\n")
    else:
        # For SFS lookup, we can use lazy loading but need to collect matches
        documents = list(load_sfs_documents_lazy(str(data_file)))
        print(f"Loaded {len(documents)} documents\n")
    
    if not documents:
        print("Error: No documents found in file")
        return 1
    
    # Execute requested inspection
    if args.batch:
        inspect_batch(documents, limit=args.limit)
    elif args.index is not None:
        if args.index < 0 or args.index >= len(documents):
            print(f"Error: Index {args.index} out of range (0-{len(documents)-1})")
            return 1
        inspect_single_document(documents[args.index], index=args.index)
    elif args.sfs:
        result = find_document_by_sfs(documents, args.sfs)
        if result:
            idx, doc = result
            inspect_single_document(doc, index=idx)
        else:
            print(f"Error: Document with SFS number '{args.sfs}' not found")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

