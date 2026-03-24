import re

def deduplicate_phrases(text: str) -> str:
    """
    Detects and removes consecutively repeated phrases.
    """
    # Regex logic:
    # Group 1: The phrase (at least 3 chars to avoid stripping simple words like "a a")
    # Group 2: The separator (space, punctuation) followed by the same phrase
    pattern = r'(?i)(\b.{3,}?)([\s,.-]+\1\b)+'
    
    # Iterative removal to handle nested loops
    while re.search(pattern, text):
        text = re.sub(pattern, r'\1', text)
    
    return text

def deduplicate_by_splitter(text: str, splitter: str) -> str:
    """
    Splits text by a specific delimiter, removes adjacent duplicates, and rejoins.
    """
    if splitter not in text:
        return text

    parts = [p.strip() for p in text.split(splitter) if p.strip()]
    
    if not parts:
        return text

    # Deduplication (preserve order, remove only adjacent duplicates)
    cleaned_parts = []
    prev_part = None
    
    for part in parts:
        # Case-insensitive comparison
        if part.lower() != (prev_part.lower() if prev_part else ""):
            cleaned_parts.append(part)
            prev_part = part

    if len(cleaned_parts) == 1:
        return cleaned_parts[0]
    
    return f"{splitter} ".join(cleaned_parts)

def clean_caption(text: str) -> str:
    """
    Main pipeline for cleaning generated captions.
    Applies filters for meta-phrases, blacklisted words, and repetitions.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Remove meta-phrases (e.g., "view of", "picture of")
    meta_patterns = [
        r'\b(a\s+|an\s+)?(screenshot|screen\s+shot|image|picture|photo|frame|video)\s+of\b',
        r'\b(high\s+quality|low\s+quality)\b',
        r'\b(blurry|noisy)\b'
    ]
    for pattern in meta_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 2. Remove specific blacklisted words (file extensions, technical terms)
    blacklist = ["screenshot", "screenshots", "jpg", "png", "avi", "mp4"]
    for word in blacklist:
        text = re.sub(r'\b' + word + r'\b', '', text, flags=re.IGNORECASE)

    # 3. Clean repetitive lists (Solves "A, A, B" or "A - A - B")
    text = deduplicate_by_splitter(text, "-")
    text = deduplicate_by_splitter(text, ",")

    # 4. Clean repeated phrases without delimiters
    text = deduplicate_phrases(text)

    # 5. Remove individual stuttering words (e.g., "word word word")
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)

    # 6. Final cleanup (whitespace and punctuation)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip(" .,-")

    # 7. Ensure capitalization
    if text:
        text = text[0].upper() + text[1:]

    return text