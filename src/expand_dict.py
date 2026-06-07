import re
from typing import List, Dict

# Common diacritic foldings
_DIACRITIC_MAP = {
    'é': '[éeèê]', 'è': '[èeé]', 'ê': '[êeé]',
    'à': '[àaâ]', 'â': '[âa]',
    'ô': '[ôo]', 'ù': '[ùu]',
    'ç': '[çc]',
    'ï': '[ïiî]', 'î': '[îi]',
}

# Optional title prefix for teachers
_TITLE = r'(pr\.?|prof\.?|dr\.?|m\.?|mr\.?)?\s*'

def expand_name(canonical: str) -> List[Dict[str, str]]:
    """Generate fuzzy regex patterns for a teacher's name."""
    canonical = canonical.strip()
    patterns = []

    # Separate title from core name
    title = ''
    for t in ['Pr.', 'Prof.', 'Dr.', 'M.', 'Mr.']:
        if canonical.lower().startswith(t.lower()):
            title = canonical[:len(t)].strip()
            canonical = canonical[len(t):].strip()
            break

    parts = canonical.split()
    # Apply diacritic folding to each word
    flex_parts = []
    for p in parts:
        w = p
        for k, v in _DIACRITIC_MAP.items():
            w = w.replace(k, v)
        flex_parts.append(re.escape(w))
    base_regex = r'\s+'.join(flex_parts)

    # Full name (title optional)
    patterns.append({
        "pattern": f"(?i){_TITLE}{base_regex}",
        "value": (title + ' ' + canonical).strip()
    })

    # Last name only
    if len(parts) > 1:
        patterns.append({
            "pattern": f"(?i){_TITLE}{flex_parts[-1]}",
            "value": (title + ' ' + canonical).strip()
        })

    # Initial + last name (if first part looks like an initial)
    if len(parts) > 1 and re.match(r'^[A-Za-z]\.?$', parts[0]):
        init = re.escape(parts[0].rstrip('.')) + r'\.?'
        patterns.append({
            "pattern": f"(?i){_TITLE}{init}\\s+{flex_parts[-1]}",
            "value": (title + ' ' + canonical).strip()
        })

    # Deduplicate
    seen = set()
    uniq = []
    for p in patterns:
        if p['pattern'] not in seen:
            seen.add(p['pattern'])
            uniq.append(p)
    return uniq

def expand_module(canonical: str) -> List[Dict[str, str]]:
    """Fuzzy patterns for module names (handles diacritics and missing words)."""
    canonical = canonical.strip()
    patterns = []
    words = canonical.split()

    # Fold diacritics
    flex_words = []
    for w in words:
        for k, v in _DIACRITIC_MAP.items():
            w = w.replace(k, v)
        flex_words.append(re.escape(w))
    full_regex = r'\s+'.join(flex_words)
    patterns.append({"pattern": f"(?i){full_regex}", "value": canonical})

    # Without last word if it's a common descriptor (like "avancées")
    if len(words) > 1:
        without = r'\s+'.join(flex_words[:-1])
        patterns.append({"pattern": f"(?i){without}", "value": canonical})

    return patterns

def expand_filiere(canonical: str) -> List[Dict[str, str]]:
    """Patterns for filière, allowing omitted 'année'."""
    canonical = canonical.strip()
    patterns = []
    words = canonical.split()
    flex_words = []
    for w in words:
        for k, v in _DIACRITIC_MAP.items():
            w = w.replace(k, v)
        flex_words.append(re.escape(w))
    full_regex = r'\s+'.join(flex_words)
    patterns.append({"pattern": f"(?i){full_regex}", "value": canonical})

    # Without the last word if it's 'année'
    if words[-1].lower() == 'année':
        without = r'\s+'.join(flex_words[:-1])
        patterns.append({"pattern": f"(?i){without}", "value": canonical})
    return patterns

def expand_year(canonical: str) -> List[Dict[str, str]]:
    """Academic year patterns: 2024-2025, 2024 2025, 24-25, 24 25."""
    m = re.match(r'(\d{4})\s*[-/]\s*(\d{4})', canonical)
    if not m:
        return [{"pattern": re.escape(canonical), "value": canonical}]
    s, e = m.groups()
    s2, e2 = s[2:], e[2:]
    patterns = [
        {"pattern": f"{s}\\s*[-/]\\s*{e}", "value": canonical},
        {"pattern": f"{s}\\s+{e}", "value": canonical},
        {"pattern": f"{s2}\\s*[-/]\\s*{e2}", "value": canonical},
        {"pattern": f"{s2}\\s+{e2}", "value": canonical},
    ]
    return patterns