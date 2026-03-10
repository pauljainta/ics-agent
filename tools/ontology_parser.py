"""Ontology parser placeholders for ICS asset normalization."""


def parse_ontology(text: str) -> dict:
    """Parse ontology text into a minimal structured representation."""
    terms = [line.strip() for line in text.splitlines() if line.strip()]
    return {"terms": terms, "count": len(terms)}
