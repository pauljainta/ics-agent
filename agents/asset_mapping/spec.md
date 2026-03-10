# AssetMapping Agent — Spec

## Name
AssetMappingAgent

## Purpose / role
Discover ICS assets, normalize names, and produce a knowledge graph (KG).

## Inputs
- Site PDF manuals
- PLC configs (CSV/IEC)
- Network scan output (nmap)
- Historian metadata (CSV/time-series)

## Outputs
- KG JSON (schema: nodes, edges, properties)
- Normalized asset list (CSV)
- Confidence scores per extraction

## Tools required
- PDF parser (pdfminer or tika)
- Ontology retriever (RAG)
- CSV/IEC parser
- Knowledge graph serializer (JSON-LD)

## Actions
- `parse_documents()`
- `extract_assets()`
- `normalize_assets()`
- `produce_kg()`
- `propose_manual_review()`

## Constraints & safety
- READ-ONLY by default — no actuation.
- If ambiguous, flag for human review with evidence.

## Autonomy level
- Start at L1 (assistive). Human approves final KG.

## Evaluation metrics
- Precision / recall vs ground-truth mapping
- Human-review false positive rate
- Time to generate KG

## Example API
- `POST /asset_map` with attachments -> returns KG id
- `GET /asset_map/{id}` -> returns KG JSON
