# Policy Rule Extractor

Extracts structured, machine-readable rules from insurance/contract policy documents.
Supports **LLM-based extraction** (Claude claude-sonnet-4-20250514, best quality) and a **regex fallback** (no API key required).

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt        
```

The only runtime dependency is `anthropic`. The regex path has zero dependencies beyond the Python standard library.

### 2a — Run with Claude (recommended)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python extract.py
```

### 2b — Run without an API key (regex fallback)

```bash
python extract.py --no-llm
```

### 3 — Run on your own document

```bash
python extract.py --file path/to/policy.txt --doc-id my_policy_v2
```

### Output

All JSON files are written to `./output/` (override with `--output-dir`).

---

## Approach and Design Decisions

### Why an LLM as the primary extractor?

The three samples illustrate exactly the kind of variability that makes rule-parsing hard:

| Challenge | Example | Regex handles it? |
|-----------|---------|-------------------|
| Simple threshold | "credit score of at least 680" | ✅ |
| Conditional threshold | "40%, or 45% if co-signer…" | ⚠️ partial |
| Nested exception | "unless enrolled in program, in which case $25k" | ❌ |
| Fee formula | "equal to 3 months of the annual service fee" | ❌ |
| Implicit operator | "weighted average… shall not be less than 720" | ⚠️ fragile |

A regex pipeline can pick up numbers and simple comparators reliably, but it breaks on multi-hop logic ("unless X, in which case Y") and formula-based thresholds ("3 months of the annual service fee"). An LLM reads the clause as a human would and fills in the semantics correctly.

**Why Claude claude-sonnet-4-20250514 specifically?**
- Nested conditionals require genuine multi-hop reasoning; smaller/cheaper models make substitution errors (e.g. confusing the primary threshold with the exception threshold).
- The structured-JSON output instruction is long; instruction-following accuracy matters here.
- Sonnet hits the right cost/quality inflection point for a batch pipeline over hundreds of documents.

### Why keep a regex fallback?

1. **Resilience** — LLM APIs can be unavailable or rate-limited. A rules-based fallback ensures the pipeline always produces *some* output.
2. **Cost control** — For documents with simple, predictable formats (fee schedules), regex output quality approaches LLM quality at zero API cost.
3. **Reviewers without keys** — The repo is runnable by anyone with `pip install -r requirements.txt`.

### Schema design rationale

```jsonc
{
  "rule_id":           "deterministic key — section + clause label",
  "rule_type":         "routes to the right evaluator without re-parsing text",
  "subject":           "what entity the rule governs (indexable)",
  "metric":            "the measurable attribute",
  "operator":          "machine-evaluable comparator (gte/lte/eq…)",
  "threshold":         "primary numeric value",
  "threshold_unit":    "USD / % / days / months",
  "condition":         "prerequisite that activates this rule",
  "exceptions":        ["relaxation conditions"],
  "alternate_threshold":"value that applies under the exception",
  "applies_to":        "applicant | policy | portfolio | fee_trigger",
  "enforcement":       "hard_limit | soft_limit | fee | informational",
  "raw_text":          "verbatim source clause — provenance/audit trail"
}
```

**Key decisions:**
- `operator` uses a fixed enum (`gte`, `lte`, `gt`, `lt`, `eq`, `neq`, `between`, `percentage_lte`, `percentage_gte`, `fixed`). A downstream rules-engine can evaluate `applicant.credit_score >= rule.threshold` programmatically without any further NLP.
- Separating `condition` from `exceptions` makes the two-threshold case (clause (d): 40% primary / 45% with co-signer) unambiguous. `alternate_threshold` holds the relaxed value.
- `raw_text` is mandatory for every rule — it creates an audit trail and lets a reviewer diff the extracted structure against the source document without re-reading the whole PDF.
- `applies_to` scoping (`applicant` vs `portfolio`) is critical: Section 7.3 rules apply to an aggregate metric; Section 5.1 rules apply per-applicant. Mixing them would cause incorrect evaluations.

---

## What the LLM Does Well vs. Where It Struggles

### Strengths (with examples from the samples)

| Strength | Example |
|----------|---------|
| **Nested conditional logic** | Clause (g): "unless enrolled in approved assistance program, in which case the minimum income shall be $25,000" — the LLM correctly sets `threshold=$35,000`, `exceptions=[enrolled in program]`, `alternate_threshold=$25,000`. Regex conflates the two numbers. |
| **Semantic subject labelling** | Clause (d) is about DTI ratio, not credit score — even though it mentions "credit score above 750" in the exception. The LLM sets `subject=debt_to_income_ratio`; a keyword regex would misclassify it. |
| **Formula thresholds** | Clause (d) in sample 3: "equal to 3 months of the annual service fee" is not a scalar — the LLM can flag it as a formula reference rather than inserting a wrong number. |
| **Operator inference from negations** | "shall not be *less than* 720" maps to `operator=gte`, not `lte`. The LLM handles the double-negation; regex requires a bespoke rule for every phrasing. |

### Weaknesses (honest limitations)

| Weakness | Example / Mitigation |
|----------|---------------------|
| **Hallucinated thresholds** | On rare occasions the model will "round" a value (e.g., "680" → "680.0") or infer a threshold not in the text. Mitigation: require `raw_text` in every output and diff numeric values against it. |
| **Inconsistent operator mapping** | Fee clauses like "subject to a minimum of $25 and a maximum of $500" involve *two* operators on the same clause. The LLM sometimes picks only one. Mitigation: post-process fee rules to detect "minimum/maximum" patterns and split into sub-rules. |
| **Ambiguous `applies_to`** | Clause (c) in sample 1 is simultaneously an applicant rule (credit score ≥ 680) and a portfolio rule (≤ 15% portfolio). The LLM may set one and miss the other. Mitigation: schema allows dual entries — prompt can instruct the model to emit two rules for such clauses. |
| **Long documents → context length** | A 50-page policy PDF may exceed context. Mitigation: chunk by section, extract per-section, merge at the end. |

---

## Evaluating Accuracy at Scale (500 Documents)

### Ground-truth creation

1. **Expert annotation** — Have a domain expert (underwriter or compliance officer) read 30–50 documents and hand-label rules using the schema. This becomes the gold set.
2. **Dual-annotation** — For the gold set, have a second expert annotate independently and compute inter-annotator agreement (Cohen's κ). High κ (> 0.8) confirms the schema is well-defined; low κ reveals ambiguous schema fields that need clarification.

### Metrics per field

```
Rule-level precision  = correctly extracted rules / total extracted rules
Rule-level recall     = correctly extracted rules / total ground-truth rules
Field-level accuracy  = fraction of fields matching gold, per rule
```

For numeric fields (`threshold`, `alternate_threshold`) use **exact-match after normalisation** (strip `$`, `,`, trailing `.`).

For text fields (`condition`, `exceptions`, `raw_text`) use **token-level F1** (ROUGE-1 or simple overlap) — verbatim match is too strict.

### Automated regression harness

```python
# pseudo-code for an eval run
for doc_id, gold_rules in gold_set.items():
    extracted = pipeline.run(doc_id)
    for g_rule in gold_rules:
        match = best_match(g_rule, extracted.rules)   # align by rule_id or raw_text similarity
        score_fields(g_rule, match)                   # per-field accuracy
report_aggregate_precision_recall()
```

### Error taxonomy to track

- **Missing rules** (recall failures) — most common in complex multi-part clauses
- **Threshold confusion** — primary vs. exception value swapped
- **Wrong `applies_to` scope** — applicant vs. portfolio mix-up
- **Operator inversion** — gte vs. lte on negated phrases

### Continuous monitoring in production

- Log every extraction alongside the source sentence span.
- Sample 1–2% for human review weekly; feed disagreements back as few-shot examples in the system prompt.
- Alert if field-level accuracy drops > 3 pp week-over-week.

---

## Production Roadmap

| Area | What to change |
|------|---------------|
| **Input** | Replace raw text with a PDF extraction layer (PyMuPDF or AWS Textract for scanned PDFs). Preserve page numbers and section hierarchy. |
| **Chunking** | Split long documents by detected section headers before sending to the LLM. Merge results post-hoc. |
| **Prompt** | Move to a structured-output call with a JSON Schema passed via the `response_format` parameter (Anthropic tool-use / structured outputs) to eliminate JSON-parse failures. |
| **Validation** | Add Pydantic models for the output schema; raise on invalid enum values or missing required fields before writing to storage. |
| **Cost control** | Route simple documents (fee schedules, single-table formats) to the regex extractor automatically. Only invoke the LLM when structural complexity is detected (e.g., clause contains "unless", "provided that", "in which case"). |
| **Parallelism** | Use `asyncio` + `anthropic.AsyncAnthropic` to process documents concurrently; a batch of 100 documents can be extracted in under 2 minutes instead of ~15. |
| **Storage** | Write extracted rules to a relational schema (PostgreSQL): `documents` table + `rules` table with a FK. Enables SQL-based auditing ("show me all rules with `applies_to=portfolio` across all documents"). |
| **Observability** | Emit a structured log line per rule with `{doc_id, rule_id, model_version, latency_ms, token_count}`. Feed into a dashboard to catch cost spikes and accuracy regressions. |
| **Human-in-the-loop** | Surface low-confidence rules (those where the model's raw text span contains keywords like "formula", "as applicable", "to be determined") for analyst review before the rules enter the production database. |

---

## Project Structure

```
policy-rule-extractor/
├── extract.py          # main pipeline (LLM + regex fallback)
├── requirements.txt    # anthropic>=0.25.0
├── README.md
└── output/
    ├── sample1_eligibility.json
    ├── sample2_concentration.json
    └── sample3_fees.json
```
