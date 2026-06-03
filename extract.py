
import os
import re
import json
import argparse
import textwrap
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False



SAMPLE_DOCUMENTS = {
    "sample1_eligibility": """SECTION 5.1 ELIGIBILITY CRITERIA:

(a) Each applicant must have a credit score of at least 680.
(b) The maximum coverage amount for any single policy shall not
    exceed $2,000,000.
(c) The applicant must have a credit score of at least 680, provided that
    no more than 15% of the total portfolio value may consist of
    policies for applicants with credit scores between 680 and 700.
(d) The debt-to-income ratio shall not exceed 40%, or 45% if the
    applicant has a co-signer with a credit score above 750.
(e) The applicant must not have any payment more than 30 days overdue
    as of the Review Date.
(f) The applicant must reside in the United States or its territories.
(g) No applicant shall have an annual income below $35,000, unless the
    applicant is enrolled in an approved assistance program, in which
    case the minimum income shall be $25,000.""",

    "sample2_concentration": """SECTION 7.3 PORTFOLIO CONCENTRATION LIMITS:

The following limits shall apply to the overall portfolio:

(i)   No more than 25% of the total portfolio value shall consist
      of policies originating from any single state.
(ii)  No more than 10% of the total portfolio value shall consist
      of policies with a coverage amount exceeding $1,500,000.
(iii) The weighted average credit score across all policies in the
      portfolio shall not be less than 720.
(iv)  No more than 5% of the total portfolio value shall consist
      of policies where the primary applicant is under 25 years of age.
(v)   The weighted average debt-to-income ratio shall not exceed 35%.""",

    "sample3_fees": """SECTION 12.2 FEES AND CHARGES:

(a) Processing Fee: A one-time fee of $500 per application, due upon
    submission.
(b) Annual Service Fee: 0.35% per annum of the outstanding coverage
    amount, payable monthly.
(c) Late Payment Fee: If any scheduled payment is more than 15 days
    overdue, a fee of 2% of the overdue amount shall apply, subject
    to a minimum of $25 and a maximum of $500.
(d) Early Termination Fee: If the policy is cancelled within the first
    24 months, the applicant shall pay a fee equal to 3 months of the
    annual service fee.
(e) Reinstatement Fee: $250 if the policy is reinstated after a lapse
    of more than 60 days.""",

# //my own test file
    "my_test_doc": """SECTION 1.1 MY TEST RULES:

(a) The applicant must be at least 18 years old.
(b) The loan amount shall not exceed $50,000.
(c) The interest rate shall not exceed 15%, or 18% if the applicant has a guarantor.
""",
}


RULE_SCHEMA_DESCRIPTION = """
Return ONLY a JSON object (no markdown, no commentary) with this exact shape:

{
  "document_id": "<string>",
  "section": "<e.g. SECTION 5.1 ELIGIBILITY CRITERIA>",
  "extracted_at": "<ISO-8601 UTC timestamp>",
  "rules": [
    {
      "rule_id": "<section_code>-<clause_letter_or_number>",
      "rule_type": "<one of: threshold | concentration_limit | eligibility | fee | prohibition | residency>",
      "subject": "<what entity or metric this rule governs, e.g. 'credit_score', 'coverage_amount'>",
      "metric": "<the measurable attribute, may equal subject>",
      "operator": "<one of: gte | lte | gt | lt | eq | neq | between | percentage_lte | percentage_gte | fixed>",
      "threshold": "<primary numeric value or string, e.g. 680, '$500', '0.35%'>",
      "threshold_unit": "<unit string, e.g. 'USD', '%', 'days', 'months', or null>",
      "condition": "<prerequisite that activates this rule, or null>",
      "exceptions": ["<list of exception strings, or empty array>"],
      "alternate_threshold": "<relaxed/alternate value when exception applies, or null>",
      "applies_to": "<one of: applicant | policy | portfolio | fee_trigger>",
      "enforcement": "<one of: hard_limit | soft_limit | fee | informational>",
      "raw_text": "<verbatim clause text, single line>"
    }
  ]
}
"""




def extract_with_llm(doc_id: str, text: str, client: "anthropic.Anthropic") -> dict:
    """
    Send the document to Claude claude-sonnet-4-20250514 and parse the JSON response.

    Why Sonnet over a smaller model?
    - Policy text has nested conditionals ("unless… in which case…") that
      require multi-hop reasoning; Sonnet handles these reliably.
    - The structured JSON output requirement needs instruction-following accuracy.
    - Sonnet's cost/quality tradeoff is optimal for this workload.
    """
    prompt = textwrap.dedent(f"""
        You are a precise policy analyst. Extract every rule from the document below.
        {RULE_SCHEMA_DESCRIPTION}

        Rules for extraction:
        1. Every numbered/lettered clause = one rule object.
        2. If a clause has TWO thresholds (normal + exception), set threshold to the
           primary value and alternate_threshold to the exception value.
        3. Preserve all numeric values exactly as written (do not round or convert).
        4. document_id must be exactly: {doc_id}
        5. extracted_at must be the current UTC time in ISO-8601 format.

        DOCUMENT:
        {text}
    """).strip()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)

_CLAUSE_RE = re.compile(
    r"^\s*"
    r"(?P<label>\([a-z]+\)|\([ivxlcdm]+\)|[a-z]\)|[ivxlcdm]+\))"  # (a), (i), a), i) …
    r"\s+"
    r"(?P<body>.+?)(?=\n\s*(?:\([a-z]+\)|\([ivxlcdm]+\)|[a-z]\)|[ivxlcdm]+\))|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)

_SECTION_RE = re.compile(r"SECTION\s+[\d.]+\s+[A-Z ]+", re.IGNORECASE)

_NUMBER_RE = re.compile(r"[\$]?\d[\d,]*(?:\.\d+)?(?:\s*%)?")

_OPERATORS = {
    "at least": "gte",
    "not.*exceed": "lte",
    "shall not be less than": "gte",
    "not less than": "gte",
    "no more than": "lte",
    "above": "gt",
    "below": "lt",
    "under": "lt",
    "over": "gt",
    "more than": "gt",
    "equal to": "eq",
}


def _infer_operator(text: str) -> str:
    t = text.lower()
    for phrase, op in _OPERATORS.items():
        if re.search(phrase, t):
            return op
    return "lte"  # safe default for policy thresholds


def _infer_rule_type(section: str, body: str) -> str:
    s, b = section.lower(), body.lower()
    if "fee" in s or "charge" in s:
        return "fee"
    if "eligib" in s:
        return "eligibility"
    if "concentrat" in s or "portfolio" in s:
        return "concentration_limit"
    if "reside" in b or "resident" in b:
        return "residency"
    if "must not" in b or "shall not" in b:
        return "prohibition"
    return "threshold"


def _infer_subject(body: str) -> str:
    b = body.lower()
    if "credit score" in b:
        return "credit_score"
    if "debt-to-income" in b or "dti" in b:
        return "debt_to_income_ratio"
    if "coverage amount" in b:
        return "coverage_amount"
    if "income" in b:
        return "annual_income"
    if "payment" in b and "overdue" in b:
        return "overdue_payment_days"
    if "portfolio" in b:
        return "portfolio_concentration"
    if "age" in b:
        return "applicant_age"
    if "fee" in b or "charge" in b:
        return "fee_amount"
    return "policy_rule"


def _infer_applies_to(section: str, body: str) -> str:
    s = section.lower()
    if "portfolio" in s:
        return "portfolio"
    if "fee" in s or "charge" in s:
        return "fee_trigger"
    return "applicant"


def _extract_condition(body: str) -> tuple[str | None, list[str]]:
    """Return (condition_string, [exception_strings])."""
    condition = None
    exceptions = []
    b = body.lower()
    # "provided that …" pattern
    m = re.search(r"provided that (.+?)(?:,|$)", body, re.IGNORECASE)
    if m:
        condition = m.group(1).strip()
    # "unless …" pattern
    m2 = re.search(r"unless (.+?)(?:,|$)", body, re.IGNORECASE)
    if m2:
        exceptions.append(m2.group(1).strip())
    # "or X if …" pattern
    m3 = re.search(r"or (.+?) if (.+?)(?:,|\.|$)", body, re.IGNORECASE)
    if m3:
        exceptions.append(f"if {m3.group(2).strip()}: {m3.group(1).strip()}")
    return condition, exceptions


def extract_with_regex(doc_id: str, text: str) -> dict:
    """Pure-regex extractor — no external dependencies."""
    # Detect section header
    m = _SECTION_RE.search(text)
    section = m.group(0).strip() if m else "UNKNOWN SECTION"

    # Strip the header line before clause matching
    body_text = _SECTION_RE.sub("", text).strip()
    # Remove leader line ("The following limits shall apply…")
    body_text = re.sub(r"^[^\n(]+\n", "", body_text).strip()

    rules = []
    for match in _CLAUSE_RE.finditer(body_text):
        label = match.group("label").strip("()")
        clause_body = re.sub(r"\s+", " ", match.group("body")).strip()

        numbers = _NUMBER_RE.findall(clause_body)
        threshold = numbers[0] if numbers else None
        alt_threshold = numbers[1] if len(numbers) > 1 else None

        # Determine unit
        unit = None
        if threshold:
            if "%" in threshold:
                unit = "%"
            elif "$" in threshold:
                unit = "USD"
            elif re.search(r"days?", clause_body, re.IGNORECASE):
                unit = "days"
            elif re.search(r"months?", clause_body, re.IGNORECASE):
                unit = "months"

        condition, exceptions = _extract_condition(clause_body)
        rule_type = _infer_rule_type(section, clause_body)
        subject = _infer_subject(clause_body)

        # Determine enforcement level
        if rule_type == "fee":
            enforcement = "fee"
        elif "shall not" in clause_body.lower() or "must not" in clause_body.lower():
            enforcement = "hard_limit"
        else:
            enforcement = "hard_limit"

        rules.append({
            "rule_id": f"{doc_id}-{label}",
            "rule_type": rule_type,
            "subject": subject,
            "metric": subject,
            "operator": _infer_operator(clause_body),
            "threshold": threshold,
            "threshold_unit": unit,
            "condition": condition,
            "exceptions": exceptions,
            "alternate_threshold": alt_threshold,
            "applies_to": _infer_applies_to(section, clause_body),
            "enforcement": enforcement,
            "raw_text": clause_body,
        })

    return {
        "document_id": doc_id,
        "section": section,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "extraction_method": "regex",
        "rules": rules,
    }



def run_pipeline(
    documents: dict[str, str],
    output_dir: Path,
    use_llm: bool = True,
) -> list[dict]:
    """
    For each document:
      1. Try LLM extraction (if enabled and API key is set).
      2. Fall back to regex extraction on failure or if LLM disabled.
      3. Write <doc_id>.json to output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    client = None

    if use_llm:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("⚠  ANTHROPIC_API_KEY not set — falling back to regex extraction.")
            use_llm = False
        elif not _ANTHROPIC_AVAILABLE:
            print("⚠  anthropic package not installed — falling back to regex extraction.")
            print("   Run: pip install anthropic")
            use_llm = False
        else:
            client = anthropic.Anthropic(api_key=api_key)
            print(f"✓  Using LLM extraction (claude-sonnet-4-20250514)")

    results = []
    for doc_id, text in documents.items():
        print(f"\n{'─'*60}")
        print(f"  Processing: {doc_id}")

        result = None
        method = "llm"

        if use_llm and client:
            try:
                result = extract_with_llm(doc_id, text, client)
                result["extraction_method"] = "llm"
                print(f"  ✓ LLM extracted {len(result['rules'])} rules")
            except Exception as e:
                print(f"  ✗ LLM failed ({e}), falling back to regex")
                method = "regex"

        if result is None:
            result = extract_with_regex(doc_id, text)
            print(f"  ✓ Regex extracted {len(result['rules'])} rules")

        out_path = output_dir / f"{doc_id}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  → Saved: {out_path}")
        results.append(result)

    return results






def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured rules from policy document text."
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to a plain-text policy document (optional; uses built-in samples if omitted)",
    )
    parser.add_argument(
        "--doc-id",
        default="custom_document",
        help="Identifier for the document when --file is used",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force regex-only extraction (no API key required)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for JSON output files (default: ./output)",
    )
    args = parser.parse_args()

    if args.file:
        docs = {args.doc_id: args.file.read_text(encoding="utf-8")}
    else:
        docs = SAMPLE_DOCUMENTS

    results = run_pipeline(
        documents=docs,
        output_dir=args.output_dir,
        use_llm=not args.no_llm,
    )

    print(f"\n{'═'*60}")
    print(f"  Done. {sum(len(r['rules']) for r in results)} rules extracted from {len(results)} document(s).")
    print(f"  Output written to: {args.output_dir}/")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
