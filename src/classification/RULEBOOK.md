# Classification Rulebook

This document describes the rule-based validation layer that runs immediately
after LLM classification (Stage 2) and before a document is trusted enough to
proceed to chunking/indexing. The rules are implemented in
[`rule_validator.py`](./rule_validator.py); this document is the
human-readable reference for what each rule checks and why it exists.

## Why rule-based validation exists

The LLM classifier has no silent fallback for outright *failure* — an
unreachable model produces an explicit `UNCLASSIFIED` record, never a guess.
But it had no defense against a call that *succeeds* while returning
something wrong: a legal contract tagged `Public`, an invented department
name, a confident-sounding classification on a document the model actually
misread. LLMs are fluent even when incorrect, and a wrong-but-confident
classification is invisible until it causes real harm — for example, a
`Restricted` document indexed as `Public` and freely retrievable by anyone.

Every rule below is deterministic and explainable — no LLM is involved in
validation itself. A rule either fires or it doesn't, and it always says
exactly why.

## Severity levels

- **error** — sets `needs_review = True` on the document. The document
  should not proceed to chunking/indexing until a human confirms or corrects
  the classification.
- **warning** — flagged in the report, but does not block the document.
  Worth a human glance, not worth halting the pipeline over.

## The rules

| Rule | Severity | Checks |
|---|---|---|
| `classification_failed` | error | The LLM call failed outright (`department == "UNCLASSIFIED"` or the classifier reports `llm_failed`). No other rule can meaningfully evaluate a record with no real classification. |
| `unknown_department` | error | `department` is not one of the recognized company departments: HR, Finance, Legal, IT, General. Catches the LLM inventing a department that doesn't exist here. |
| `unknown_doc_type` | error | `doc_type` is not one of: Policy, Invoice, Report, Contract, Data Table, Document, Email. |
| `unknown_sensitivity` | error | `sensitivity` is not one of: Public, Internal, Confidential, Restricted. This matters doubly because `access_control.py` fails closed on an unrecognized label — an invalid label here silently makes the document maximally restricted at retrieval time without anyone deciding that on purpose. |
| `unexpected_language` | warning | `language` is outside the corpus's expected set (EN, FR, AR). Either a genuinely new language appeared, or the classifier misread the document — either way, worth a look. |
| `missing_confidence` | error | No usable numeric confidence score was returned. |
| `low_confidence` | warning | Self-reported confidence is below `0.5`. Doesn't block the document, but flags it as worth a second look before being fully trusted. |
| `contract_under_classified` | error | `doc_type == "Contract"` but `sensitivity` is below `Confidential`. Legally binding documents are a real business risk if under-classified — this is treated as an error, not a warning, regardless of what the LLM's confidence was. |

## How to change these rules

Canonical value sets (departments, doc types, sensitivity labels, languages)
and the confidence threshold are constants at the top of `rule_validator.py`
— update them there, not in the prompt in `classify.py` alone, or the two
will drift out of sync (which is itself exactly the kind of thing
`unknown_department` / `unknown_doc_type` are designed to catch if it
happens).

Sensitivity labels are imported from `access_control.py`'s
`SENSITIVITY_LEVELS` rather than duplicated here, so classification and
retrieval-time access control can never disagree on what the four levels are.

## Running validation

Validation runs automatically as part of every `classify.py` run — see the
`validation` field in each `.classified.json` output, and the
`needs_review` count in the run summary.

To re-validate an already-classified batch without re-running the LLM (e.g.
after updating a rule):

```bash
python -m src.classification.rule_validator --metadata classified_metadata.json
```

This writes a full report to `classified_metadata.validation.json` and
prints a summary of anything flagged for review.

## Sensitivity classification rubric

- Public: safe for anyone, no risk if leaked (announcements, published policies)
- Internal: employee-only, low risk if leaked (status reports, SOPs, no PII)
- Confidential: real harm if leaked — financial data, PII, vendor/client details, strategic plans
- Restricted: legally binding or high-liability — signed contracts, legal agreements
