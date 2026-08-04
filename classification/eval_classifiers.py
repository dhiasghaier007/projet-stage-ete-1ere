#!/usr/bin/env python3
"""
Evaluation script for LLM classification accuracy (Stage 2 QA).

Runs the current LLM-only classifier on a labeled test dataset and reports
accuracy per field.

Run:
    python eval_classifiers.py --verbose
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Import classifier from classify.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from classify import classify_document_litellm


# ============================================================================
# LABELED TEST DATASET
# ============================================================================

LABELED_TEST_SET = [
    {
        "filename": "hr_policy_2026.txt",
        "content": """
            HR POLICY: Remote Work Guidelines - FOR INTERNAL USE ONLY
            Effective Date: January 1, 2026
            
            All full-time employees are eligible for remote work arrangements.
            Employees must have a quiet workspace and high-speed internet connection.
            Default schedule: 3 days remote, 2 days in office per week.
            
            This is an internal policy document for company use.
        """,
        "expected": {
            "department": "HR",
            "doc_type": "Policy",
            "language": "EN",
            "sensitivity": "Internal"
        }
    },
    {
        "filename": "q3_financial_report.html",
        "content": """
            Q3 2026 Financial Statement - INTERNAL
            Department: Finance
            
            The Q3 2026 financial statement shows strong revenue growth of 15% YoY.
            
            Revenue Breakdown:
            - Product Sales: $2.5M
            - Services: $1.2M
            - Consulting: $800K
            
            Operating expenses totaled $1.8M, an increase from $1.5M in Q2.
            
            For internal use only by Finance and Executive teams.
        """,
        "expected": {
            "department": "Finance",
            "doc_type": "Financial Report",
            "language": "EN",
            "sensitivity": "Internal"
        }
    },
    {
        "filename": "invoice_2026_07_15.pdf",
        "content": """
            Invoice #INV-2026-07-15
            
            Bill To: Acme Corporation
            Invoice Date: July 15, 2026
            Amount Due: $25,000.00
            
            Services rendered: IT consulting and infrastructure setup
            Payment Terms: Net 30 days
        """,
        "expected": {
            "department": "Finance",
            "doc_type": "Invoice",
            "language": "EN",
            "sensitivity": "Internal"
        }
    },
    {
        "filename": "employee_data.csv",
        "content": """
            name,department,salary,start_date
            Alice Johnson,HR,85000,2020-01-15
            Bob Smith,Finance,95000,2019-06-20
            Carol Davis,IT,90000,2021-03-10
            
            CONFIDENTIAL SALARY DATA - Do not share externally.
        """,
        "expected": {
            "department": "General",
            "doc_type": "Data Table",
            "language": "EN",
            "sensitivity": "Confidential"
        }
    },
    {
        "filename": "service_agreement_confidential.docx",
        "content": """
            SERVICE AGREEMENT - CONFIDENTIAL
            
            This legal agreement outlines the terms and conditions between
            Provider and Client. This document contains confidential information
            and proprietary terms. Do not share without written consent.
            
            Liability: Provider shall not be liable for any indirect damages.
        """,
        "expected": {
            "department": "Legal",
            "doc_type": "Contract",
            "language": "EN",
            "sensitivity": "Confidential"
        }
    },
    {
        "filename": "bonus_calculations.xlsx",
        "content": """
            Q3 2026 BONUS CALCULATIONS - Internal Use Only
            
            Employee Bonuses:
            - Senior Staff: 10% of salary
            - Mid-level: 7% of salary
            - Junior: 5% of salary
            
            Total budget allocated: $500K
            Expected payout: Q4 2026
            
            Finance and HR Department only.
        """,
        "expected": {
            "department": "Finance",
            "doc_type": "Financial Report",
            "language": "EN",
            "sensitivity": "Internal"
        }
    }
]


# ============================================================================
# EVALUATION LOGIC
# ============================================================================

def evaluate_classifier(classifier_name: str, classifier_func, test_set: List[Dict], verbose: bool = False) -> Dict:
    """
    Run a classifier against the test set and compute metrics.
    
    Returns accuracy per field + confusion matrix.
    """
    results = {
        "classifier": classifier_name,
        "total": len(test_set),
        "correct": 0,
        "field_accuracy": {
            "department": {"correct": 0, "total": 0},
            "doc_type": {"correct": 0, "total": 0},
            "language": {"correct": 0, "total": 0},
            "sensitivity": {"correct": 0, "total": 0},
        },
        "predictions": [],
        "errors": []
    }
    
    for test_case in test_set:
        filename = test_case["filename"]
        content = test_case["content"]
        expected = test_case["expected"]
        
        try:
            # Run classifier
            prediction = classifier_func(content, filename)
            
            # Track prediction
            prediction_record = {
                "filename": filename,
                "expected": expected,
                "predicted": {
                    "department": prediction.get("department"),
                    "doc_type": prediction.get("doc_type"),
                    "language": prediction.get("language"),
                    "sensitivity": prediction.get("sensitivity"),
                },
                "confidence": prediction.get("confidence", 0),
            }
            
            # Check if fully correct
            all_match = all(
                prediction.get(field) == expected.get(field)
                for field in ["department", "doc_type", "language", "sensitivity"]
            )
            
            if all_match:
                results["correct"] += 1
            
            # Compute per-field accuracy
            for field in ["department", "doc_type", "language", "sensitivity"]:
                results["field_accuracy"][field]["total"] += 1
                if prediction.get(field) == expected.get(field):
                    results["field_accuracy"][field]["correct"] += 1
                else:
                    prediction_record["error"] = field
                    results["errors"].append({
                        "file": filename,
                        "field": field,
                        "expected": expected.get(field),
                        "predicted": prediction.get(field),
                    })
            
            results["predictions"].append(prediction_record)
            
            if verbose:
                status = "✅" if all_match else "❌"
                print(f"  {status} {filename:40s} | Dept: {str(prediction.get('department') or 'None'):10s} | Type: {str(prediction.get('doc_type') or 'None'):15s}")
                
        except Exception as e:
            results["errors"].append({
                "file": filename,
                "error": str(e),
            })
            if verbose:
                print(f"  ❌ {filename:40s} | ERROR: {e}")
    
    return results


def format_report(results_litellm: Dict) -> str:
    """Format evaluation results as a readable report."""

    report = []
    report.append("\n" + "=" * 90)
    report.append("LLM CLASSIFICATION ACCURACY EVALUATION".center(90))
    report.append("=" * 90 + "\n")

    report.append("🤖 LITELLM CLASSIFIER")
    report.append("-" * 90)
    accuracy_llm = (results_litellm["correct"] / results_litellm["total"]) * 100
    report.append(f"Overall Accuracy: {accuracy_llm:.1f}% ({results_litellm['correct']}/{results_litellm['total']})\n")

    for field, metrics in results_litellm["field_accuracy"].items():
        field_accuracy = (metrics["correct"] / metrics["total"] * 100) if metrics["total"] > 0 else 0
        report.append(f"  {field.capitalize():15s}: {field_accuracy:5.1f}% ({metrics['correct']}/{metrics['total']})")

    report.append(f"\nConfidence (avg): {sum(p['confidence'] for p in results_litellm['predictions']) / len(results_litellm['predictions']):.2f}")
    report.append(f"Confidence (min): {min(p['confidence'] for p in results_litellm['predictions']):.2f}")
    report.append(f"Confidence (max): {max(p['confidence'] for p in results_litellm['predictions']):.2f}")

    report.append("\n" + "=" * 90)
    report.append("⚠️  MISCLASSIFICATIONS")
    report.append("-" * 90)

    if results_litellm["errors"]:
        for error in results_litellm["errors"][:5]:
            if "field" in error:
                report.append(f"  {error['file']:40s} | {error['field']:12s}: expected '{error['expected']}' got '{error['predicted']}'")
            else:
                report.append(f"  {error['file']:40s} | ERROR: {error.get('error', 'Unknown')}")

        if len(results_litellm["errors"]) > 5:
            report.append(f"  ... and {len(results_litellm['errors']) - 5} more errors")
    else:
        report.append("  None! Perfect score.")

    report.append("\n" + "=" * 90)

    return "\n".join(report)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate the LLM classifier")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output per document")
    args = parser.parse_args()

    print(f"\n🧪 LLM Classification Evaluation")
    print(f"Test set size: {len(LABELED_TEST_SET)} documents")
    print(f"Timestamp: {datetime.now().isoformat()}\n")

    print("Running LiteLLM classifier...")
    results_litellm = evaluate_classifier(
        "litellm",
        classify_document_litellm,
        LABELED_TEST_SET,
        verbose=args.verbose
    )

    report = format_report(results_litellm)
    print(report)

    report_file = Path("classification_eval_report.txt")
    report_file.write_text(report)
    print(f"✅ Report saved to {report_file}\n")


if __name__ == "__main__":
    main()
