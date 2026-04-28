#!/usr/bin/env python
"""
Model Governance Audit CLI

Run fairness and bias audits on ClimateVision models.

Usage:
    python scripts/audit_model.py --model models/best_model.pth --regions amazon,congo
    python scripts/audit_model.py --model models/best_model.pth --metric demographic_parity
    python scripts/audit_model.py --model models/best_model.pth --ci-gate
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climatevision.governance import (
    run_bias_audit,
    check_fairness_gate,
    SUPPORTED_REGIONS,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run bias and fairness audits on ClimateVision models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full audit
  python scripts/audit_model.py --model models/best_model.pth

  # Audit specific regions
  python scripts/audit_model.py --model models/best_model.pth --regions amazon,congo

  # Use different fairness metric
  python scripts/audit_model.py --model models/best_model.pth --metric demographic_parity

  # CI gate mode (exit 1 if fails)
  python scripts/audit_model.py --model models/best_model.pth --ci-gate --threshold 0.85
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--regions",
        type=str,
        default="amazon,congo,southeast_asia",
        help="Comma-separated list of regions to audit",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["demographic_parity", "equalized_odds", "predictive_parity"],
        default="equalized_odds",
        help="Fairness metric to use",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum fairness score to pass",
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        default="deforestation",
        help="Analysis type (deforestation, ice_melting, flooding)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for JSON report",
    )
    parser.add_argument(
        "--ci-gate",
        action="store_true",
        help="Run in CI gate mode (exit 1 if fails)",
    )
    parser.add_argument(
        "--list-regions",
        action="store_true",
        help="List supported regions and exit",
    )

    args = parser.parse_args()

    # List regions mode
    if args.list_regions:
        print("Supported regions:")
        for key, info in SUPPORTED_REGIONS.items():
            print(f"  {key}: {info['name']}")
            print(f"    bbox: {info['bbox']}")
            print(f"    {info['description']}")
            print()
        return 0

    # Parse regions
    regions = [r.strip() for r in args.regions.split(",")]

    print(f"Running bias audit on: {args.model}")
    print(f"Regions: {regions}")
    print(f"Metric: {args.metric}")
    print(f"Threshold: {args.threshold}")
    print()

    # CI gate mode
    if args.ci_gate:
        passed = check_fairness_gate(
            model_path=args.model,
            regions=regions,
            threshold=args.threshold,
        )
        if passed:
            print("\n✅ FAIRNESS GATE PASSED")
            return 0
        else:
            print("\n❌ FAIRNESS GATE FAILED")
            return 1

    # Full audit mode
    result = run_bias_audit(
        model_path=args.model,
        regions=regions,
        metric=args.metric,
        threshold=args.threshold,
        analysis_type=args.analysis_type,
    )

    # Print results
    print("=" * 60)
    print("BIAS AUDIT RESULTS")
    print("=" * 60)
    print(f"Fairness Score: {result['score']:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Status: {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
    print()

    if result["disparity_regions"]:
        print(f"Disparity detected in: {', '.join(result['disparity_regions'])}")
        print()

    print("Per-Region Metrics:")
    print("-" * 60)
    for metrics in result["region_metrics"]:
        print(f"  {metrics['region_name']} ({metrics['region']}):")
        print(f"    IoU: {metrics['iou']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print()

    print("Recommendations:")
    for rec in result["recommendations"]:
        print(f"  • {rec}")

    print()
    print(f"Report saved to: {result['report_path']}")

    # Save to custom output if specified
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"JSON output saved to: {args.output}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
