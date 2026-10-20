"""RetinaGuard - Unified Master Pipeline Runner.

Consolidates all execution stages into a single, clean, cross-platform runner:
- test: Run pytest test suite
- metadata: Build harmonized metadata & manifests across all 5 datasets
- preprocess: Verify FOV cropping and CLAHE illumination normalization
- eval: Run multi-cohort cross-domain evaluation and benchmark comparison
- figures: Generate t-SNE disentanglement and attribution visualizations
- slides: Rebuild scientific PowerPoint presentation deck
- paper: Compile publication IEEE PDF
- all: Run complete end-to-end verification pipeline

Usage:
    python scripts/run_pipeline.py --stage all
    python scripts/run_pipeline.py --stage eval
    python scripts/run_pipeline.py --stage test
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n========================================================")
    print(f" [RetinaGuard Pipeline] {desc}")
    print(f" Command: {' '.join(cmd)}")
    print(f"========================================================")
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print(f"\n[ERROR] Step failed with return code {res.returncode}: {desc}")
        sys.exit(res.returncode)
    print(f"[OK] Completed: {desc}")


def run_test():
    run_cmd([sys.executable, "-m", "pytest", "tests/", "-q"], "Unit & Clinical Test Suite")


def run_metadata():
    run_cmd([sys.executable, "-m", "src.data.build_metadata"], "Harmonize Master Metadata")
    run_cmd([sys.executable, "-m", "src.data.split_data"], "Generate Zero-Leakage Split Manifests")


def run_preprocess():
    run_cmd([sys.executable, "-m", "src.data.preprocessing"], "Verify FOV & Illumination Normalization")


def run_eval():
    run_cmd([sys.executable, "-m", "src.evaluation.evaluate_cross_domain"], "Cross-Domain Evaluation Across 5 Cohorts")
    run_cmd([sys.executable, "-m", "src.evaluation.benchmark_comparison"], "Head-to-Head Baseline vs. RetinaGuard Benchmark")


def run_figures():
    run_cmd([sys.executable, "-m", "src.evaluation.visualize_disentanglement"], "Generate Disentanglement & Attribution Figures")


def run_slides():
    run_cmd([sys.executable, "slides/build_pptx.py"], "Build 18-Slide PowerPoint Presentation Deck")


def run_paper():
    run_cmd([sys.executable, "-m", "src.utils.generate_paper_pdf"], "Compile IEEE Publication PDF")


def main():
    parser = argparse.ArgumentParser(description="RetinaGuard Master Pipeline Runner")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "test", "metadata", "preprocess", "eval", "figures", "slides", "paper"],
        help="Pipeline stage to execute (default: all)",
    )
    args = parser.parse_args()

    print("====================================================================")
    print(f" RetinaGuard Master Pipeline Runner | Stage: {args.stage.upper()}")
    print("====================================================================")

    if args.stage == "test":
        run_test()
    elif args.stage == "metadata":
        run_metadata()
    elif args.stage == "preprocess":
        run_preprocess()
    elif args.stage == "eval":
        run_eval()
    elif args.stage == "figures":
        run_figures()
    elif args.stage == "slides":
        run_slides()
    elif args.stage == "paper":
        run_paper()
    elif args.stage == "all":
        run_test()
        run_preprocess()
        run_eval()
        run_figures()
        run_slides()
        print("\n====================================================================")
        print(" [SUCCESS] All RetinaGuard pipeline stages executed successfully!")
        print("====================================================================")


if __name__ == "__main__":
    main()
