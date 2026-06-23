### This script was used to generate the loss curves for the SAMwise paper. 
### It reads CSV files containing training loss data and plots them for comparison. 
### The csv files were extracted via the tensorboard logs web view. 


from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_loss_csv(csv_path: Path) -> tuple[list[float], list[float]]:
    steps: list[float] = []
    values: list[float] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "Step" not in reader.fieldnames or "Value" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path.name} does not contain expected columns: Step and Value"
            )

        for row in reader:
            try:
                steps.append(float(row["Step"]))
                values.append(float(row["Value"]))
            except (TypeError, ValueError):
                continue

    return steps, values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training loss curves from CSV files in custom/data/plots."
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "plots",
        help="Directory containing CSV files (default: custom/data/plots).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, shows the plot window.",
    )
    args = parser.parse_args()

    selected_curves = [
        (
            args.plots_dir
            / "sam2.1_meta_SAMwise_1.2.1_nonlin_adapter_stage_2_lr_1e-4.yaml_..csv",
            "Adapter only alignment",
        ),
        (
            args.plots_dir
            / "sam2.1_meta_SAMwise_reinit_stage_1_pretrain_non_linear_lr_1e-4.yaml_..csv",
            "Reinitialized SAM parts",
        ),
    ]

    if len(selected_curves) < 2:
        raise SystemExit(
            f"Expected at least 2 selected curves, found {len(selected_curves)}"
        )

    missing_files = [str(path) for path, _ in selected_curves if not path.exists()]
    if missing_files:
        missing = "\n".join(missing_files)
        raise SystemExit(f"These CSV files were not found:\n{missing}")

    plt.figure(figsize=(10, 6))

    for csv_file, label in selected_curves:
        steps, values = read_loss_csv(csv_file)
        if steps and values:
            plt.plot(steps, values, label=label)

    plt.title("Training Loss Curves")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=200)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
