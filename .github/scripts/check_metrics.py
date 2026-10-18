"""CI quality gate: compare the freshly evaluated artifact against the champion."""

import json
import sys

from arabic_sentiment.config import CHAMPION_PATH, REPORTS_DIR

TOLERANCE = 0.01


def main() -> None:
    with open(CHAMPION_PATH) as f:
        champion = json.load(f)
    with open(REPORTS_DIR / "eval_metrics.json") as f:
        current = json.load(f)

    threshold = champion["f1_macro"] - TOLERANCE
    print(f"Champion f1_macro : {champion['f1_macro']:.4f}")
    print(f"Current  f1_macro : {current['f1_macro']:.4f}")
    print(f"Tolerance         : {TOLERANCE}")
    print(f"Fail below        : {threshold:.4f}")

    if current["n_samples"] != champion["n_samples"]:
        print(
            f"WARNING: eval sample count changed "
            f"({champion['n_samples']} → {current['n_samples']})"
        )

    if current["f1_macro"] < threshold:
        print("FATAL: model quality degraded below champion - tolerance")
        sys.exit(1)

    print("Quality gate passed")


if __name__ == "__main__":
    main()