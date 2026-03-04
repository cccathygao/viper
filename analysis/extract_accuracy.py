import argparse
import json
import os
import re


DEFAULT_LOG_PATH = "intermediate_results.txt"
DEFAULT_OUT_PATH = "accuracy.json"

SAMPLE_RESULT_RE = re.compile(
    r"^\s*Sample (\S+): (correct|incorrect) \(pred=(.*), gt=(.*)\)\s*$"
)
FINAL_ACCURACY_RE = re.compile(r"^Final accuracy:\s*([\d.]+)\s*$")


def main():
    parser = argparse.ArgumentParser(description="Extract accuracy from intermediate_results.txt")
    parser.add_argument("--dir", type=str, required=True, default=None, help="Directory: output path becomes dir/accuracy.json")
    args = parser.parse_args()

    log_path = os.path.join(args.dir, DEFAULT_LOG_PATH) if not os.path.isabs(DEFAULT_LOG_PATH) else DEFAULT_LOG_PATH
    out_path = os.path.join(args.dir, DEFAULT_OUT_PATH) if not os.path.isabs(DEFAULT_OUT_PATH) else DEFAULT_OUT_PATH

    if not os.path.isfile(log_path):
        raise SystemExit(f"Log file not found: {log_path}")

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    per_sample = []
    final_accuracy = None

    for line in lines:
        m = SAMPLE_RESULT_RE.match(line)
        if m:
            sample_id, status, pred_raw, gt_raw = m.groups()
            correct = status == "correct"
            pred = pred_raw.strip()
            gt = gt_raw.strip()
            per_sample.append({
                "id": sample_id,
                "correct": correct,
                "pred": pred,
                "gt": gt,
            })
            continue

        mf = FINAL_ACCURACY_RE.match(line)
        if mf:
            final_accuracy = float(mf.group(1))
            continue

    n_correct = sum(1 for p in per_sample if p["correct"])
    n_total = len(per_sample)

    metadata = {
        "log_path": log_path,
        "total_samples": n_total,
        "correct_count": n_correct,
        "overall_accuracy": final_accuracy,
    }

    data = {"metadata": metadata, "per_sample": per_sample}
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    acc_str = f"{final_accuracy:.4f}" if final_accuracy is not None else "N/A"
    print(f"Wrote accuracy for {n_total} samples to {out_path} (overall_accuracy={acc_str})")


if __name__ == "__main__":
    main()
