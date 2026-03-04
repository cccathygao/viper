import argparse
import json
import os
import re
from collections import OrderedDict


LOG_PATH = "intermediate_results.txt"
TOOLCALL_LOGS_OUT_PATH = "toolcall_logs.txt"
TOOLCALL_ANALYSIS_OUT_PATH = "toolcall_analysis.json"


RUN_SAMPLE_RE = re.compile(r"\[DEBUG\] Running sample (\S+)")
TOOLCALL_RE = re.compile(r"\[DEBUG\]\s+[^,]+,\s+(.+?) return:")


def is_toolcall_line(line: str) -> bool:
    """Return True if this debug line is a toolcall 'return' log."""
    if "[DEBUG]" not in line or "return:" not in line:
        return False
    # Skip GLIP wrapper return to avoid duplication
    if "vision_models.py, GLIPModel wrapper return" in line:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect toolcall log and generate toolcall summary from intermediate_results.txt")
    parser.add_argument("--dir", type=str, required=True, default=None, help="Directory: output path becomes dir/toolcall_logs.txt and dir/toolcall_analysis.json")
    args = parser.parse_args()

    log_path = os.path.join(args.dir, LOG_PATH) if args.dir else LOG_PATH
    toolcall_logs_out_path = os.path.join(args.dir, TOOLCALL_LOGS_OUT_PATH) if args.dir else TOOLCALL_LOGS_OUT_PATH
    toolcall_analysis_out_path = os.path.join(args.dir, TOOLCALL_ANALYSIS_OUT_PATH) if args.dir else TOOLCALL_ANALYSIS_OUT_PATH

    if not os.path.isfile(log_path):
        raise SystemExit(f"Log file not found: {log_path}")

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    # Ordered mapping: sample_id -> list of toolcall log lines
    sample_toolcalls: "OrderedDict[str, list[str]]" = OrderedDict()
    # Global per-"model" call counts (model here is the name before 'return:')
    model_call_counts: dict[str, int] = {}
    current_sample = None

    for line in lines:
        m = RUN_SAMPLE_RE.search(line)
        if m:
            current_sample = m.group(1)
            sample_toolcalls.setdefault(current_sample, [])
            continue

        if current_sample is None:
            continue

        if is_toolcall_line(line):
            sample_toolcalls[current_sample].append(line)
            m_tool = TOOLCALL_RE.search(line)
            if m_tool:
                model_name = m_tool.group(1).strip()
                model_call_counts[model_name] = model_call_counts.get(model_name, 0) + 1

    # Prepare metadata
    nonempty_samples = {
        sid: tool_lines for sid, tool_lines in sample_toolcalls.items() if tool_lines
    }
    total_samples_with_toolcalls = len(nonempty_samples)
    total_toolcalls = sum(len(v) for v in nonempty_samples.values())
    total_models = len(model_call_counts)

    header_lines = [
        "TOOLCALL ANALYSIS SUMMARY",
        f"total_samples_with_toolcalls: {total_samples_with_toolcalls}",
        f"total_toolcalls: {total_toolcalls}",
        f"total_distinct_models: {total_models}",
        "model_call_counts:",
    ]
    for model_name in sorted(model_call_counts.keys()):
        header_lines.append(f"  {model_name}: {model_call_counts[model_name]}")
    header_lines.append("")  # blank line after header

    # Write toolcall_logs.txt
    out_blocks = ["\n".join(header_lines)]
    for sample_id, tool_lines in nonempty_samples.items():
        header = f"{'=' * 20} Sample {sample_id} {'=' * 20}"
        count_line = f"toolcall_count: {len(tool_lines)}"
        out_blocks.append("\n".join([header, count_line, *tool_lines, ""]))
    with open(toolcall_logs_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_blocks))

    # Build toolcall_analysis.json
    metadata = {
        "total_samples_with_toolcalls": total_samples_with_toolcalls,
        "total_toolcalls": total_toolcalls,
        "total_distinct_models": total_models,
        "model_call_counts": model_call_counts,
    }
    toolcall_list = []
    for sample_id, tool_lines in nonempty_samples.items():
        # Count per-toolcall for this sample
        per_sample_counts: dict[str, int] = {}
        for line in tool_lines:
            m_tool = TOOLCALL_RE.search(line)
            if m_tool:
                name = m_tool.group(1).strip()
                per_sample_counts[name] = per_sample_counts.get(name, 0) + 1
        toolcalls = [
            {"name": name, "count": cnt}
            for name, cnt in sorted(per_sample_counts.items(), key=lambda x: (-x[1], x[0]))
        ]
        total_count = sum(cnt for _, cnt in per_sample_counts.items())
        toolcall_list.append({
            "id": sample_id,
            "total_count": total_count,
            "toolcalls": toolcalls,
        })
    data = {"metadata": metadata, "toolcall": toolcall_list}
    with open(toolcall_analysis_out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote toolcall logs to {toolcall_logs_out_path}, analysis to {toolcall_analysis_out_path} ({total_samples_with_toolcalls} samples)")


if __name__ == "__main__":
    main()

