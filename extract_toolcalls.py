import re
from pathlib import Path
from collections import OrderedDict


LOG_PATH = Path("intermediate_results.txt")
OUT_PATH = Path("toolcall_analysis.txt")


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
    if not LOG_PATH.is_file():
        raise SystemExit(f"Log file not found: {LOG_PATH}")

    lines = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()

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

    # Write per-sample blocks
    out_blocks = ["\n".join(header_lines)]
    for sample_id, tool_lines in nonempty_samples.items():
        header = f"{'=' * 20} Sample {sample_id} {'=' * 20}"
        count_line = f"toolcall_count: {len(tool_lines)}"
        out_blocks.append("\n".join([header, count_line, *tool_lines, ""]))

    OUT_PATH.write_text("\n".join(out_blocks), encoding="utf-8")
    print(f"Wrote toolcall analysis for {total_samples_with_toolcalls} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()

