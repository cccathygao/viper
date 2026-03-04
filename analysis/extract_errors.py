import argparse
import json
import os
import re


RUN_SAMPLE_RE = re.compile(r"\[DEBUG\] Running sample (\S+)")
ERROR_RE = re.compile(r"Sample (\S+) failed (?:with error|at compilation time with error):")
ERROR_WITH_MSG_RE = re.compile(
    r"Sample (\S+) failed (?:with error|at compilation time with error): (.+)"
)
ERROR_LINE_RE = re.compile(
    r"^Sample (\S+) failed (?:with error|at compilation time with error):\s*(.+)$"
)


def _normalize_error(msg: str) -> str:
    """Group similar error messages into coarse error types (for errors.txt)."""
    m = msg.strip()
    lower = m.lower()

    if "not defined" in lower or "not found" in lower:
        return "NameError: variable not found"
    if "list index out of range" in lower:
        return "IndexError: list index out of range"
    if "was never closed" in lower:
        return "SyntaxError: bracket was never closed"
    if "expected an indented block" in lower:
        return "IndentationError: expected an indented block"
    if "isinstance() arg 2 must be a type" in lower:
        return "TypeError: invalid isinstance second argument"
    if "has no attribute" in lower:
        return "AttributeError: attribute missing on object"
    if "missing" in lower and "required positional argument" in lower:
        return "TypeError: missing required positional argument(s)"
    if "division by zero" in lower or "float division by zero" in lower:
        return "ZeroDivisionError: division by zero"
    if "invalid syntax" in lower:
        return "SyntaxError: invalid syntax"

    if len(m) > 200:
        m = m[:200] + "..."
    return m


def _normalize_error_type(msg: str) -> str:
    """
    Keep the original error sentence but replace only variable/identifier names
    (for error_summary.json deduplication).
    """
    m = msg.strip()
    lower = m.lower()

    suffix = 'next you will see an "expected an indented block" error.'
    if suffix in lower:
        m = m[: lower.index(suffix)].strip()
        lower = m.lower()

    if "failed at compilation time with error: '[' was never closed" in lower:
        return "__SKIP__"

    m = re.sub(r"name\s+'[^']+'\s+is\s+not\s+defined", "name '<...>' is not defined", m)
    m = re.sub(r"(object has no attribute)\s+'[^']+'", r"\1 '<...>'", m, flags=re.IGNORECASE)
    m = re.sub(
        r"(missing\s+\d+\s+required\s+positional\s+arguments?)\s*:\s*[^.]+\.?",
        r"\1",
        m,
        flags=re.IGNORECASE,
    )
    m = re.sub(r'(could not convert string to float)\s*:\s*"[^"]*"', r"\1: '<...>'", m, flags=re.IGNORECASE)
    m = re.sub(r"(could not convert string to float)\s*:\s*'[^']*'", r"\1: '<...>'", m, flags=re.IGNORECASE)
    m = re.sub(r"(No model named)\s+\S+(\s*\.\s*The available models)", r"\1 '<...>'\2", m, flags=re.IGNORECASE)
    m = re.sub(r"invalid syntax\s*\(Codex,\s*line\s*\d+\)", "invalid syntax (Codex, line <...>)", m, flags=re.IGNORECASE)
    m = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'", "'<...>'", m)
    m = re.sub(r'"([a-zA-Z_][a-zA-Z0-9_]*)"', '"<...>"', m)
    m = re.sub(r"<\.\.\.>\s*(,\s*<\.\.\.>\s*)*", "<...>", m)
    m = m.strip().rstrip(".")
    return m


def main():
    parser = argparse.ArgumentParser(
        description="Extract errors from intermediate_results.txt: writes errors.txt and error_summary.json"
    )
    parser.add_argument("--dir", type=str, required=True, help="Directory: reads dir/intermediate_results.txt, writes dir/errors.txt and dir/error_summary.json")
    args = parser.parse_args()

    log_path = os.path.join(args.dir, "intermediate_results.txt")
    errors_txt_path = os.path.join(args.dir, "raw_error_logs.txt")
    summary_json_path = os.path.join(args.dir, "error_summary.json")

    if not os.path.isfile(log_path):
        raise SystemExit(f"Log file not found: {log_path}")

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    n = len(lines)

    # Pre-index "Running sample" lines
    run_indices = []
    for i, line in enumerate(lines):
        m = RUN_SAMPLE_RE.search(line)
        if m:
            run_indices.append((i, m.group(1)))

    # Collect data for both outputs
    sample_first_error_idx = {}
    error_messages = []
    error_type_to_samples = {}
    total_error_lines = 0

    for i, line in enumerate(lines):
        m = ERROR_RE.search(line)
        if m:
            sample_first_error_idx.setdefault(m.group(1), i)

        m2 = ERROR_WITH_MSG_RE.search(line)
        if m2:
            error_messages.append(m2.group(2))

        m3 = ERROR_LINE_RE.match(line)
        if m3:
            if "failed at compilation time with error: '[' was never closed" in line:
                continue
            sample_id, msg = m3.group(1), m3.group(2)
            total_error_lines += 1
            et = _normalize_error_type(msg)
            if et != "__SKIP__":
                error_type_to_samples.setdefault(et, set()).add(sample_id)

    # Build error_type_counts for errors.txt header
    error_type_counts = {}
    for msg in error_messages:
        et = _normalize_error(msg)
        error_type_counts[et] = error_type_counts.get(et, 0) + 1

    total_samples_with_errors = len(sample_first_error_idx)
    total_errors = len(error_messages)
    total_error_types = len(error_type_counts)

    # ---- Write errors.txt ----
    output_blocks = []
    header_lines = [
        "ERROR ANALYSIS SUMMARY",
        f"total_samples_with_errors: {total_samples_with_errors}",
        f"total_errors: {total_errors}",
        f"total_distinct_error_types: {total_error_types}",
        "error_type_counts:",
    ]
    for et in sorted(error_type_counts.keys()):
        header_lines.append(f"  {et}: {error_type_counts[et]}")
    header_lines.append("")
    output_blocks.append("\n".join(header_lines))

    for sample_id, err_idx in sorted(sample_first_error_idx.items(), key=lambda x: x[1]):
        start_idx = 0
        for idx, sid in run_indices:
            if idx <= err_idx and sid == sample_id:
                start_idx = idx
            if idx > err_idx:
                break

        end_idx = n - 1
        for idx, sid in run_indices:
            if idx <= err_idx:
                continue
            if sid != sample_id:
                end_idx = idx - 1
                break

        block = lines[start_idx : end_idx + 1]
        header = f"{'=' * 20} Sample {sample_id} {'=' * 20}"
        output_blocks.append("\n".join([header, *block, ""]))

    with open(errors_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_blocks))
    print(f"Wrote {len(output_blocks)} error blocks to {errors_txt_path}")

    # ---- Write error_summary.json ----
    summary_error_types = len(error_type_to_samples)
    summary_sample_count = len(set().union(*error_type_to_samples.values())) if error_type_to_samples else 0

    metadata = {
        "log_path": log_path,
        "total_error_lines_in_log_after_skipping_bracket_never_closed": total_error_lines,
        "total_distinct_error_types": summary_error_types,
        "total_distinct_sample_ids_with_errors": summary_sample_count,
    }

    error_types_list = []
    for et in sorted(error_type_to_samples.keys(), key=lambda k: (-len(error_type_to_samples[k]), k)):
        samples_sorted = sorted(error_type_to_samples[et])
        error_types_list.append({
            "error": et,
            "sample id count": len(samples_sorted),
            "sample_ids": samples_sorted,
        })

    data = {"metadata": metadata, "error_types": error_types_list}
    os.makedirs(args.dir, exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Wrote {summary_error_types} error types to {summary_json_path}")


if __name__ == "__main__":
    main()
