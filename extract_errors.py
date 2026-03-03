import re
from pathlib import Path


LOG_PATH = Path("intermediate_results.txt")
OUT_PATH = Path("errors.txt")


RUN_SAMPLE_RE = re.compile(r"\[DEBUG\] Running sample (\S+)")
ERROR_RE = re.compile(r"Sample (\S+) failed (?:with error|at compilation time with error):")
ERROR_WITH_MSG_RE = re.compile(
    r"Sample (\S+) failed (?:with error|at compilation time with error): (.+)"
)


def _normalize_error(msg: str) -> str:
    """Group similar error messages into coarse error types."""
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

    # Fallback: use the full message, truncated for readability
    if len(m) > 200:
        m = m[:200] + "..."
    return m


def main() -> None:
    if not LOG_PATH.is_file():
        raise SystemExit(f"Log file not found: {LOG_PATH}")

    lines = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    n = len(lines)

    # Pre-index all "Running sample" lines: index -> sample_id
    run_indices = []
    for i, line in enumerate(lines):
        m = RUN_SAMPLE_RE.search(line)
        if m:
            run_indices.append((i, m.group(1)))

    # Find error lines per sample + collect error messages
    sample_first_error_idx: dict[str, int] = {}
    error_messages: list[str] = []
    for i, line in enumerate(lines):
        m = ERROR_RE.search(line)
        if m:
            sample_id = m.group(1)
            sample_first_error_idx.setdefault(sample_id, i)
        m2 = ERROR_WITH_MSG_RE.search(line)
        if m2:
            error_messages.append(m2.group(2))

    # Build high-level error stats
    error_type_counts: dict[str, int] = {}
    for msg in error_messages:
        et = _normalize_error(msg)
        error_type_counts[et] = error_type_counts.get(et, 0) + 1

    total_samples_with_errors = len(sample_first_error_idx)
    total_errors = len(error_messages)
    total_error_types = len(error_type_counts)

    output_blocks: list[str] = []

    # Header summary
    header_lines = [
        "ERROR ANALYSIS SUMMARY",
        f"total_samples_with_errors: {total_samples_with_errors}",
        f"total_errors: {total_errors}",
        f"total_distinct_error_types: {total_error_types}",
        "error_type_counts:",
    ]
    for et in sorted(error_type_counts.keys()):
        header_lines.append(f"  {et}: {error_type_counts[et]}")
    header_lines.append("")  # blank line after header
    output_blocks.append("\n".join(header_lines))

    for sample_id, err_idx in sorted(sample_first_error_idx.items(), key=lambda x: x[1]):
        # Find the last "Running sample <sample_id>" before this error
        start_idx = 0
        for idx, sid in run_indices:
            if idx <= err_idx and sid == sample_id:
                start_idx = idx
            if idx > err_idx:
                break

        # Find the next "Running sample" for a *different* sample after this error
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

    OUT_PATH.write_text("\n".join(output_blocks), encoding="utf-8")
    print(f"Wrote {len(output_blocks)} error blocks to {OUT_PATH}")


if __name__ == "__main__":
    main()

