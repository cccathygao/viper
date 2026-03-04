import argparse
import json
import os
import re


DEFAULT_LOG_PATH = "intermediate_results.txt"


ERROR_LINE_RE = re.compile(
    r"^Sample (\S+) failed (?:with error|at compilation time with error):\s*(.+)$"
)

def _normalize_error_type(msg: str) -> str:
    """
    Keep the original error sentence but replace only variable/identifier names
    so that e.g. "name 'table_left' is not defined" and "name 'door_left' is not defined"
    both become "name '<...>' is not defined", and
    "ImagePatch.overlaps_with() missing 3 required positional arguments: 'lower', 'right', and 'upper'"
    becomes "ImagePatch.overlaps_with() missing 3 required positional arguments".
    """
    m = msg.strip()
    lower = m.lower()

    # Drop the common suffix added by main_batch.py
    suffix = 'next you will see an "expected an indented block" error.'
    if suffix in lower:
        m = m[: lower.index(suffix)].strip()
        lower = m.lower()

    if "failed at compilation time with error: '[' was never closed" in lower:
        return "__SKIP__"

    # Replace variable/identifier names with placeholder; keep full sentence structure.

    # name '...' is not defined -> name '<...>' is not defined
    m = re.sub(r"name\s+'[^']+'\s+is\s+not\s+defined", "name '<...>' is not defined", m)

    # 'ImagePatch' object has no attribute '...' -> 'ImagePatch' object has no attribute '<...>'
    m = re.sub(r"(object has no attribute)\s+'[^']+'", r"\1 '<...>'", m, flags=re.IGNORECASE)

    # ImagePatch.overlaps_with() missing 3 required positional arguments: 'lower', 'right', and 'upper'
    # -> ImagePatch.overlaps_with() missing 3 required positional arguments
    m = re.sub(
        r"(missing\s+\d+\s+required\s+positional\s+arguments?)\s*:\s*[^.]+\.?",
        r"\1",
        m,
        flags=re.IGNORECASE,
    )

    # could not convert string to float: "..." -> could not convert string to float: '<...>'
    m = re.sub(r'(could not convert string to float)\s*:\s*"[^"]*"', r"\1: '<...>'", m, flags=re.IGNORECASE)
    m = re.sub(r"(could not convert string to float)\s*:\s*'[^']*'", r"\1: '<...>'", m, flags=re.IGNORECASE)

    # No model named maskrcnn. The available models are: ...
    m = re.sub(r"(No model named)\s+\S+(\s*\.\s*The available models)", r"\1 '<...>'\2", m, flags=re.IGNORECASE)

    # invalid syntax (Codex, line 6) -> invalid syntax (Codex, line <...>)
    m = re.sub(r"invalid syntax\s*\(Codex,\s*line\s*\d+\)", "invalid syntax (Codex, line <...>)", m, flags=re.IGNORECASE)

    # name 'execute_command' is not defined (compilation) - already covered by first rule

    # Fallback: replace remaining quoted identifiers (word chars/underscores only) to reduce duplication
    m = re.sub(r"'([a-zA-Z_][a-zA-Z0-9_]*)'", "'<...>'", m)
    m = re.sub(r'"([a-zA-Z_][a-zA-Z0-9_]*)"', '"<...>"', m)

    # Collapse repeated placeholders into one for readability
    m = re.sub(r"<\.\.\.>\s*(,\s*<\.\.\.>\s*)*", "<...>", m)
    # Normalize trailing period so "name 'x' is not defined" and "name 'x' is not defined." merge
    m = m.strip().rstrip(".")
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and de-duplicate error lines from intermediate_results.txt")
    parser.add_argument("--dir", type=str, default=None, help="Directory: output path becomes dir/error_summary.json")
    args = parser.parse_args()

    log_path = os.path.join(args.dir, DEFAULT_LOG_PATH) if args.dir else DEFAULT_LOG_PATH
    out_path = os.path.join(args.dir, "error_summary.json") if args.dir else "error_summary.json"

    if not os.path.isfile(log_path):
        raise SystemExit(f"Log file not found: {log_path}")

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    # error_type -> set(sample_id)
    error_type_to_samples: dict[str, set[str]] = {}
    total_error_lines = 0

    for line in lines:
        m = ERROR_LINE_RE.match(line)
        if not m:
            continue
        sample_id = m.group(1)
        msg = m.group(2)

        # Skip compilation-time '[' was never closed duplicates
        if "failed at compilation time with error: '[' was never closed" in line:
            continue

        total_error_lines += 1
        et = _normalize_error_type(msg)
        if et == "__SKIP__":
            continue
        error_type_to_samples.setdefault(et, set()).add(sample_id)

    # Write output as JSON
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    total_error_types = len(error_type_to_samples)
    total_samples_with_errors = len(set().union(*error_type_to_samples.values())) if error_type_to_samples else 0

    metadata = {
        "log_path": str(log_path),
        "total_error_lines_in_log_after_skipping_bracket_never_closed": total_error_lines,
        "total_distinct_error_types": total_error_types,
        "total_distinct_sample_ids_with_errors": total_samples_with_errors,
    }

    error_types = []
    for et in sorted(error_type_to_samples.keys(), key=lambda k: (-len(error_type_to_samples[k]), k)):
        samples_sorted = sorted(error_type_to_samples[et])
        error_types.append({
            "error": et,
            "sample id count": len(samples_sorted),
            "sample_ids": samples_sorted,
        })

    data = {"metadata": metadata, "error_types": error_types}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Wrote {total_error_types} error types to {out_path}")


if __name__ == "__main__":
    main()

