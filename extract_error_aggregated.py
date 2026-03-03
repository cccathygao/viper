import re
from pathlib import Path


LOG_PATH = Path("intermediate_results.txt")
OUT_PATH = Path("error_aggregated.txt")


ERROR_LINE_RE = re.compile(
    r"^Sample (\S+) failed (?:with error|at compilation time with error):"
)


def main() -> None:
    if not LOG_PATH.is_file():
        raise SystemExit(f"Log file not found: {LOG_PATH}")

    lines = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()

    aggregated: list[str] = []
    for line in lines:
        if not ERROR_LINE_RE.search(line):
            continue
        # Skip compilation-time '[' was never closed duplicates
        if "failed at compilation time with error: '[' was never closed" in line:
            continue
        aggregated.append(line)

    OUT_PATH.write_text("\n".join(aggregated) + ("\n" if aggregated else ""), encoding="utf-8")
    print(f"Wrote {len(aggregated)} error summary lines to {OUT_PATH}")


if __name__ == "__main__":
    main()

