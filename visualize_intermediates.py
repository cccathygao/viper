import argparse
import os
import re
from typing import Dict, List, Tuple

from PIL import Image
import torch

from utils import draw_bounding_boxes


def load_cvbench_image_map(data_path: str) -> Dict[str, str]:
    """
    Build a mapping from sample_id (e.g. 'Count_0') to absolute image path
    for a CVBench jsonl file like cvbench_data_small_test.jsonl.
    """
    import json

    id_to_path: Dict[str, str] = {}
    base_dir = os.path.dirname(os.path.abspath(data_path))

    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            sample_id = sample.get("id")
            img = sample.get("image")
            if isinstance(img, list):
                img = img[0]
            # Mirror the logic in CVBenchDataset: resolve via ../dataset
            if not os.path.isabs(img):
                img_path = os.path.join(base_dir, "..", "dataset", img)
            else:
                img_path = img
            id_to_path[sample_id] = os.path.normpath(img_path)

    return id_to_path


def parse_glip_bboxes_from_log(
    log_path: str,
) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """
    Parse GLIPModel bbox outputs from intermediate_results.txt.

    We track the current sample via lines like:
        [DEBUG] Running sample Count_0
    and then collect the first bbox tensor we see for GLIPModel under that sample:
        [DEBUG] vision_models.py, GLIPModel return: ...

    Returns:
        mapping: sample_id -> list of (left, lower, right, upper) boxes
    """
    sample_to_bboxes: Dict[str, List[Tuple[int, int, int, int]]] = {}

    current_sample = None
    collecting_block = False
    block_lines: List[str] = []

    running_re = re.compile(r"\[DEBUG\] Running sample ([^\s]+)")
    glip_return_re = re.compile(r"\[DEBUG\] vision_models\.py, GLIPModel return:")

    def flush_block_for_sample(sample_id: str, block: List[str]) -> None:
        if sample_id is None or not block:
            return
        text = "\n".join(block)
        # Find all [x1, y1, x2, y2] occurrences, robust to spaces/newlines
        matches = re.findall(
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", text
        )
        if not matches:
            return
        boxes = [(int(a), int(b), int(c), int(d)) for a, b, c, d in matches]
        if boxes:
            sample_to_bboxes.setdefault(sample_id, []).extend(boxes)

    with open(log_path, "r") as f:
        for line in f:
            # Detect new running sample
            m = running_re.search(line)
            if m:
                # If we were in the middle of a GLIP block, flush it for the previous sample
                if collecting_block:
                    flush_block_for_sample(current_sample, block_lines)
                    collecting_block = False
                    block_lines = []

                current_sample = m.group(1)
                continue

            # Start of a GLIPModel return block
            if glip_return_re.search(line):
                # If we were already collecting (should be rare), flush first
                if collecting_block:
                    flush_block_for_sample(current_sample, block_lines)
                    block_lines = []
                collecting_block = True
                block_lines.append(line.rstrip("\n"))
                continue

            # While collecting, keep appending until we hit a separator or another [DEBUG]
            if collecting_block:
                if line.strip().startswith("[DEBUG]") or not line.strip():
                    flush_block_for_sample(current_sample, block_lines)
                    collecting_block = False
                    block_lines = []
                else:
                    block_lines.append(line.rstrip("\n"))

    # Flush any trailing block
    if collecting_block:
        flush_block_for_sample(current_sample, block_lines)

    return sample_to_bboxes


def draw_and_save_boxes(
    sample_to_image: Dict[str, str],
    sample_to_bboxes: Dict[str, List[Tuple[int, int, int, int]]],
    output_dir: str,
) -> None:
    """
    For each sample that has GLIP bboxes, load the corresponding image,
    draw all boxes using utils.draw_bounding_boxes, and save to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    from torchvision import transforms

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    for sample_id, boxes in sample_to_bboxes.items():
        img_path = sample_to_image.get(sample_id)
        if not img_path or not os.path.isfile(img_path):
            print(f"[WARN] No image found for sample {sample_id} ({img_path})")
            continue

        print(f"[INFO] Drawing {len(boxes)} GLIP boxes for {sample_id} on {img_path}")
        img = Image.open(img_path).convert("RGB")
        img_t = to_tensor(img)

        bboxes_tensor = torch.tensor(boxes, dtype=torch.int64)

        # Draw in red, leave space for labels if needed later
        drawn = draw_bounding_boxes(
            img_t,
            bboxes_tensor,
            colors=["red"] * len(boxes),
            width=4,
        )

        out_img = to_pil(drawn)
        out_name = f"{sample_id}_glip.png"
        out_path = os.path.join(output_dir, out_name)
        out_img.save(out_path)
        print(f"[INFO] Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize GLIPModel intermediate bounding boxes from ViperGPT debug logs."
    )
    parser.add_argument(
        "--log",
        type=str,
        default="intermediate_results.txt",
        help="Path to intermediate_results.txt containing [DEBUG] logs.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="cvbench_data_small_test.jsonl",
        help="Path to CVBench jsonl used as dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="debug_intermediate_visualizations",
        help="Directory to save visualized images.",
    )

    args = parser.parse_args()

    sample_to_image = load_cvbench_image_map(args.dataset_path)
    sample_to_bboxes = parse_glip_bboxes_from_log(args.log)

    if not sample_to_bboxes:
        print("[WARN] No GLIPModel return blocks found in log.")
        return

    draw_and_save_boxes(sample_to_image, sample_to_bboxes, args.output_dir)


if __name__ == "__main__":
    main()

