import os
import time
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import general_postprocessing

class CVBenchDataset(Dataset):
    """
    Dataset loader for CV-Bench. Supports .jsonl and .csv formats.
    """

    def __init__(self, split="", data_path="", 
                 image_transforms=None, max_samples=None, **kwargs):
        start_time = time.time()
        self.split = split
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.samples = []

        # 1. Handle JSONL loading
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))
        # 2. Handle CSV loading (fallback)
        elif data_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(data_path)
            self.samples = df.to_dict('records')
        else:
            raise ValueError(f"Unsupported file format for {data_path}. Use .jsonl or .csv")

        # Optional: Filter by split if the key exists in your jsonl
        if split:
            self.samples = [s for s in self.samples if s.get('split') == split]

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.n_samples = len(self.samples)
        print(f"Loaded CV-Bench with {self.n_samples} samples ({time.time() - start_time:.2f}s)")

    def __getitem__(self, index):
        sample = self.samples[index]
        
        # Load Image: Assumes 'image' field is a path relative to the data_path directory
        # or an absolute path. Adjust if images are in a specific subfolder.
        img_path = sample['image']
        if not os.path.isabs(img_path):
            # If the jsonl only has filenames, join with the directory of the data_path
            img_path = os.path.join(os.path.dirname(self.data_path), img_path)
            
        img = Image.open(img_path).convert("RGB")

        if self.image_transforms:
            img_tensor = self.image_transforms(img)
        else:
            img_tensor = img

        # Format the query with Multiple Choice options if they exist
        question = sample['question']
        options_text = ""
        # Common CV-Bench keys: 'option_a', 'option_b', etc.
        for letter in ['a', 'b', 'c', 'd']:
            key = f'option_{letter}'
            if key in sample and sample[key]:
                options_text += f" ({letter.upper()}) {sample[key]}"
        
        query = f"{question}{options_text}"

        out_dict = {
            "image": img_tensor,
            "query": query,
            "answer": str(sample.get('answer', '')),
            "id": sample.get('id', index),
            "index": index,
            "possible_answers": [sample.get(f'option_{l}') for l in ['a','b','c','d'] if f'option_{l}' in sample],
            "info_to_prompt": query
        }

        return out_dict

    def post_process(self, prediction):
        """Clean the prediction to match ground truth format."""
        prediction = general_postprocessing(prediction)
        # If the ground truth is just 'a' and model says '(A) Dog', we strip to help matching
        prediction = prediction.lower().strip()
        return prediction

    def accuracy(self, prediction, ground_truth, *args):
        if len(prediction) == 0:
            return 0
        score = 0
        for p, g in zip(prediction, ground_truth):
            # Check for direct match or matching the choice letter
            if self.post_process(p) == self.post_process(str(g)):
                score += 1
        return score / len(prediction)

    def __len__(self):
        return self.n_samples