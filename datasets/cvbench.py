import os
import time
import json
import re
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
        self.input_type = 'image'

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
        
        # 1. Fix Image Path: Extract from list and handle absolute/relative paths
        img_path = sample['image']
        if isinstance(img_path, list):
            img_path = img_path[0]

        if not os.path.isabs(img_path):
            # Joins the directory of the jsonl with the image path
            # img_path = os.path.join(os.path.dirname(self.data_path), img_path)
            img_path = os.path.join('../dataset', img_path)
            
        img = Image.open(img_path).convert("RGB")

        if self.image_transforms:
            img_tensor = self.image_transforms(img)
        else:
            img_tensor = img

        # 2. Extract Question and Answer from 'conversations'
        # Human provides the question, GPT provides the answer
        human_msg = next(m['value'] for m in sample['conversations'] if m['from'] == 'human')
        gpt_msg = next(m['value'] for m in sample['conversations'] if m['from'] == 'gpt')

        # Extract Choice Letters (A, B, C, D) for possible_answers
        # This is critical to force the model to return a letter
        possible_answers = re.findall(r'\((\w)\)', human_msg)
        
        query = human_msg.strip()
        answer = gpt_msg.strip()

        if possible_answers:
            query += (
            "\n\nWrite a python program that first analyzes the image to find the answer, "
            "then ensures the final result returned is the single uppercase letter corresponding "
            f"to the correct choice: {', '.join(possible_answers)}."
        )

        out_dict = {
            "image": img_tensor,
            "query": query,
            "answer": answer,
            "sample_id": sample.get('id', index),
            "index": index,
            "possible_answers": possible_answers,
            "info_to_prompt": query,
            "extra_context": '',
            "query_type": 'MCQ'
        }

        return out_dict

    def get_sample_path(self, index):
        sample = self.samples[index]
        img_path = sample['image']
        if isinstance(img_path, list):
            img_path = img_path[0]
        sample_path = os.path.join('../dataset', img_path)
        return sample_path

    def post_process(self, prediction):
        """Clean the prediction to match ground truth format."""
        # prediction = general_postprocessing(prediction)
        # # If the ground truth is just 'a' and model says '(A) Dog', we strip to help matching
        # prediction = prediction.lower().strip()
        # return prediction
    
        if not prediction:
            return "none"
        
        prediction = str(prediction).strip()
        
        # 1. Check if prediction is exactly one of the letters (A, B, C, D)
        match = re.search(r'\b([A-G])\b', prediction.upper())
        if match:
            return match.group(1)
        
        # 2. Fallback to general cleaning
        return general_postprocessing(prediction)

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