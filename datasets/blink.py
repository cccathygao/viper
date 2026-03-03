import os
import time
import json
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import general_postprocessing 

class BLINKDataset(Dataset):
    """
    Dataset loader for BLINK. Adapted from CVBench to support multiple images per sample.
    """

    def __init__(self, split="", data_path="", 
                 image_transforms=None, max_samples=None, **kwargs):
        start_time = time.time()
        self.split = split
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.samples = []
        self.input_type = 'image'

        if data_path.endswith('.jsonl'):
            with open(data_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format for {data_path}. Use .jsonl")

        if max_samples:
            self.samples = self.samples[:max_samples]

        self.n_samples = len(self.samples)
        print(f"Loaded BLINK with {self.n_samples} samples ({time.time() - start_time:.2f}s)")

    def __getitem__(self, index):
        sample = self.samples[index]
        
        # BLINK usually contains 3 images for comparison tasks
        img_paths = sample['image'] 
        images = []
        
        for p in img_paths:
            # Handle pathing consistent with your cvbench.py setup
            if not os.path.isabs(p):
                full_path = os.path.join('../dataset', p)
            else:
                full_path = p
                
            img = Image.open(full_path).convert("RGB")
            if self.image_transforms:
                img = self.image_transforms(img)
            images.append(img)

        # Extract Question and Answer from 'conversations'
        human_msg = next(m['value'] for m in sample['conversations'] if m['from'] == 'human')
        gpt_msg = next(m['value'] for m in sample['conversations'] if m['from'] == 'gpt')

        # Extract MCQ options (A, B, C...)
        possible_answers = re.findall(r'\((\w)\)', human_msg)
        
        query = human_msg.strip()
        answer = gpt_msg.strip()

        # Tailor the prompt for ViperGPT/Code-generation logic
        # if possible_answers:
        #     query += (
        #         "\n\nWrite a python program that uses the provided images to find the answer. "
        #         "The first image is the reference. Ensure the final result returned is the "
        #         f"single uppercase letter corresponding to the correct choice: {', '.join(possible_answers)}."
        #     )

        return {
            "image": images, # Returning the list of processed images
            "query": query,
            "answer": answer,
            "sample_id": sample.get('id', index),
            "index": index,
            "possible_answers": possible_answers,
            "info_to_prompt": query,
            "extra_context": '',
            "query_type": 'MCQ'
        }

    def get_sample_path(self, index):
        sample = self.samples[index]
        img_path = sample['image']
        if isinstance(img_path, list):
            img_path = img_path[0]
        sample_path = os.path.join('../dataset', img_path)
        return sample_path
    
    def post_process(self, prediction):
        if not prediction:
            return "none"
        
        prediction = str(prediction).strip()
        # Look for the MCQ letter in the output
        match = re.search(r'\b([A-G])\b', prediction.upper())
        if match:
            return match.group(1)
        
        return general_postprocessing(prediction)

    def accuracy(self, prediction, ground_truth, *args):
        if not prediction:
            return 0
        score = 0
        for p, g in zip(prediction, ground_truth):
            if self.post_process(p) == self.post_process(str(g)):
                score += 1
        return score / len(prediction)

    def __len__(self):
        return self.n_samples