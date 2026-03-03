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
        img_paths = sample['image'] 
        images = []
        
        for p in img_paths:
            full_path = os.path.join('../dataset', p) if not os.path.isabs(p) else p
            img = Image.open(full_path).convert("RGB")
            if self.image_transforms:
                img = self.image_transforms(img)
            # Ensure the image is a torch tensor
            if not isinstance(img, torch.Tensor):
                from torchvision.transforms import ToTensor
                img = ToTensor()(img)
            images.append(img)

        # Concatenate images horizontally
        # Ensure they all have the same height (resize to the height of the first image)
        ref_h, ref_w = images[0].shape[1], images[0].shape[2]
        processed_images = []
        for img in images:
            if img.shape[1] != ref_h:
                import torch.nn.functional as F
                img = F.interpolate(img.unsqueeze(0), size=(ref_h, img.shape[2])).squeeze(0)
            processed_images.append(img)
        
        combined_image = torch.cat(processed_images, dim=-1) # Concatenate width-wise

        human_msg = next(m['value'] for m in sample['conversations'] if m['from'] == 'human')
        gpt_msg = next(m['value'] for m in sample['conversations'] if m['from'] == 'gpt')
        possible_answers = re.findall(r'\((\w)\)', human_msg)
        
        # Add spatial context to help the LLM generate correct code
        query = human_msg.strip()
        extra_context = (
            "The input image contains three paintings concatenated horizontally. "
            "From left to right: the first painting is the reference, the second is Choice (A), "
            "and the third is Choice (B)."
        )

        return {
            "image": combined_image,
            "query": query,
            "answer": gpt_msg.strip(),
            "sample_id": sample.get('id', index),
            "index": index,
            "possible_answers": possible_answers,
            "info_to_prompt": query,
            "extra_context": extra_context,
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