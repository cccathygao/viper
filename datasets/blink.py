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
        
        # Load all images in the list
        for p in img_paths:
            if not os.path.isabs(p):
                # Search in ../dataset or current dir
                full_path = os.path.join('../dataset', p)
                if not os.path.exists(full_path):
                    full_path = p
            else:
                full_path = p
                
            img = Image.open(full_path).convert("RGB")
            images.append(img)

        # Concatenate images horizontally so ImagePatch receives one object
        if len(images) > 1:
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            
            # Create a black canvas and paste images side-by-side
            new_im = Image.new('RGB', (total_width, max_height), (0, 0, 0))
            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]
            image = new_im
        elif len(images) == 1:
            image = images[0]
            widths = [image.size[0]]
        else:
            raise ValueError(f"No images found for sample {sample.get('id', index)}")

        if self.image_transforms:
            image = self.image_transforms(image)

        # Extract Question and Answer
        human_msg = next(m['value'] for m in sample['conversations'] if m['from'] == 'human')
        gpt_msg = next(m['value'] for m in sample['conversations'] if m['from'] == 'gpt')
        possible_answers = re.findall(r'\((\w)\)', human_msg)
        
        query = human_msg.strip()
        answer = gpt_msg.strip()

        # Add layout hint to the query for the LLM
        if len(images) > 1:
            layout_hint = (
                f"Note: The input is {len(images)} images concatenated horizontally. "
                f"The first image (reference) ends at x={widths[0]}. "
                f"The second image is between x={widths[0]} and x={widths[0]+widths[1]}. "
                "Use these horizontal boundaries to distinguish images.\n\n"
            )
            query = layout_hint + query

        return {
            "image": image, # Now a single Image/Tensor
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