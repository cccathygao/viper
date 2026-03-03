from __future__ import annotations

import numpy as np
import re
import torch
from dateutil import parser as dateparser
from PIL import Image
from rich.console import Console
from torchvision import transforms
from torchvision.ops import box_iou
from typing import Union, List
from word2number import w2n

from utils import show_single_image, load_json
from vision_processes import forward, config

console = Console(highlight=False)

def _debug_summarize(value, *, max_str: int = 800, max_items: int = 10) -> str:
    if isinstance(value, torch.Tensor):
        t = value
        summary = f"torch.Tensor(shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device})"
        try:
            if t.numel() <= 20:
                summary += f", value={t.detach().cpu()!r}"
            else:
                flat = t.detach().flatten()
                n = min(max_items, flat.numel())
                summary += f", sample={flat[:n].cpu().tolist()!r}"
        except Exception:
            pass
        return summary

    if isinstance(value, np.ndarray):
        return f"np.ndarray(shape={value.shape}, dtype={value.dtype})"

    if isinstance(value, Image.Image):
        return f"PIL.Image(size={value.size}, mode={value.mode})"

    if isinstance(value, (list, tuple)):
        try:
            head = list(value)[:max_items]
            return f"{type(value).__name__}(len={len(value)}): {repr(head)[:max_str]}"
        except Exception:
            return f"{type(value).__name__}"

    if isinstance(value, dict):
        try:
            keys = list(value.keys())[:max_items]
            return f"dict(keys={keys!r})"
        except Exception:
            return "dict(?)"

    if isinstance(value, str):
        s = value.replace("\n", "\\n")
        if len(s) > max_str:
            s = s[:max_str] + "..."
        return f"str(len={len(value)}): {s}"

    r = repr(value)
    if len(r) > max_str:
        r = r[:max_str] + "..."
    return r


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    def __init__(self, image: Union[Image.Image, torch.Tensor, np.ndarray], left: int = None, lower: int = None,
                 right: int = None, upper: int = None, parent_left=0, parent_lower=0, queues=None,
                 parent_img_patch=None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, image.shape[1]-upper:image.shape[1]-lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        self.possible_options = load_json('./useful_lists/possible_options.json')

    def forward(self, model_name, *args, **kwargs):
        return forward(model_name, *args, queues=self.queues, **kwargs)

    @property
    def original_image(self):
        if self.parent_img_patch is None:
            return self.cropped_image
        else:
            return self.parent_img_patch.original_image

    def find(self, object_name: str) -> list[ImagePatch]:
        print(f'[DEBUG] image_patch.py, find({object_name})\n', flush=True)

        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop
        """
        if object_name in ["object", "objects"]:
            all_object_coordinates = self.forward('maskrcnn', self.cropped_image)[0]
        else:

            if object_name == 'person':
                object_name = 'people'  # GLIP does better at people than person

            all_object_coordinates = self.forward('glip', self.cropped_image, object_name)
        if len(all_object_coordinates) == 0:
            result = []
            print(f'[DEBUG] image_patch.py, find return: {_debug_summarize(result)}\n', flush=True)
            return result

        threshold = config.ratio_box_area_to_image_area
        if threshold > 0:
            area_im = self.width * self.height
            all_areas = torch.tensor([(coord[2]-coord[0]) * (coord[3]-coord[1]) / area_im
                                      for coord in all_object_coordinates])
            mask = all_areas > threshold
            # if not mask.any():
            #     mask = all_areas == all_areas.max()  # At least return one element
            all_object_coordinates = all_object_coordinates[mask]

        
        result = [self.crop(*coordinates) for coordinates in all_object_coordinates]
        print(f'[DEBUG] image_patch.py, find return: {_debug_summarize(result)}\n', flush=True)
        return result

    def exists(self, object_name) -> bool:
        print(f'[DEBUG] image_patch.py, exist{object_name})\n', flush=True)

        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        if object_name.isdigit() or object_name.lower().startswith("number"):
            object_name = object_name.lower().replace("number", "").strip()

            object_name = w2n.word_to_num(object_name)
            answer = self.simple_query("What number is written in the image (in digits)?")
            result = w2n.word_to_num(answer) == object_name
            print(f'[DEBUG] image_patch.py, exists return: {_debug_summarize(result)}\n', flush=True)
            return result

        patches = self.find(object_name)

        filtered_patches = []
        for patch in patches:
            if "yes" in patch.simple_query(f"Is this a {object_name}?"):
                filtered_patches.append(patch)
        result = len(filtered_patches) > 0
        print(f'[DEBUG] image_patch.py, exists return: {_debug_summarize(result)}\n', flush=True)
        return result

    def _score(self, category: str, negative_categories=None, model='clip') -> float:
        print(f'[DEBUG] image_patch.py, _score({category},{model})\n', flush=True)

        """
        Returns a binary score for the similarity between the image and the category.
        The negative categories are used to compare to (score is relative to the scores of the negative categories).
        """
        if model == 'clip':
            res = self.forward('clip', self.cropped_image, category, task='score',
                               negative_categories=negative_categories)
        elif model == 'tcl':
            res = self.forward('tcl', self.cropped_image, category, task='score')
        else:  # xvlm
            task = 'binary_score' if negative_categories is not None else 'score'
            res = self.forward('xvlm', self.cropped_image, category, task=task, negative_categories=negative_categories)
            res = res.item()

        result = res
        print(f'[DEBUG] image_patch.py, _score return: {_debug_summarize(result)}\n', flush=True)
        return result

    def _detect(self, category: str, thresh, negative_categories=None, model='clip') -> bool:
        print(f'[DEBUG] image_patch.py, _detect({category},{model})\n', flush=True)

        result = self._score(category, negative_categories, model) > thresh
        print(f'[DEBUG] image_patch.py, _detect return: {_debug_summarize(result)}\n', flush=True)
        return result

    def verify_property(self, object_name: str, attribute: str) -> bool:
        print(f'[DEBUG] image_patch.py, verify_property({object_name},{attribute})\n', flush=True)

        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        name = f"{attribute} {object_name}"
        model = config.verify_property.model
        negative_categories = [f"{att} {object_name}" for att in self.possible_options['attributes']]
        if model == 'clip':
            result = self._detect(name, negative_categories=negative_categories,
                                  thresh=config.verify_property.thresh_clip, model='clip')
        elif model == 'tcl':
            result = self._detect(name, thresh=config.verify_property.thresh_tcl, model='tcl')
        else:  # 'xvlm'
            result = self._detect(name, negative_categories=negative_categories,
                                  thresh=config.verify_property.thresh_xvlm, model='xvlm')
        print(f'[DEBUG] image_patch.py, verify_property return: {_debug_summarize(result)}\n', flush=True)
        return result

    def best_text_match(self, option_list: list[str] = None, prefix: str = None) -> str:
        print(f'[DEBUG] image_patch.py, best_text_match({option_list})\n', flush=True)

        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options
        """
        option_list_to_use = option_list
        if prefix is not None:
            option_list_to_use = [prefix + " " + option for option in option_list]

        model_name = config.best_match_model
        image = self.cropped_image
        text = option_list_to_use
        if model_name in ('clip', 'tcl'):
            selected = self.forward(model_name, image, text, task='classify')
        elif model_name == 'xvlm':
            res = self.forward(model_name, image, text, task='score')
            res = res.argmax().item()
            selected = res
        else:
            raise NotImplementedError

        result = option_list[selected]
        print(f'[DEBUG] image_patch.py, best_text_match return: {_debug_summarize(result)}\n', flush=True)
        return result

    def simple_query(self, question: str):
        print(f'[DEBUG] image_patch.py, simple_query({question})\n', flush=True)

        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        result = self.forward('blip', self.cropped_image, question, task='qa')
        print(f'[DEBUG] image_patch.py, simple_query return: {_debug_summarize(result)}\n', flush=True)
        return result

    def compute_depth(self):
        print(f'[DEBUG] image_patch.py, compute_depth()\n', flush=True)

        """Returns the median depth of the image crop
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop
        """
        original_image = self.original_image
        depth_map = self.forward('depth', original_image)
        depth_map = depth_map[original_image.shape[1]-self.upper:original_image.shape[1]-self.lower,
                              self.left:self.right]
        result = depth_map.median()  # Ideally some kind of mode, but median is good enough for now
        print(f'[DEBUG] image_patch.py, compute_depth return: {_debug_summarize(result)}\n', flush=True)
        return result

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
        print(f'[DEBUG] image_patch.py, crop(left:{left}, lower:{lower}, right:{right}, upper:{upper})\n', flush=True)

        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)

        if config.crop_larger_margin:
            left = max(0, left - 10)
            lower = max(0, lower - 10)
            right = min(self.width, right + 10)
            upper = min(self.height, upper + 10)

        result = ImagePatch(self.cropped_image, left, lower, right, upper, self.left, self.lower, queues=self.queues,
                            parent_img_patch=self)
        print(f'[DEBUG] image_patch.py, crop return: {_debug_summarize(result)}\n', flush=True)
        return result

    def overlaps_with(self, left, lower, right, upper):
        print(f'[DEBUG] image_patch.py, overlaps_with()\n', flush=True)

        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        result = self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower
        print(f'[DEBUG] image_patch.py, overlaps_with return: {_debug_summarize(result)}\n', flush=True)
        return result

    def llm_query(self, question: str, long_answer: bool = True) -> str:
        print(f'[DEBUG] image_patch.py, llm_query({question})\n', flush=True)

        result = llm_query(question, None, long_answer)
        print(f'[DEBUG] image_patch.py, llm_query(method) return: {_debug_summarize(result)}\n', flush=True)
        return result

    def print_image(self, size: tuple[int, int] = None):
        show_single_image(self.cropped_image, size)

    def __repr__(self):
        return "ImagePatch({}, {}, {}, {})".format(self.left, self.lower, self.right, self.upper)


def best_image_match(list_patches: list[ImagePatch], content: List[str], return_index: bool = False) -> \
        Union[ImagePatch, None]:
    
    print(f'[DEBUG] image_patch.py, best_image_match()\n', flush=True)

    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    if len(list_patches) == 0:
        result = None
        print(f'[DEBUG] image_patch.py, best_image_match return: {_debug_summarize(result)}\n', flush=True)
        return result

    model = config.best_match_model

    scores = []
    for cont in content:
        if model == 'clip':
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='compare',
                                          return_scores=True)
        else:
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='score')
        scores.append(res)
    scores = torch.stack(scores).mean(dim=0)
    scores = scores.argmax().item()  # Argmax over all image patches

    if return_index:
        result = scores
        print(f'[DEBUG] image_patch.py, best_image_match return: {_debug_summarize(result)}\n', flush=True)
        return result
    result = list_patches[scores]
    print(f'[DEBUG] image_patch.py, best_image_match return: {_debug_summarize(result)}\n', flush=True)
    return result


def distance(patch_a: Union[ImagePatch, float], patch_b: Union[ImagePatch, float]) -> float:
    print(f'[DEBUG] image_patch.py, distance()\n', flush=True)

    """
    Returns the distance between the edges of two ImagePatches, or between two floats.
    If the patches overlap, it returns a negative distance corresponding to the negative intersection over union.
    """

    if isinstance(patch_a, ImagePatch) and isinstance(patch_b, ImagePatch):
        a_min = np.array([patch_a.left, patch_a.lower])
        a_max = np.array([patch_a.right, patch_a.upper])
        b_min = np.array([patch_b.left, patch_b.lower])
        b_max = np.array([patch_b.right, patch_b.upper])

        u = np.maximum(0, a_min - b_max)
        v = np.maximum(0, b_min - a_max)

        dist = np.sqrt((u ** 2).sum() + (v ** 2).sum())

        if dist == 0:
            box_a = torch.tensor([patch_a.left, patch_a.lower, patch_a.right, patch_a.upper])[None]
            box_b = torch.tensor([patch_b.left, patch_b.lower, patch_b.right, patch_b.upper])[None]
            dist = - box_iou(box_a, box_b).item()

    else:
        dist = abs(patch_a - patch_b)

    result = dist
    print(f'[DEBUG] image_patch.py, distance return: {_debug_summarize(result)}\n', flush=True)
    return result


def bool_to_yesno(bool_answer: bool) -> str:
    print(f'[DEBUG] image_patch.py, bool_to_yesno()\n', flush=True)
    """Returns a yes/no answer to a question based on the boolean value of bool_answer.
    Parameters
    ----------
    bool_answer : bool
        a boolean value

    Returns
    -------
    str
        a yes/no answer to a question based on the boolean value of bool_answer
    """
    result = "yes" if bool_answer else "no"
    print(f'[DEBUG] image_patch.py, bool_to_yesno return: {_debug_summarize(result)}\n', flush=True)
    return result


def llm_query(query, context=None, long_answer=True, queues=None):
    print(f'[DEBUG] image_patch.py, llm_query()\n', flush=True)

    """Answers a text question using GPT-3. The input question is always a formatted string with a variable in it.

    Parameters
    ----------
    query: str
        the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
    """
    if long_answer:
        result = forward(model_name='gpt3_general', prompt=query, queues=queues)
    else:
        result = forward(model_name='gpt3_qa', prompt=[query, context], queues=queues)
    print(f'[DEBUG] image_patch.py, llm_query(func) return: {_debug_summarize(result)}\n', flush=True)
    return result


def process_guesses(prompt, guess1=None, guess2=None, queues=None):
    print(f'[DEBUG] image_patch.py, process_guesses()\n', flush=True)

    result = forward(model_name='gpt3_guess', prompt=[prompt, guess1, guess2], queues=queues)
    print(f'[DEBUG] image_patch.py, process_guesses return: {_debug_summarize(result)}\n', flush=True)
    return result


def coerce_to_numeric(string, no_string=False):
    print(f'[DEBUG] image_patch.py, coerce_to_numeric()\n', flush=True)

    """
    This function takes a string as input and returns a numeric value after removing any non-numeric characters.
    If the input string contains a range (e.g. "10-15"), it returns the first value in the range.
    # TODO: Cases like '25to26' return 2526, which is not correct.
    """
    if any(month in string.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                                                 'august', 'september', 'october', 'november', 'december']):
        try:
            result = dateparser.parse(string).timestamp().year
            print(f'[DEBUG] image_patch.py, coerce_to_numeric return: {_debug_summarize(result)}\n', flush=True)
            return result
        except:  # Parse Error
            pass

    try:
        # If it is a word number (e.g. 'zero')
        result = w2n.word_to_num(string)
        print(f'[DEBUG] image_patch.py, coerce_to_numeric return: {_debug_summarize(result)}\n', flush=True)
        return result
    except ValueError:
        pass

    # Remove any non-numeric characters except the decimal point and the negative sign
    string_re = re.sub("[^0-9\.\-]", "", string)

    if string_re.startswith('-'):
        string_re = '&' + string_re[1:]

    # Check if the string includes a range
    if "-" in string_re:
        # Split the string into parts based on the dash character
        parts = string_re.split("-")
        result = coerce_to_numeric(parts[0].replace('&', '-'))
        print(f'[DEBUG] image_patch.py, coerce_to_numeric return: {_debug_summarize(result)}\n', flush=True)
        return result
    else:
        string_re = string_re.replace('&', '-')

    try:
        # Convert the string to a float or int depending on whether it has a decimal point
        if "." in string_re:
            numeric = float(string_re)
        else:
            numeric = int(string_re)
    except:
        if no_string:
            raise ValueError
        # No numeric values. Return input
        result = string
        print(f'[DEBUG] image_patch.py, coerce_to_numeric return: {_debug_summarize(result)}\n', flush=True)
        return result
    result = numeric
    print(f'[DEBUG] image_patch.py, coerce_to_numeric return: {_debug_summarize(result)}\n', flush=True)
    return result
