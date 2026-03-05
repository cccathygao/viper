import datetime
import json
import math
import os
import pathlib
from functools import partial
import warnings
import traceback
import re

import pandas as pd
import torch.multiprocessing as mp
from joblib import Memory
from num2words import num2words
import numpy as np
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from utils import seed_everything
import datasets
from datasets import general_postprocessing

# See https://github.com/pytorch/pytorch/issues/11201, https://github.com/pytorch/pytorch/issues/973
# Not for dataloader, but for multiprocessing batches
mp.set_sharing_strategy('file_system')
queue_results = None

cache = Memory('cache/' if config.use_cache else None, verbose=0)
runs_dict = {}
seed_everything()
console = Console(highlight=False)


def my_collate(batch):
    # Avoid stacking images (different size). Return everything as a list
    to_return = {k: [d[k] for d in batch] for k in batch[0].keys()}
    return to_return


def _extract_python_code_blocks(response_text):
    """
    Extract the first Python code block from a model response.
    Supports both <code>...</code> wrapped blocks (PyVision-style) and bare ```python fences.
    Rejects invalid blocks that redefine ImagePatch or copy prompt content.
    """
    # Prefer explicitly wrapped <code>...</code> blocks
    wrapped_pattern = re.compile(
        r"<code>.*?```python(.*?)```.*?</code>",
        re.DOTALL | re.IGNORECASE,
    )
    m = wrapped_pattern.search(response_text)
    if m:
        block = m.group(1).strip()
        if _is_valid_code_body(block):
            return [block]
        return []

    # Fallback: any python fenced block
    bare_pattern = re.compile(r"```python(.*?)```", re.DOTALL | re.IGNORECASE)
    m = bare_pattern.search(response_text)
    if m:
        block = m.group(1).strip()
        if _is_valid_code_body(block):
            return [block]
        return []

    return []


def _is_valid_code_body(code):
    """
    Reject code that redefines ImagePatch or copies prompt (e.g. class ImagePatch).
    Valid code should only USE ImagePatch, not define it.
    """
    if not code or len(code.strip()) < 10:
        return False
    # Invalid: model regurgitated the API doc
    if "class ImagePatch" in code or "def ImagePatch" in code:
        return False
    if '"""A Python class containing a crop' in code or "cropped_image : array_like" in code:
        return False
    # Reject trivial placeholders or comment-only stubs
    if "# your code here" in code or "<python code here" in code:
        return False
    lines = [l for l in code.split("\n") if l.strip()]
    if not lines:
        return False
    if all(l.lstrip().startswith("#") for l in lines):
        return False
    return True


def _build_react_continuation_prompt(exec_result):
    """
    Short prompt for ReAct turn 2+: the previous code ran, here is the result.
    Ask for either more code or final answer.
    """
    return (
        f"<interpreter>\nResult:\n{exec_result}\n</interpreter>\n\n"
        "If the result above answers the query, output the final answer in this format (no more code):\n"
        "<answer>\\n\\boxed{{'your answer here'}}\\n</answer>\n\n"
        "If you need to run more code to refine your answer, respond ONLY with a <code> block containing a "
        "```python ... ``` fenced snippet with executable code, and no explanations outside the code.\n"
    )


def run_program(parameters, queues_in_, input_type_, retrying=False):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno
    from video_segment import VideoSegment

    global queue_results

    code, sample_id, image, possible_answers, query = parameters

    # Clean the code from LLM
    import re
    code = re.sub(r'```python\n?', '', code)
    code = re.sub(r'```\n?', '', code)
    code = code.strip('`').strip()
    
    # Remove the def execute_command line so only the body remains
    lines = code.split('\n')
    code = '\n'.join([l for l in lines if 'def execute_command' not in l])

    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query, ' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match, question=None):\n' \
                  f'    if question is None: question = query\n' \
                  f'    # Answer is:\n'
    code = code_header + code

    print(f'\n[DEBUG] Running sample {sample_id}', flush=True)
    print(f'[DEBUG] Code to run:\n{code}\n', flush=True)

    try:
        exec(compile(code, 'Codex', 'exec'), globals())
    except Exception as e:
        print(f'Sample {sample_id} failed at compilation time with error: {e}')
        try:
            with open(config.fixed_code_file, 'r') as f:
                fixed_code = f.read()
            code = code_header + fixed_code 
            exec(compile(code, 'Codex', 'exec'), globals())
        except Exception as e2:
            print(f'Not even the fixed code worked. Sample {sample_id} failed at compilation time with error: {e2}')
            return None, code

    queues = [queues_in_, queue_results]

    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)

    try:
        result = globals()[f'execute_command_{sample_id}'](
            # Inputs to the function
            image, possible_answers, query,
            # Classes to be used
            image_patch_partial, video_segment_partial,
            # Functions to be used
            llm_query_partial, bool_to_yesno, distance, best_image_match)
    except Exception as e:
        # print full traceback
        traceback.print_exc()
        if retrying:
            return None, code
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        # Retry again with fixed code
        new_code = "["  # This code will break upon execution, and it will be caught by the except clause
        result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_,
                             retrying=True)[0]

    # The function run_{sample_id} is defined globally (exec doesn't work locally). A cleaner alternative would be to
    # save it in a global dict (replace globals() for dict_name in exec), but then it doesn't detect the imported
    # libraries for some reason. Because defining it globally is not ideal, we just delete it after running it.
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']  # If it failed to compile the code, it won't be defined
    return result, code


def worker_init(queue_results_):
    global queue_results
    index_queue = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[index_queue]


def main():
    mp.set_start_method('spawn', force=True)

    from vision_processes import queues_in, finish_all_consumers, forward, manager
    from datasets import get_dataset

    batch_size = config.dataset.batch_size
    num_processes = min(batch_size, 50)

    if config.multiprocessing:
        queue_results_main = manager.Queue()
        queues_results = [manager.Queue() for _ in range(batch_size)]
    else:
        queue_results_main = None
        queues_results = [None for _ in range(batch_size)]

    model_name_codex = 'codellama' if config.codex.model == 'codellama' else 'codex'
    codex = partial(forward, model_name=model_name_codex, queues=[queues_in, queue_results_main])

    if config.clear_cache:
        cache.clear()

    if config.wandb:
        import wandb
        wandb.init(project="viper", config=OmegaConf.to_container(config))
        # log the prompt file
        wandb.save(config.codex.prompt)

    dataset = get_dataset(config.dataset)

    with open(config.codex.prompt) as f:
        base_prompt = f.read().strip()

    codes_all = None
    if config.use_cached_codex:
        results = pd.read_csv(config.cached_codex_path)
        codes_all = [r.split('# Answer is:')[1] for r in results['code']]
    # python -c "from joblib import Memory; cache = Memory('cache/', verbose=0); cache.clear()"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=my_collate)
    input_type = dataset.input_type

    all_results = []
    all_answers = []
    all_codes = []
    all_ids = []
    all_queries = []
    all_img_paths = []
    all_possible_answers = []
    all_query_types = []

    # Multi-turn ReAct: only when config.codex.multi_turn is True and model supports chat.
    use_react = (
        getattr(config.codex, "multi_turn", False)
        and getattr(config.codex, "model", None) is not None
        and (
            "gpt-4" in str(config.codex.model)
            or "gpt-3.5" in str(config.codex.model)
            or "qwen" in str(config.codex.model).lower()
        )
    )

    with mp.Pool(processes=num_processes, initializer=worker_init, initargs=(queues_results,)) \
            if config.multiprocessing and not use_react else open(os.devnull, "w") as pool:
        try:
            n_batches = len(dataloader)

            for i, batch in tqdm(enumerate(dataloader), total=n_batches):

                # Combine all queries and get Codex predictions for them
                # TODO compute Codex for next batch as current batch is being processed

                if not config.use_cached_codex and not use_react:
                    codes = codex(prompt=batch['query'], base_prompt=base_prompt, input_type=input_type,
                                  extra_context=batch['extra_context'])

                elif not config.use_cached_codex and use_react:
                    # In ReAct mode we do not pre-generate code; we will call the chat model
                    # inside the per-sample loop below.
                    codes = None

                else:
                    codes = codes_all[i * batch_size:(i + 1) * batch_size]  # If cache

                # Run the code / multi-turn loop
                if config.execute_code:
                    if use_react:
                        # ReAct-style multi-turn: turn 1 uses CodexModel (same as single-turn), turn 2+ use chat.
                        results = []
                        all_intermediate_logs = []  # Accumulate logs for all samples
                        max_turns = getattr(config.codex, "max_turns", 5)
                        from vision_models import codex_react_chat

                        ec_list = batch.get('extra_context', [''] * len(batch['query']))
                        for c_idx, (sample_id, img, possible_answers, query, extra_context) in enumerate(
                                zip(batch['sample_id'], batch['image'], batch['possible_answers'],
                                    batch['query'], ec_list)):
                            ec = str(extra_context) if extra_context is not None else ''

                            print(f'[DEBUG] main_batch.py, ReAct sample {sample_id} (turn 1 using CodexModel)', flush=True)

                            # Turn 1: Use CodexModel via forward (same flow as single-turn) for proper code
                            turn1_code = codex(
                                prompt=[query],
                                base_prompt=base_prompt,
                                input_type=input_type,
                                extra_context=[ec],
                            )[0]

                            # Execute turn 1 code
                            exec_result, full_code = run_program(
                                [turn1_code, sample_id, img, possible_answers, query],
                                queues_in,
                                input_type,
                            )
                            turn_codes = [turn1_code]
                            final_response = str(exec_result) if exec_result is not None else ""

                            print(f'[DEBUG] main_batch.py, ReAct sample {sample_id} turn 1 result: {exec_result}', flush=True)

                            # Turn 2..max_turns: continuation via chat
                            messages = [
                                {"role": "user", "content": base_prompt.replace("INSERT_QUERY_HERE", query)
                                 .replace("INSERT_TYPE_HERE", input_type)
                                 .replace("EXTRA_CONTEXT_HERE", str(ec))},
                                {"role": "assistant", "content": turn1_code},
                                {"role": "user", "content": _build_react_continuation_prompt(exec_result)},
                            ]

                            intermediate_log = [
                                {"turn": 1, "messages": [{"role": m["role"], "message": m["content"]} for m in messages]}
                            ]

                            for turn in range(1, max_turns):
                                print(f'[DEBUG] main_batch.py, ReAct sample {sample_id} turn {turn + 1}', flush=True)
                                try:
                                    response_text = codex_react_chat(messages)
                                except Exception as e:
                                    console.print(f"Error in ReAct chat for sample {sample_id}, turn {turn + 1}: {e}")
                                    break

                                intermediate_log.append({
                                    "turn": turn + 1,
                                    "messages": [{"role": m["role"], "message": m["content"]} for m in messages]
                                })
                                _current_log = all_intermediate_logs + [{"sample_id": sample_id, "turns": intermediate_log}]
                                with open("intermediate_messages.json", "w", encoding="utf-8") as f:
                                    json.dump(_current_log, f, indent=2, ensure_ascii=False)
                                print(f'[DEBUG] main_batch.py, ReAct sample {sample_id} turn {turn + 1} response: {response_text}', flush=True)

                                python_blocks = _extract_python_code_blocks(response_text)

                                if python_blocks:
                                    code_snippet = python_blocks[0]
                                    turn_codes.append(code_snippet)
                                    exec_result, full_code = run_program(
                                        [code_snippet, sample_id, img, possible_answers, query],
                                        queues_in,
                                        input_type,
                                    )
                                    final_response = str(exec_result) if exec_result is not None else final_response
                                    messages.append({"role": "assistant", "content": response_text})
                                    messages.append({"role": "user", "content": _build_react_continuation_prompt(exec_result)})
                                    if turn == max_turns - 1:
                                        print(f'[DEBUG] main_batch.py, ReAct sample {sample_id} turn {turn + 1} max turns reached', flush=True)
                                        break
                                else:
                                    # No code block: treat as final answer, try to extract \boxed{}
                                    boxed = re.search(r'\\boxed\{([^}]*)\}', response_text)
                                    if boxed:
                                        final_response = boxed.group(1).strip().strip("'\"")
                                    elif "<answer>" in response_text:
                                        # Use content between <answer> and </answer>
                                        ans = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
                                        if ans:
                                            final_response = ans.group(1).strip()
                                    break

                                print(f'[DEBUG] main_batch.py, ReAct sample {sample_id} turn {turn + 1} final response: {final_response}', flush=True)
                                        
                            all_intermediate_logs.append({"sample_id": sample_id, "turns": intermediate_log})
                            with open("intermediate_messages.json", "w", encoding="utf-8") as f:
                                json.dump(all_intermediate_logs, f, indent=2, ensure_ascii=False)
                            combined_code = "\n\n# ---- NEXT TOOL CALL ----\n\n".join(turn_codes) if turn_codes else ""
                            results.append((final_response, combined_code))

                    else:
                        if not config.multiprocessing:
                            # Otherwise, we would create a new model for every process
                            results = []
                            for c, sample_id, img, possible_answers, query in \
                                    zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query']):
                                result = run_program([c, sample_id, img, possible_answers, query], queues_in, input_type)
                                results.append(result)
                        else:
                            results = list(pool.imap(partial(
                                run_program, queues_in_=queues_in, input_type_=input_type),
                                zip(codes, batch['sample_id'], batch['image'], batch['possible_answers'], batch['query'])))
                else:
                    if use_react:
                        warnings.warn(
                            "execute_code is False but ReAct mode is enabled. "
                            "ReAct requires executing tool code; falling back to no-op results."
                        )
                        results = [("", "") for _ in batch['sample_id']]
                    else:
                        results = [(None, c) for c in codes]
                        warnings.warn("Not executing code! This is only generating the code. We set the flag "
                                      "'execute_code' to False by default, because executing code generated by a language "
                                      "model can be dangerous. Set the flag 'execute_code' to True if you want to execute "
                                      "it.")

                all_results += [r[0] for r in results]
                all_codes += [r[1] for r in results]
                all_ids += batch['sample_id']
                all_answers += batch['answer']
                all_possible_answers += batch['possible_answers']
                all_query_types += batch['query_type']
                all_queries += batch['query']
                all_img_paths += [dataset.get_sample_path(idx) for idx in batch['index']]

                # Per-sample correctness (when log_per_sample is True)
                if getattr(config, 'log_per_sample', False):
                    for j in range(len(results)):
                        pred = results[j][0]
                        gt = batch['answer'][j]
                        sid = batch['sample_id'][j]
                        idx = batch['index'][j]
                        if hasattr(dataset, 'post_process'):
                            try:
                                p_clean = dataset.post_process(pred, idx)
                                g_clean = dataset.post_process(str(gt), idx)
                                correct = (p_clean == g_clean)
                            except Exception:
                                correct = (general_postprocessing(pred) == str(gt).strip())
                        else:
                            correct = (general_postprocessing(pred) == str(gt).strip())
                        status = "correct" if correct else "incorrect"
                        console.print(f"  Sample {sid}: {status} (pred={pred!r}, gt={gt!r})")

                if i % config.log_every == 0:
                    try:
                        accuracy = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
                        console.print(f'Accuracy at Batch {i}/{n_batches}: {accuracy}')
                    except Exception as e:
                        console.print(f'Error computing accuracy: {e}')

        except Exception as e:
            # print full stack trace
            traceback.print_exc()
            console.print(f'Exception: {e}')
            console.print("Completing logging and exiting...")

    try:
        accuracy = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
        console.print(f'Final accuracy: {accuracy}')
    except Exception as e:
        print(f'Error computing accuracy: {e}')

    if config.save:
        results_dir = pathlib.Path(config['results_dir'])
        results_dir = results_dir / config.dataset.split
        results_dir.mkdir(parents=True, exist_ok=True)
        if not config.save_new_results:
            filename = 'results.csv'
        else:
            existing_files = list(results_dir.glob('results_*.csv'))
            if len(existing_files) == 0:
                filename = 'results_0.csv'
            else:
                filename = 'results_' + str(max([int(ef.stem.split('_')[-1]) for ef in existing_files if
                                                 str.isnumeric(ef.stem.split('_')[-1])]) + 1) + '.csv'
        print('Saving results to', filename)
        df = pd.DataFrame([all_results, all_answers, all_codes, all_ids, all_queries, all_img_paths,
                           all_possible_answers]).T
        df.columns = ['result', 'answer', 'code', 'id', 'query', 'img_path', 'possible_answers']
        # make the result column a string
        df['result'] = df['result'].apply(str)
        df.to_csv(results_dir / filename, header=True, index=False, encoding='utf-8')
        # torch.save([all_results, all_answers, all_codes, all_ids, all_queries, all_img_paths], results_dir/filename)

        if config.wandb:
            wandb.log({'accuracy': accuracy})
            wandb.log({'results': wandb.Table(dataframe=df, allow_mixed_types=True)})

    finish_all_consumers()


if __name__ == '__main__':
    main()
