#
# Copyright © 2024 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
import io
import sys
import argparse
import datetime
import gc
import json
import os.path
import torch
from argparse import Namespace
from pathlib import Path
from subprocess import Popen
from typing import Tuple, List
import tracemalloc
import time

import psutil

from transformers import AutoTokenizer, TextStreamer

from benchmark.battery import BatteryReport
from intel_npu_acceleration_library import NPUModelForCausalLM, int4

from intel_npu_acceleration_library.compiler import CompilerConfig

OUTPUT_FILE_PARAM = "output_file"
PROMPTS_FILE_PARAM = "prompts_file"


def build_arg_parser():
    parser = argparse.ArgumentParser(prog="llama3.py", description="Run llama3 on the NPU")
    parser.add_argument(f"--{OUTPUT_FILE_PARAM}",
                        help="Where to save text generation metrics")
    parser.add_argument(f"--{PROMPTS_FILE_PARAM}",
                        help="the file to read the prompts from, one per line. If not provided prompts must be "
                             "provided by stdin (keyboard)", default=None)
    return parser
def get_filenames(output_path_: str | None) -> Tuple[str, str]:
    now_ = datetime.datetime.now().isoformat(timespec='seconds')
    sanitized_now = now_.replace(":", "_").replace(",", ".")
    if output_path_:
        filename_ = output_path_
    else:
        filename_ = f"llama-benchmarks-{sanitized_now}"
    filename_monitoring_ = f"monitoring-process-{sanitized_now}.csv"
    return filename_, filename_monitoring_


def get_prompts(prompts_path_: str | None) -> List[str] | None:
    if prompts_path_:
        with open(prompts_path_) as prompts_file:
            return prompts_file.readlines()
    return None


def extract_command_line_args() -> Namespace:
    parser = build_arg_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    return parser.parse_args()


def generate(user_prompt, tokenizer_, streamer_, model_):

    streamer_.token_cache.clear()
    messages = [
        {
            "role": "system",
            "content": "You are an helpful chatbot that can provide information about different topics",
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    input_ids = (tokenizer_.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
                 .to(model_.device))

    terminators = [tokenizer_.eos_token_id, tokenizer_.convert_tokens_to_ids("<|eot_id|>")]

    start_time = datetime.datetime.now(datetime.UTC)
    btr_start: BatteryReport = BatteryReport.create_battery_report(Path("./battery_report_start.xml"))
    record = {}
    try:
        print(f"processing prompt {user_prompt}")
        outputs = model_.generate(input_ids, eos_token_id=terminators, do_sample=True, streamer=streamer_, )
        generated_token_array = outputs[0][len(input_ids[0]):]
        generated_tokens = "".join(tokenizer_.batch_decode(generated_token_array, skip_special_tokens=True))
        tokens_count = len(generated_token_array)
        record = {
            "generated_tokens": generated_tokens,
            "tokens_count": tokens_count
        }
    except Exception as err:
        error = err
        generated_tokens = f"Terminated with error: {type(error)}, {str(error)}"
        record = {
            "generated_tokens": generated_tokens,
            "tokens_count": -1,
            "tokens_per_second": -1
        }

    finally:
        end_time = datetime.datetime.now(datetime.UTC)
        elapsed_seconds = (end_time - start_time).total_seconds()
        btr_stop: BatteryReport = BatteryReport.create_battery_report(Path("./battery_report_end.xml"))
        if "tokens_per_second" not in record:
            record["tokens_per_second"] = record["tokens_count"]/elapsed_seconds
        record["battery_consumption_mWh"] = btr_start.current_mWh() - btr_stop.current_mWh()
        record["plugged_in"] = btr_stop.is_plugged_in()
        record["query"] = user_prompt
        btr_start.delete_battery_report()
        btr_stop.delete_battery_report()
    return record


if __name__ == '__main__':
    args_kv = extract_command_line_args()
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    compiler_conf = CompilerConfig(dtype=int4)
    print("Run inference with Llama3 on NPU\n")

    model = NPUModelForCausalLM.from_pretrained(model_id, use_cache=True, config=compiler_conf).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    now = datetime.datetime.now().isoformat(timespec='seconds')
    prompts = get_prompts(getattr(args_kv, PROMPTS_FILE_PARAM))
    get_prompts_from_file = bool(prompts)
    if get_prompts_from_file:
        prompts = iter(prompts)
    filename, filename_monitoring = get_filenames(getattr(args_kv,OUTPUT_FILE_PARAM))

    monitoring_process = Popen(f".venv\\Scripts\\python benchmark\\memory.py {filename_monitoring}")
    benchmark_file = None
    try:
        with open(f"llama3-benchmarks-{filename}.json", "a") as benchmark_file:
            benchmark_file.write("[\n")
            first_entry = True
            while True:
                prompt = next(prompts, None) if get_prompts_from_file else input(">")
                if not prompt:
                    benchmark_file.write("\n]")
                    break
                data_point = generate(prompt, tokenizer, streamer, model)
                if not first_entry:
                    benchmark_file.write(",")
                benchmark_file.write(f"{json.dumps(data_point)}\n")
                if first_entry:
                    first_entry = False
                if data_point["generated_tokens"].startswith("Terminated with error"):
                    print("model does not seem to be responding. Trying to send prompts to see if it restores itself")
                    tokenizer = AutoTokenizer.from_pretrained("llama3-8B-Instruct.tok",
                                                              **{"eos_token_id" : 128009})
                    streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
                    max_attempts = 4
                    i = 0
                    for i in range(max_attempts):
                        test = generate(f"Are you OK {i}?", tokenizer, streamer, model)
                        if test["generated_tokens"].startswith("Terminated with error"):
                            print(f"attempt number {i+1} did not work, trying again")
                            benchmark_file.write(",")
                            benchmark_file.write(json.dumps({"generated_tokens": f"attempt #{i+1} to restore model", "tokens_count": -1, "tokens_per_second": -1 }) + "\n")
                        else:
                            print("model seems to be working, resuming processing of the remaining prompts")
                            break

    finally:
        if benchmark_file is not None:
            benchmark_file.write("]")
            benchmark_file.close()
        process = psutil.Process(monitoring_process.pid)
        _ = [p.kill() for p in process.children(recursive=True)]
        process.kill()







