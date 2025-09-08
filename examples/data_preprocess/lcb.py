# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/lcb")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    print(f"Loading the LiveCodeBench dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset("livecodebench/code_generation_lite", version_tag="release_v6")
    dataset = dataset.filter(lambda example: example['public_test_cases'] != '[]')
    dataset = dataset.map(remove_columns=['private_test_cases', 'platform', 'question_title', 'contest_date', 'difficulty'])

    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            public_test_cases = json.loads(
                example['public_test_cases']
            )
            private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(example['private_test_cases'].encode("utf-8"))  # type: ignore
                    )
                )
            )
            eval_types = ["call" if r['testtype'] == "functional" else "stdio" for r in public_test_cases] + ["call" if r['testtype'] == "functional" else "stdio" for r in private_test_cases]
            inputs = [r['input'] for r in public_test_cases] + [r['input'] for r in private_test_cases]
            outputs = [r['output'] for r in public_test_cases] + [r['output'] for r in private_test_cases] 
            metadata = json.loads(example['metadata'])

            assert all(x == eval_types[0] for x in eval_types), "Evaluation is a mix of both!"

            if len(example['starter_code']):
                example['question_content'] += '\n\n' + example['starter_code']
            return {
                "prompt": [{"role": "user", "content": example['question_content']}],
                'code': None,
                'ground_truth': {
                    'eval_type': eval_types[0],
                    "fn_name": metadata.get("func_name", None),
                    'input_output': {
                        "inputs": inputs,
                        "outputs": outputs
                    }
                }
            }

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    print(test_dataset)
    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    # Save one example as JSON for reference
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
