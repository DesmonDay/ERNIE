# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
read dataset json for vl_sft
"""

import gzip
import json
import logging
import math
import random
import re
from collections import OrderedDict, namedtuple
from copy import deepcopy

import numpy as np
import paddle

IDTYPES_2_ID = {"text": 0, "image": 1, "video": 2, "audio": 3}
IMAGETYPES_2_ID = {"image": 0, "video": 1, "padded_image": 2}
DATATYPE_2_ID = {"mm": 0, "lm": 1, "audio": 2}

from contextlib import contextmanager

from paddle import distributed as dist
from paddle.distributed import fleet
from paddle.io import IterableDataset

log = logging.getLogger(__name__)


def fetch_worker():
    """
    获取当前工作进程的信息。

    Args:
        无

    Returns:
        tuple: 包含两个元素的元组，分别表示工作进程的总数和工作进程的ID。

    """
    worker_info = paddle.io.get_worker_info()
    if worker_info is None:
        num_workers = 1
        worker_id = 0
    else:
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
    return num_workers, worker_id


def make_seed(*args):
    """
    生成一个基于多个参数的哈希种子。

    Args:
        *args: 任意数量的参数，可以是任何可哈希对象。

    Returns:
        int: 基于参数生成的哈希种子。

    """
    seed = 0
    for arg in args:
        seed = hash(seed * 1e3 + hash(arg)) & 0x7FFFFFFF
    return seed


def get_global_data_info():
    """
    获取全局数据的信息。

    Args:
        无。

    Returns:
        tuple: 包含两个元素的元组，分别表示全局数据的排名和全局数据的大小。

    """
    if dist.get_world_size() > 1:
        hcg = fleet.get_hybrid_communicate_group()
        dp_rank = hcg.get_data_parallel_rank()
        dp_size = hcg.get_data_parallel_world_size()
        sharding_rank = hcg.get_sharding_parallel_rank()
        sharding_size = hcg.get_sharding_parallel_world_size()
        global_data_rank = dp_rank * sharding_size + sharding_rank
        global_data_size = dp_size * sharding_size
        log.info(f"global_data_rank: {global_data_rank}, global_data_size: {global_data_size}")
        return global_data_rank, global_data_size
    else:
        return 0, 1


def equal_shard(datasets, rank, world_size):
    """
    如果有权重，根据权重概率累计概率相等的原则切分 train part.
    没有权重直接均分parts.
    args:
        datasets: List[ExampleSetSingleDataSource]
        rank: int
        world_size: int
    """
    assert len(datasets) >= world_size, f"#filelist={len(datasets)} < world_size{world_size}"
    if world_size == 1:
        return datasets

    if datasets[0]['weight'] is None:
        ran = np.array_split(np.arange(len(datasets)), world_size)[rank]
        s, e = ran[0], ran[-1]
        shard = datasets[s : e + 1]
        return shard
    buckets = [[] for _ in range(world_size)]

    bucketsize = np.zeros(len(buckets), dtype="float64")

    datasets = sorted(datasets, key=lambda d: d['weight'], reverse=True)  # 先分大part，或许有利于均匀分发？
    for d in datasets:
        this_bucket = np.argmin(bucketsize)
        buckets[this_bucket].append(d)
        bucketsize[this_bucket] += d['weight']

    log.info(f"sharding dataset according to prob, group vs probs={[sum([rr['weight'] for rr in r])for r in buckets]}")
    bucketsize = bucketsize[rank]
    diff = bucketsize - (1 / world_size)
    log.info(f"unable to perfect shard. prob sum of this bucket:{bucketsize}, diff to perfect portion:{diff}")
    assert len(buckets) == world_size, f"#ret={len(buckets)} prob not normalized:{[d['weight'] for d in datasets]}"
    return buckets[rank]


@contextmanager
def open_file(filename):
    """Construct a file handler that can read normal or gzip-compressed files.

    The handler automatically detects compression based on file extension.

    Args:
        filename (str): Path to the target file, which may end with .gz for gzip compression.

    Returns:
        Generator[TextIO]: A file object generator that yields lines from the file.
    """
    if filename.endswith(".gz"):
        fp = gzip.open(filename, "rt")
    else:
        fp = open(filename)
    yield fp
    fp.close()


def contains_markdown_table(text):
    """
    判断给定的文本中是否包含Markdown格式的表格。

    Args:
        text (str): 要检查的文本。

    Returns:
        bool: 如果文本中包含Markdown格式的表格，则返回True；否则返回False。

    """
    lines = text.strip().split('\n')
    for i in range(1, len(lines) - 1):
        separator_line = lines[i].strip()
        # 检查是否是表格分隔线
        if re.match(r'^(\s*\|?\s*:?-+:?\s*\|)+\s*$', separator_line):
            header_line = lines[i - 1].strip()
            body_line = lines[i + 1].strip()
            # 检查上下行是否可能是表格的内容行
            if '|' in header_line and '|' in body_line:
                # 检查每行的 | 数量是否一致
                header_pipes = header_line.count('|')
                body_pipes = body_line.count('|')
                separator_pipes = separator_line.count('|')
                if header_pipes == body_pipes == separator_pipes:
                    return True
    return False


def process_markdown_table(table: str) -> str:
    """
    处理 Markdown 表格，格式化表格中的单元格和对齐行。

    Args:
        table (str): 待处理的 Markdown 表格字符串。

    Returns:
        str: 处理后的 Markdown 表格字符串。

    Note:
        可能存在一些bug。
    """
    # may some bug
    if not contains_markdown_table(table):
        return table

    # 去掉每个单元格前后的多余空格，并确保每个单元格前后至少有一个空格
    def format_row(row: str) -> str:
        return '| ' + ' | '.join(cell.strip() for cell in row.split('|')[1:-1]) + ' |'

    # 将 "---" 以上的字符限制为最多三个 '-'，适用于对齐行
    def format_alignment_row(row: str) -> str:
        # maybe have some bugs
        line = '| ' + ' | '.join(re.sub(r'-{3,}', '---', cell.strip()) for cell in row.split('|')[1:-1]) + ' |'
        line = line.replace(':---:', ':-:')
        line = line.replace(':--:', ':-:')
        line = line.replace(':---', ':--')
        line = line.replace('---:', '--:')

        return line

    # 处理每一行
    lines = table.splitlines()
    processed_lines = []

    for line in lines:
        if re.match(r'^\s*\|', line):  # 判断是否是表格行
            if '-' in line:  # 对齐行处理
                processed_lines.append(format_alignment_row(line))
            else:  # 正常行处理
                processed_lines.append(format_row(line))
        else:
            processed_lines.append(line)  # 非表格行原样输出

    return '\n'.join(processed_lines)


Example = namedtuple("Example", ["meta", "task", "prompt", "labels", "src"])


class ExampleSet:
    """use to manage one json file_name data"""

    def __init__(
        self,
        file_name,
        src,
        prompt_list,
        shuffle_json: bool = False,
        process_fn=None,
    ):

        self._file_name = file_name
        self.src = int(src)
        self._process_fn = process_fn
        self._shuffle_json = shuffle_json
        self.prompt_list = prompt_list
        self.process_fn = process_fn
        log.info(f"loading json {self._file_name}")
        with open_file(self._file_name) as fin:
            self.exs = json.load(fin)

    def __len__(self):
        return len(self.exs)

    def __iter__(self):
        if self._shuffle_json:
            np.random.shuffle(self.exs)
        idx = 0
        for meta in self.exs:
            ret = Example(
                meta=meta,
                src=self.src,
                task="lm",
                prompt=None,
                labels=None,
            )

            ret = self.process_fn(ret)
            ret.update(data_id=idx, example_id=idx)
            idx += 1

            yield ret


class ExampleSetSingleDataSource(IterableDataset):
    """use to manage multiple json file_names"""

    def __init__(
        self,
        src,
        file_list,
        prompt_list,
        seqlen,
        weight=None,
        seed: int = 42,
        shuffle: bool = False,
        shuffle_files: bool = False,
        shuffle_json: bool = False,
        num_consecutive: int = 1,
        dataset_type=None,
        name=None,
        batch_size=1,
        dp_rank=None,
        dp_size=None,
        process_fn=None,
        dp_shard_data=True,
    ):

        self.file_list = file_list if isinstance(file_list, list) else [file_list]

        self.src = int(src)

        self.prompt_list = prompt_list

        self.dataset_type = dataset_type
        self.name = name
        self.process_fn = process_fn

        self.seqlen = seqlen
        self.weight = weight
        self.seed = seed
        self.shuffle_files = shuffle_files
        self.shuffle_json = shuffle_json

        self.length = 0
        self.epoch = 0
        self.batch_size = batch_size
        self._sub_datasets = []
        self._load_single_source()

    def _load_single_source(self):
        log.info(f"loading data source:{self.src}, weight={self.weight},json_lists={self.file_list}")
        for path in self.file_list:
            example = ExampleSet(
                file_name=path,
                src=self.src,
                prompt_list=self.prompt_list,
                shuffle_json=self.shuffle_json,
                process_fn=self.process_fn,
            )
            self._sub_datasets.append(example)
            self.length += len(example)

    def __len__(self):
        return math.ceil(self.length / self.batch_size) * self.batch_size

    def __iter__(self):
        while True:
            if self.shuffle_files:
                sub_datasets = self._sub_datasets
                np.random.shuffle(self._sub_datasets)
            else:
                sub_datasets = self._sub_datasets
            for ds in sub_datasets:
                yield from ds


class SFTMultimodalDatasetJson(IterableDataset):
    """
    SFT Multimodal Dataset Json
    """

    def __init__(
        self,
        dataset_config,
        tokenizer,
        image_preprocess,
        seed,
        image_token_len,
        seqlen,
        special_token_loss_mask_ratio=None,
        use_prompt=False,
        adaptive_resolution=False,
        im_patch_id=10000000,
        use_crop=False,
        crop_tile_option="16,24",
        crop_tile_rate="0.95,0.05",
        dp_rank=None,
        dp_size=None,
        batch_size=1,
        data_processor=None,
        **kwargs,
    ):
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.image_token_len = image_token_len
        # self.im_patch_id = len(self.vocab)
        self.im_patch_id = im_patch_id
        # img_token = tokenizer.special_tokens_map.get("img_token", "<mask:1>")
        # self.im_patch_id= self.vocab[img_token]
        self.image_preprocess = image_preprocess
        self.adaptive_resolution = adaptive_resolution
        self.use_crop = use_crop
        if self.use_crop:  # 1
            self.crop_tile_option = [int(op.strip()) for op in crop_tile_option.strip().split(',')]
            self.crop_tile_rate = [float(op.strip()) for op in crop_tile_rate.strip().split(',')]
            assert len(self.crop_tile_option) == len(
                self.crop_tile_rate
            ), 'crop_tile_option and crop_tile_rate len are no match'
            log.info(f'[Use Crop] crop_tile_option={self.crop_tile_option} crop_tile_rate={self.crop_tile_rate}')

        self.wo_vit = False

        self.grouding_loss_mask_rng = random.Random(seed)
        self.special_token_loss_mask_ratio = special_token_loss_mask_ratio

        self.seed = seed
        self.seqlen = seqlen
        self.use_prompt = use_prompt

        self.sys_start_token = self.tokenizer.special_tokens_map.get("sys_start_token", "<mask:4>")
        self.sys_end_token = self.tokenizer.special_tokens_map.get("sys_end_token", "<mask:5>")
        self.eos_token = self.tokenizer.special_tokens_map.get("eos_token", "</s>")
        self.cls_token = self.tokenizer.special_tokens_map.get("cls_token", "<mask:0>")
        self.sep_token = self.tokenizer.special_tokens_map.get("sep_token", "<|endofprompt|>")
        log.info(f"cls token: {self.cls_token}, sep token: {self.sep_token}, eos token: {self.eos_token}")

        # IMAGE相关token，所有45V模型统一special token
        # "<|IMAGE_START|>", "<|IMAGE_END|>" "<|CROP_COL_SEP|>", "<|CROP_ROW_SEP|>"
        self.img_start_token = "<|IMAGE_START|>"
        self.img_end_token = "<|IMAGE_END|>"
        self.crop_col_sep_token = "<|CROP_COL_SEP|>"
        self.crop_row_sep_token = "<|CROP_ROW_SEP|>"
        log.info(
            f"""img_start_token: {self.img_start_token},
            img_end_token: {self.img_end_token},
            crop_col_sep_token: {self.crop_col_sep_token},
            crop_row_sep_token: {self.crop_row_sep_token}"""
        )

        self.cls_token_id = self.tokenizer._convert_token_to_id(self.cls_token)
        self.sep_token_id = self.tokenizer._convert_token_to_id(self.sep_token)
        self.eos_token_id = self.tokenizer._convert_token_to_id(self.eos_token)

        log.info(f"cls token: {self.cls_token_id}, sep token: {self.sep_token_id}, eos token: {self.eos_token_id}")

        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(self.img_start_token)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(self.img_end_token)
        self.crop_col_sep_token_id = self.tokenizer.convert_tokens_to_ids(self.crop_col_sep_token)
        self.crop_row_sep_token_id = self.tokenizer.convert_tokens_to_ids(self.crop_row_sep_token)
        self.sys_start_id = self.tokenizer.convert_tokens_to_ids(self.sys_start_token)
        self.sys_end_id = self.tokenizer.convert_tokens_to_ids(self.sys_end_token)
        log.info(
            f"""img_start_token_id: {self.img_start_token_id},
            img_end_token_id: {self.img_end_token_id},
            crop_col_sep_token_id: {self.crop_col_sep_token_id},
            crop_row_sep_token: {self.crop_row_sep_token_id}"""
        )
        log.info(f"sys_start_id: {self.sys_start_id}, sys_end_id: {self.sys_end_id}")
        self.tokenizer.cls_token_id = self.cls_token_id
        self.tokenizer.sep_token_id = self.sep_token_id
        self.tokenizer.eos_token_id = self.eos_token_id

        self.data_processor = data_processor

        self.task_group = {}
        self.src_id_list = []
        self.weight_list = []
        self.batch_size = batch_size
        self.data_rank, self.data_size = dp_rank, dp_size
        self.num_workers, self.worker_id = fetch_worker()
        self.global_worker_id = self.data_rank * self.num_workers + self.worker_id
        self.local_seed = make_seed(self.seed, self.global_worker_id)
        self.length = 0
        self.epoch = 0

    def load_files_info(self, filelist):
        """
        加载文件列表信息。

        Args:
            filelist (str): 文件列表的路径。

        Returns:
            list: 包含文件信息的行列表。

        Raises:
            None

        """
        with open_file(filelist) as f:
            lines = f.read().strip().split("\n")
        if len(lines) <= self.data_size:
            log.warning(
                f"""Expected filelist size >= data_size, but got {len(lines)} vs. {self.data_size},
                different nodes will be assigned the same data"""
            )
        return lines

    def reformat_meta(self, meta):
        """reformat_meta"""
        text_list = "".join([text["text"] for text in meta["text_info"] if text["tag"] == "no_mask"])
        if type(text_list) == str and "<think>" in text_list and "</think>" in text_list:
            # think-data
            pass
        else:
            meta["prefix"] = "<think>\n\n</think>\n\n"
        return meta

    def _load(self, use_shard=True, shuffle_files=True, shuffle_json=True):
        single_source_list = []
        source_sub_info = {}

        weight_sum = sum([dataset_info["weight"] for dataset_info in self.dataset_config])
        log.info(f"weight_sum of all src: {weight_sum}")

        for dataset_info in self.dataset_config:
            src = dataset_info["src_id"]
            filelist = dataset_info["filelist"]

            weight = dataset_info["weight"]

            if weight == 0:
                continue

            name = dataset_info["name"]
            process_fn = None
            dataset_type = dataset_info.get("dataset_type", None)
            if dataset_type is not None and dataset_type == 'video':
                process_fn = self.example_to_feature_stage3_video
            else:
                process_fn = self.example_to_feature_stage3

            prompt_file = dataset_info.get("prompt_file", None)
            if prompt_file is not None:
                with open(prompt_file) as f:
                    prompt_list = f.read().strip().split("\n")
            else:
                prompt_list = None
            file_lists = self.load_files_info(filelist)

            source_sub_info[int(src)] = {
                'prompt_list': prompt_list,
                'weight': weight / weight_sum,
                'name': name,
                'dataset_type': dataset_type,
                'process_fn': process_fn,
                'file_list': [],
            }

            weight = weight / len(file_lists) / weight_sum

            for file_list in file_lists:
                single_source_info = {
                    "src": int(src),
                    "file_list": file_list,
                    "weight": weight,
                }
                single_source_list.append(single_source_info)

        if use_shard:
            example_per_dp = equal_shard(single_source_list, self.data_rank, self.data_size)
        else:
            example_per_dp = single_source_list

        log.debug(
            f"using source shard, # files before shard={len(single_source_list)}, after shard={len(example_per_dp)}"
        )

        for example in example_per_dp:
            source_sub_info[example['src']]['file_list'].append(example['file_list'])

        for src, info in source_sub_info.items():
            file_list = info.pop('file_list')
            if len(file_list) == 0:
                continue
            part = ExampleSetSingleDataSource(
                src=src,
                file_list=file_list,
                seqlen=self.seqlen,
                seed=self.seed,
                shuffle_files=shuffle_files,
                shuffle_json=shuffle_json,
                batch_size=self.batch_size,
                dp_rank=self.data_rank,
                dp_size=self.data_size,
                **info,
            )

            self.task_group[part.src] = iter(part)
            self.weight_list.append(part.weight)
            self.src_id_list.append(part.src)
            self.length += len(part)

        weight_sum = sum(self.weight_list)
        self.weight_list = [item / weight_sum for item in self.weight_list]

    def example_to_feature_stage3(self, example):
        """
        将示例转换为特征表示（第三阶段）。

        Args:
            example (dict): 包含元数据和原始数据的示例字典。

        Returns:
            OrderedDict: 包含处理后的特征数据的字典。

        Raises:
            Exception: 在处理过程中出现异常时抛出。

        """
        try:
            meta = example.meta
            raw_meta = deepcopy(meta)

            # Fix Table
            for idx, _ in enumerate(raw_meta["text_info"]):
                raw_meta["text_info"][idx]["text"] = raw_meta["text_info"][idx]["text"].strip()
                if raw_meta["text_info"][idx]["tag"] == "no_mask":
                    raw_text = raw_meta["text_info"][idx]["text"]
                    raw_meta = self.reformat_meta(raw_meta)
                    raw_meta["text_info"][idx]["text"] = process_markdown_table(raw_meta["text_info"][idx]["text"])
                    # if raw_text != raw_meta["text_info"][idx]["text"]:
                    #     log.info(f"Reformat-Data.")
                    # if " " * 32 in raw_meta["text_info"][idx]["text"]:
                    #     raise Exception("Too many space.")

            result = self.data_processor.process(raw_meta)
            assert len(result) == 1, f"result: {len(result)}, {result}"
            result = result[0]
            assert len(result["images"]) > 0, "images is None"
            assert result["token_type_ids"] is not None, "token_type_ids is None"
            assert result["image_type_ids"] is not None, "image_type_ids is None"

            return OrderedDict(
                src_id=example.src,
                part_id=0,
                images=result["images"],
                input_ids=result["input_ids"],
                labels=result["labels"],
                token_type_ids=result["token_type_ids"],
                image_type_ids=result["image_type_ids"],
                grid_thw=result["grid_thw"],
                position_ids=result["position_ids"],
                data_not_valid=0,
                data_type=DATATYPE_2_ID["mm"],
            )
        except Exception as e:
            log.info(f"e: {e}")
            features = OrderedDict(
                src_id=example.src,
                part_id=0,
                images=None,
                input_ids=[1],
                labels=[1],
                grid_thw=np.array([[1, 2, 2]]),
                position_ids=np.array([[0, 0, 0]]),
                data_not_valid=1,  # indicate this is a invalid data
                data_type=DATATYPE_2_ID["mm"],
            )
            log.info(meta)
            log.info(f"****** Exception raised in dataset: {e} ******")
            log.exception(e)
            log.info("*********************************************************")
            return features

    def example_to_feature_stage3_video(self, example):
        """
        Deprecated: This function will be removed.
        """
        try:
            meta = json.loads(example.ids.tobytes().decode())
            raw_meta = deepcopy(meta)

            result = self.data_processor.process(raw_meta)
            assert len(result) == 1, f"result: {len(result)}, {result}"
            result = result[0]
            return OrderedDict(
                src_id=example.src,
                part_id=example.part,
                images=result["images"],
                input_ids=result["input_ids"],
                labels=result["labels"],
                token_type_ids=result["token_type_ids"],
                image_type_ids=result["image_type_ids"],
                data_not_valid=0,
                data_type=DATATYPE_2_ID["mm"],
            )
        except Exception as e:
            log.info(f"e: {e}")
            features = OrderedDict(
                src_id=example.src,
                part_id=example.part,
                images=None,
                input_ids=[1],
                labels=[1],
                data_not_valid=1,  # indicate this is a invalid data
                data_type=DATATYPE_2_ID["mm"],
            )
            log.info(meta)
            log.info(f"****** Exception raised in dataset: {e} ******")
            log.exception(e)
            log.info("*********************************************************")
            return features

    def __len__(self):
        return self.length

    def __iter__(self):
        while True:
            np.random.seed(make_seed(self.local_seed, self.epoch))
            sample_list = np.random.choice(self.src_id_list, size=5120, p=self.weight_list)
            self.epoch += 1
            for sample in sample_list:
                data = next(self.task_group[int(sample)])
                yield data
