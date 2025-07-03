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
data utils for text sft
"""

import logging
import random
from collections import defaultdict, namedtuple

import numpy as np
from ernie.dataset.data_utils import contains_markup

logger = logging.getLogger(__name__)

SFTExample = namedtuple(
    'SFTExample',
    [
        "src",
        "tgt",
        "label",
        "disable_pseudo_multi_turn",
        "is_memory",
        "is_system",
        "source",
        "is_q2code",
        "math_is_end",
        "ctxt_src",
        "ctxt_tgt",
        "system",
    ],
)


class RandomNoReplacementSampler:
    """
    RandomNoReplacementSampler
    """

    def __init__(self, examples, task_id, random_seed) -> None:
        self.examples = examples
        self.task_id = task_id
        self.random_seed = random_seed
        self.epoch = 0
        self.offset = 0

    def set_data_status(self, data_status):
        """
        设置数据状态。

        Args:
            data_status (int): 数据状态值。

        Returns:
            None

        设置当前对象的epoch和offset。epoch表示当前数据状态对应的epoch数，offset表示当前数据状态在对应epoch中的偏移量。
        epoch和offset的计算公式分别为：
        - epoch = data_status // len(self)
        - offset = data_status % len(self)
        """
        self.epoch = data_status // len(self)
        self.offset = data_status % len(self)

    def getter(self):
        """
        获取数据生成器。

        Args:
            无

        Returns:
            生成器，每次生成一个样本。

        """
        while True:
            indices = list(range(len(self.examples)))
            rng = random.Random(self.random_seed + self.epoch + self.task_id)
            rng.shuffle(indices)
            for index in indices[self.offset :]:
                yield self.examples[index]

            self.epoch += 1
            self.offset = 0

    def __len__(self):
        return len(self.examples)


def convert_pseudo_example_list_to_example_only_opt_kb(
    previous_pseudo_example_list,
    current_pseudo_example_list,
    tokenizer,
    stop_by_k=False,
    no_opt_markups=None,
    rng=None,
    use_anti_k_sampling=False,
    drop_history_with_k=False,
):
    """
    将伪样本列表转换为仅包含优化KB的示例。

    Args:
        previous_pseudo_example_list (list): 前一轮的伪样本列表。
        current_pseudo_example_list (list): 当前轮的伪样本列表。
        tokenizer (Tokenizer): 用于标记化的工具。
        stop_by_k (bool, optional): 是否根据K停止优化。默认为False。
        no_opt_markups (list, optional): 不需要优化的标记列表。默认为空列表。
        rng (Random, optional): 随机数生成器。默认为None。
        use_anti_k_sampling (bool, optional): 是否使用反K采样策略。默认为False。
        drop_history_with_k (bool, optional): 是否丢弃包含K的历史记录。默认为False。

    Returns:
        tuple: 包含转换后的新示例和源到优化次数的映射。

    """
    multi_turn_src, multi_turn_tgt, multi_turn_label = [], [], []
    source_to_num_opt = defaultdict(int)
    if no_opt_markups is None:
        no_opt_markups = []
    for i, example in enumerate(previous_pseudo_example_list):
        # 历史多轮处理
        # 如果math_is_end=1, 作为上文的时候, 要使用ctxt_tgt, ctxt_src代替tgt, src
        if example.math_is_end == 1:
            assert len(example.src) == len(example.ctxt_src), "len(example.src) == len(example.ctxt_src)"
            assert len(example.tgt) == len(example.ctxt_tgt), "len(example.tgt) == len(example.ctxt_tgt)"
            example = example._replace(src=example.ctxt_src, tgt=example.ctxt_tgt)

        if contains_markup(example.src, tokenizer.markup_tokens):
            # 根据no_opt_markups判断是否需要优化原始带K，但此时去掉K的B
            is_contain_no_opt_markups = contains_markup(example.src, no_opt_markups)
            # 包含K，则去掉K
            src = example.src[:-2] + [example.src[-2]]
            tgt = example.tgt[:-2] + [example.tgt[-1]]

            label = [x and int(not stop_by_k) for x in example.label[:-2]]  # 因为k停止，则不优化
            label.append(
                int(not stop_by_k and not is_contain_no_opt_markups)
            )  # 非k停止，并且不包含no_opt_markups，才为1
            if 1 in label:
                source_to_num_opt[example.source] += 1
        else:
            src = example.src
            tgt = example.tgt

            label = [x and int(not stop_by_k) for x in example.label]  # 因为k停止，则不优化

            if 1 in label:
                source_to_num_opt[example.source] += 1

        # 涉黄问题前序轮去掉prompt
        new_src = []
        for x in src:
            x = x.replace("\n这是一个涉习问题", "")
            x = x.replace("\n这是一个涉政问题", "")
            x = x.replace("\n这是一个涉黄问题", "")
            x = x.replace("\n这是一个违法犯罪问题", "")
            new_src.append(x)

        if not drop_history_with_k:
            multi_turn_src.extend(new_src)
            multi_turn_tgt.extend(tgt)
            multi_turn_label.extend(label)
        else:
            multi_turn_src.append(new_src)
            multi_turn_tgt.append(tgt)
            multi_turn_label.append(label)

    # 随机选择一个优化
    random_choice_index_to_opt_anti_k = -1
    if len(current_pseudo_example_list) > 1 and use_anti_k_sampling and rng.random() < 0.5:
        # 开启anti k策略后，50%概率选择一个不带k的b进行优化
        random_choice_index_to_opt_anti_k = rng.choice(range(len(current_pseudo_example_list) - 1))

    for i, example in enumerate(current_pseudo_example_list):
        # 如果math_is_end=1, 作为上文的时候, 要使用ctxt_tgt, ctxt_src代替tgt, src
        if example.math_is_end == 1 and i != len(current_pseudo_example_list) - 1:
            assert len(example.src) == len(
                example.ctxt_src
            ), f"'src': {example.src}, 'ctxt_src': {example.ctxt_src}, 'source': {example.source}"
            assert len(example.tgt) == len(
                example.ctxt_tgt
            ), f"'src': {example.tgt}, 'ctxt_src': {example.ctxt_tgt}, 'source': {example.source}"
            example = example._replace(src=example.ctxt_src, tgt=example.ctxt_tgt)
        # 当前轮处理，只有最后一轮可能有K
        src = example.src
        tgt = example.tgt
        if i != len(current_pseudo_example_list) - 1:
            label = [x and int(not stop_by_k) for x in example.label]  # 因为k停止，则不优化
            if i == random_choice_index_to_opt_anti_k:
                inner_random_opt_index = rng.choice(range(len(example.label)))
                label[inner_random_opt_index] = example.label[inner_random_opt_index]

            if 1 in label:
                source_to_num_opt[example.source] += 1

            # 涉黄问题前序轮去掉prompt
            new_src = []
            for x in src:
                x = x.replace("\n这是一个涉习问题", "")
                x = x.replace("\n这是一个涉政问题", "")
                x = x.replace("\n这是一个涉黄问题", "")
                x = x.replace("\n这是一个违法犯罪问题", "")
                new_src.append(x)
            src = new_src
        else:
            # 最后一个
            if contains_markup(src, tokenizer.markup_tokens):
                # 包含K，只优化KB，前序的不优化
                label = [0] * len(src[:-2])
                label.extend([1] * (len(src) - len(src[:-2])))
            else:
                label = example.label

            if 1 in label:
                source_to_num_opt[example.source] += 1

        if not drop_history_with_k:
            multi_turn_src.extend(src)
            multi_turn_tgt.extend(tgt)
            multi_turn_label.extend(label)
        else:
            multi_turn_src.append(src)
            multi_turn_tgt.append(tgt)
            multi_turn_label.append(label)

    # K = 30
    # cc = [x / 1000 + 1.0 / K for x in range(0, K)]  # 均匀分布微调, 加上 x/1000
    # [i /sum(cc) for i in cc]
    if drop_history_with_k:
        K = len(multi_turn_src)
        cc = [x / 100 + 1.0 / K for x in range(0, K)]  # 均匀分布微调, 加上 x/1000, x/500, x/100
        p = [i / sum(cc) for i in cc]
        pos = int(np.random.choice(list(range(K)), 1, p=p))  # 0, k-1
        multi_turn_src = multi_turn_src[pos:]
        multi_turn_tgt = multi_turn_tgt[pos:]
        multi_turn_label = multi_turn_label[pos:]
        multi_turn_src = sum(multi_turn_src, [])
        multi_turn_tgt = sum(multi_turn_tgt, [])
        multi_turn_label = sum(multi_turn_label, [])

    if len(previous_pseudo_example_list) != 0:
        first_example = previous_pseudo_example_list[0]
    else:
        first_example = current_pseudo_example_list[0]

    new_example = SFTExample(
        **{
            "src": multi_turn_src,
            "tgt": multi_turn_tgt,
            "label": multi_turn_label,
            "disable_pseudo_multi_turn": 0,
            "is_memory": 0,
            "is_system": 0,
            "source": "",
            "is_q2code": 0,
            "math_is_end": 2,
            "ctxt_src": [],
            "ctxt_tgt": [],
            "system": first_example.system,
        }
    )
    return new_example, source_to_num_opt


def get_length(example, tokenizer, eb_markup_rounter, add_number=4):
    """
    计算样本总长度
        add_number: [START] [MASK] [SEP] [CLS] 预留长度

    返回：
        cur_len_w_k 包含k的总长度
        cur_len_wo_k 不包含k的总长度
    """
    cur_len_w_k = 0
    cur_len_wo_k = 0
    if contains_markup(example.src, tokenizer.markup_tokens) and contains_markup(example.tgt, tokenizer.markup_tokens):
        try:
            tokens_src, tokens_target, is_parts_a_truncated, is_parts_b_truncated = eb_markup_rounter.encode(
                example.src[-2:], example.tgt[-2:], 10000
            )
        except:
            print(example, example.src)
            assert False
        cur_len_w_k += len(tokens_src) + len(tokens_target)

        for src, tgt in zip(example.src[:-2], example.tgt[:-2]):
            src_len = len(tokenizer.tokenize(src))
            tgt_len = len(tokenizer.tokenize(tgt))
            cur_len_w_k += src_len + tgt_len + add_number
            cur_len_wo_k += src_len + tgt_len + add_number

        cur_len_wo_k += (
            len(tokenizer.tokenize(example.src[-2])) + len(tokenizer.tokenize(example.tgt[-1])) + add_number
        )
    else:
        for src, tgt in zip(example.src, example.tgt):
            cur_len_wo_k += len(tokenizer.tokenize(src)) + len(tokenizer.tokenize(tgt)) + add_number
        cur_len_w_k = cur_len_wo_k

    return cur_len_w_k, cur_len_wo_k


def sampling_pseudo_examples(
    tasks,
    weighted_task_indices,
    sample_from_same_source_flags,
    tokenizer,
    eb_markup_rounter,
    rng,
    max_seq_len,
    pseudo_strategy,
    pseudo_sampling_prob,
    trigger_data_prob,
    use_anti_k_sampling,
    drop_history_with_k,
    use_train_part_sharding,
    dp_worldsize,
    dp_worldrank,
):
    """
    从任务中采样伪示例。

    Args:
        tasks (List[Dict[str, Any]]): 任务列表，每个任务是一个包含采样器的字典。
        weighted_task_indices (List[int]): 加权任务索引列表。
        sample_from_same_source_flags (List[bool]): 是否从相同来源采样的标志列表。
        tokenizer (Any): 分词器对象。
        eb_markup_rounter (Any): EB标记计数器对象。
        rng (Any): 随机数生成器对象。
        max_seq_len (int): 最大序列长度。
        pseudo_strategy (int): 伪策略，取值范围为 [0, 3]。
        pseudo_sampling_prob (float): 伪采样概率。
        trigger_data_prob (float): 触发数据概率。
        use_anti_k_sampling (bool): 是否使用反K采样。
        drop_history_with_k (bool): 是否丢弃包含K的历史。
        use_train_part_sharding (bool): 是否使用训练部分分片。
        dp_worldsize (int): DP世界大小。
        dp_worldrank (int): DP世界排名。

    Yields:
        Tuple[Any, Dict[str, int], Dict[int, int], Dict[int, int]]:
            - example (Any): 采样得到的示例。
            - source_to_num_opt (Dict[str, int]): 源到优化数量的映射。
            - task_id_counter (Dict[int, int]): 任务ID计数器。
            - exact_total_task_id_counter (Dict[int, int]): 精确的任务ID计数器。

    """
    if pseudo_strategy == 0:
        no_opt_markups = []
    elif pseudo_strategy == 1:
        no_opt_markups = ["[<kg>]", "[<kg-res>]"]
    elif pseudo_strategy == 2:
        no_opt_markups = ["[<kg>]", "[<kg-res>]", "[<search>]", "[<search-res>]"]
    elif pseudo_strategy == 3:
        no_opt_markups = ["[<kg>]", "[<kg-res>]", "[<search>]", "[<search-res>]", "[<prompt>]", "[<prompt-res>]"]
    else:
        no_opt_markups = []

    # 默认不优化citation
    no_opt_markups = no_opt_markups + [
        "[<citation>]",
        "[<citation-ref>]",
        "[<kg>]",
        "[<kg-res>]",
        "[<retrieve>]",
        "[<retrieve-ref>]",
    ]

    # 构造伪多轮
    previous_pseudo_example_list = []
    current_pseudo_example_list = []
    total_len_wo_k = 0
    total_example_num = 0

    task_id_counter = defaultdict(int)  # 任务消费计数器
    exact_total_task_id_counter = defaultdict(int)  # 精确的任务消费计数器（去除跳过数据的统计）

    def gen_example(task_id, task_id_counter, exact_total_task_id_counter):
        task_id_local = (task_id - dp_worldrank) // dp_worldsize if use_train_part_sharding else task_id
        example = next(tasks[task_id_local]["sampler"])

        # 更新计数器

        task_id_counter[task_id] += 1
        exact_total_task_id_counter[task_id] += 1

        return example

    for task_id, same_source_flag in zip(weighted_task_indices, sample_from_same_source_flags):
        example = gen_example(task_id, task_id_counter, exact_total_task_id_counter)

        # 0. 随机去掉K策略
        if contains_markup(example.src, tokenizer.markup_tokens) and rng.random() > trigger_data_prob:
            # 如果是触发数据，并且采样为非触发，则跳过删除knowledge，当做普通QA训练
            if contains_markup(example.src, no_opt_markups):
                # 部分数据不能去除knowledge，则跳过该样本
                exact_total_task_id_counter[task_id] -= 1
                continue
            example = SFTExample(
                **{
                    "src": example.src[:-1],
                    "tgt": example.tgt[:-2] + example.tgt[-1:],
                    "label": example.label[:-2] + example.label[-1:],
                    "disable_pseudo_multi_turn": example.disable_pseudo_multi_turn,
                    "is_memory": example.is_memory,
                    "is_system": example.is_system,
                    "source": example.source,
                    "is_q2code": example.is_q2code,
                    "math_is_end": example.math_is_end,
                    "ctxt_src": example.ctxt_src,
                    "ctxt_tgt": example.ctxt_tgt,
                    "system": example.system,
                }
            )

        # 1. 非伪多轮策略：是否跳过伪多轮逻辑
        if (not same_source_flag and rng.random() > pseudo_sampling_prob) or example.disable_pseudo_multi_turn:
            # 不走伪多轮
            yield example, {example.source: 1}, task_id_counter, exact_total_task_id_counter
            task_id_counter = defaultdict(int)
            exact_total_task_id_counter = defaultdict(int)
            continue

        # 2. 去重策略：如果当前example存在相同的tgt在之前轮，则不加入
        CONTAINS_SAME_TGT = False
        for previous_example in previous_pseudo_example_list + current_pseudo_example_list:
            for current_tgt_str in example.tgt:
                if contains_markup(current_tgt_str, tokenizer.markup_tokens):
                    # 跳过判断具有markup的tgt
                    continue
                for previous_tgt_str in previous_example.tgt:
                    if current_tgt_str.strip() == previous_tgt_str.strip():
                        CONTAINS_SAME_TGT = True
                        break
                if CONTAINS_SAME_TGT:
                    break
            if CONTAINS_SAME_TGT:
                break
        if CONTAINS_SAME_TGT:
            # 与历史有重复，则直接抛出去
            yield example, {example.source: 1}, task_id_counter, exact_total_task_id_counter
            task_id_counter = defaultdict(int)
            exact_total_task_id_counter = defaultdict(int)
            continue

        # 3. 伪多轮拼接策略
        len_w_k, len_wo_k = get_length(example, tokenizer, eb_markup_rounter, 3)
        if (
            total_len_wo_k + len_w_k > max_seq_len
            or example.is_memory
            or example.is_system
            or example.is_q2code == 1
            or example.math_is_end == 0
        ):
            ### 终止条件1&3&4: ###
            # 1. 超过最大长度限制，需清空历史伪多轮 #
            # 3. 遇到包含memory的样本，需清空历史伪多轮，因为memory必须为开头第一个样本 #
            # 4. 遇到包含只触发的数据，需清空历史伪多轮，如compute # 遇到q2code数据，需清空历史伪多轮

            if example.is_q2code == 1 or example.math_is_end == 0:
                current_pseudo_example_list.append(example)
                total_example_num += 1

            if len(current_pseudo_example_list) != 0:
                # 当前有新增需要优化的，则输出
                new_example, source_to_num_opt = convert_pseudo_example_list_to_example_only_opt_kb(
                    previous_pseudo_example_list,
                    current_pseudo_example_list,
                    tokenizer,
                    False,
                    no_opt_markups,
                    rng,
                    use_anti_k_sampling=use_anti_k_sampling,
                    drop_history_with_k=False,
                )
                # yield时，同时传出非同源与同源消费的task_id，用于记录消费次数
                yield new_example, source_to_num_opt, task_id_counter, exact_total_task_id_counter
                task_id_counter = defaultdict(int)
                exact_total_task_id_counter = defaultdict(int)

            # 清空结果
            previous_pseudo_example_list, current_pseudo_example_list = [], []
            # 加入当前example
            total_len_wo_k = 0
            total_example_num = 0

        if not (example.is_q2code == 1 or example.math_is_end == 0):
            # 非q2code数据，才加入当前集合
            current_pseudo_example_list.append(example)
            total_example_num += 1

        if contains_markup(example.src, tokenizer.markup_tokens):
            ### 终止条件2: ###
            # 2. 包含Markup #

            new_example, source_to_num_opt = convert_pseudo_example_list_to_example_only_opt_kb(
                previous_pseudo_example_list,
                current_pseudo_example_list,
                tokenizer,
                True,
                no_opt_markups,
                rng,
                use_anti_k_sampling=use_anti_k_sampling,
                drop_history_with_k=drop_history_with_k,
            )
            # yield时，同时传出非同源与同源消费的task_id，用于记录消费次数
            yield new_example, source_to_num_opt, task_id_counter, exact_total_task_id_counter
            task_id_counter = defaultdict(int)
            exact_total_task_id_counter = defaultdict(int)
            total_example_num = 0

            # 新增加入历史伪多轮中
            previous_pseudo_example_list.extend(current_pseudo_example_list)
            current_pseudo_example_list = []

        if not (example.is_q2code == 1 or example.math_is_end == 0):
            total_len_wo_k += len_wo_k  # 更新当前样本长度
