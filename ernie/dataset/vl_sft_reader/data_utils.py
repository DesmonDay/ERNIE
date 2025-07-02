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
data utils
"""
import copy
import logging
import math
import re

import numpy as np
import paddle
from PIL import Image

# from paimon import PaimonBosClient

# bos = PaimonBosClient()
logger = logging.getLogger(__name__)

PATTERN = r"<ref>(.*?)</ref><quad>(.*?)</quad>"
PATTERN_QUAD = r"\((\d+),(\d+)\),\((\d+),(\d+)\),\((\d+),(\d+)\),\((\d+),(\d+)\)"
DEBUG_PRINT_CNT = 0


class InterleaveValueError(Exception):
    """
    Interleave Value Error
    """

    def __init__(self, message):
        super().__init__(message)


def pad_sequence(sequences, padding_value=0, fix_len=None):
    """Fill sequences(np.ndarray) into a fixed-length matrix."""
    # don't use any paddle.Tensor in collate-fn
    #   which prevent leakage in multi-process
    max_size = sequences[0].shape
    trailing_dims = tuple(max_size[1:])
    # print("trailing_dims: ", trailing_dims)

    max_len = max([s.shape[0] for s in sequences])
    if fix_len is not None:
        if fix_len < max_len:
            logger.warning(f"truncating example from {max_len} to {fix_len}")
        max_len = fix_len
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = np.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        tensor = tensor[:max_len]
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor


DEBUG_PRINT_CNT = 0


def cumulative_indices_merging(x):
    """
    合并并生成累积索引列表。

    Args:
        x (list of lists of int): 输入的索引列表，其中每个子列表表示一组索引。

    Returns:
        list of int: 合并后的累积索引列表。

    Example:
        >>> cumulative_indices_merging([[1, 2], [3, 4], [5]])
        [1, 3, 6, 10]

    """
    cur_cumulative_indices = []
    tmp = 0
    for i, indices in enumerate(x):
        if i != 0:
            indices = indices[1:]
        tmp = cur_cumulative_indices[-1] if len(cur_cumulative_indices) != 0 else tmp
        for index in indices:
            cur_cumulative_indices.append(tmp + index)
    return cur_cumulative_indices


class Bcolors:
    """
    定义一些颜色常量以便于在终端中使用不同颜色的文本输出。
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def fancy_print(data, tokenizer, im_prefix_length):
    """
    对给定的数据进行格式化输出。

    Args:
        data (dict): 包含输入和标签的数据字典。
        tokenizer (Tokenizer): 用于文本编码的分词器。
        im_prefix_length (int): 图像前缀的长度，该参数未在此函数中使用。

    Returns:
        str: 格式化后的输出字符串。

    """
    marker1 = "[unused99]"
    marker2 = "[unused98]"
    image_token = len(tokenizer.get_vocab())

    for ids, labels in zip(data["input_ids"].tolist(), data["labels"].tolist()):
        # log.info(labels)
        ids2 = []
        assert len(ids) == len(labels)
        last_j = 0
        for i, j in zip(ids, labels):
            j = int(j != tokenizer.ignored_index)
            if i == image_token:
                ids2 += tokenizer.encode("<|image|>", return_attention_mask=False)["input_ids"]
            else:
                ids2.append(i)
            if j != last_j:
                ids2 += tokenizer.encode(
                    marker1 if (j > last_j) else marker2, add_special_tokens=False, return_attention_mask=False
                )["input_ids"]
            last_j = j
        if j == 1:
            ids2 += tokenizer.encode(marker2, add_special_tokens=False, return_attention_mask=False)
        ret = tokenizer.decode(ids2).replace("[unused99]", Bcolors.FAIL).replace("[unused98]", Bcolors.ENDC)

        # 花里胡哨的代码
        image_tag = "<|image|>"
        pat = re.compile(f"({re.escape(image_tag)})+")
        build = []
        for i in pat.finditer(ret):
            cnt = i.group(0).count(image_tag)
            build.append((i.span(), f"<|image@{cnt}|>"))

        pad_tag = ["<pad>", "<mask:0>"]
        pat = re.compile(f"({'|'.join(pad_tag)})+")
        for i in pat.finditer(ret):
            cnt = sum(i.group(0).count(t) for t in pad_tag)
            build.append((i.span(), f"<pad@{cnt}>"))

        for s, t in build[::-1]:
            l, r = s
            ret = ret[:l] + t + ret[r:]
        return ret


def merge_fn(tokenizer, batch, pad_to_max_seqlen=None, debug_print=1, im_prefix_length=32, shift_label=False):
    """
    collate_fn
    **seqlen超长时会从右边开始截断，seqlen不够时会从右边开始pad**
    """
    global DEBUG_PRINT_CNT
    if pad_to_max_seqlen and shift_label:
        pad_to_max_seqlen += 1

    keys = list(batch[0].keys())

    ret = {}
    for k in keys:
        if isinstance(batch[0][k], (int, float)):
            ret[k] = np.stack([b[k] for b in batch], 0)
        elif k == "images":
            ret[k] = paddle.concat([b[k] for b in batch])
        else:
            if k == "input_ids":
                pad_value = tokenizer.pad_token_id
            elif k == "labels":
                pad_value = tokenizer.ignored_index
            else:
                pad_value = 0

            if batch[0][k] is not None:
                ret[k] = pad_sequence([b[k] for b in batch], padding_value=pad_value, fix_len=pad_to_max_seqlen)

    batch = ret
    # batch.update(ignored_index=tokenizer.ignored_index)

    if DEBUG_PRINT_CNT < debug_print:
        DEBUG_PRINT_CNT += 1
        for k, v in batch.items():
            logger.debug(
                f"""Example={DEBUG_PRINT_CNT} key={k},
                len={len(v[0])if isinstance(v, np.ndarray) and v.ndim > 1 else 0},
                value={v[0] if isinstance(v, np.ndarray) else v}"""
            )
        logger.debug(f"Example={DEBUG_PRINT_CNT} text={fancy_print(batch, tokenizer, im_prefix_length)}")

    if shift_label:
        batch["labels"] = batch["labels"][:, 1:]
        batch["input_ids"] = batch["input_ids"][:, :-1]
    return batch


def merge_rope_3d_position(list_position_ids):
    """merge two 3d position"""

    def merge(position_ids, new_position_ids):
        position_ids = np.array(position_ids)
        new_position_ids = np.array(new_position_ids)
        return np.concatenate((position_ids, np.max(position_ids) + new_position_ids), axis=0)

    returned = list_position_ids[0]
    for position_ids in list_position_ids[1:]:
        returned = merge(returned, position_ids)
    return returned


def unmerge_rope_3d_position(position_ids, input_ids_batch):
    """unmerge 3d position according to input_ids_batch"""
    accu_id_length = 0
    position_ids = copy.deepcopy(position_ids)
    for idx, item in enumerate(input_ids_batch):
        if idx == 0:
            assert position_ids[accu_id_length][0] == 0, "the fisrt postition-ids should be 0"
        position_ids[accu_id_length : accu_id_length + len(item)] -= position_ids[accu_id_length][0]

        accu_id_length += len(item)
    return position_ids


def merge_fn_group_batch(
    batch,
    tokenizer=None,
    pad_to_max_seqlen=None,
    debug_print=1,
    im_prefix_length=32,
    shift_label=False,
    rng=None,
    combine_batch: int = 1,
    image_dtype="uint8",
    multimodal_multiround_ratio=0.3,
):
    """
    batch 内 n合一
    """
    need_multiround = batch[0].get("need_multiround", rng.random() < multimodal_multiround_ratio)
    if "need_multiround" in batch[0]:
        del batch[0]["need_multiround"]

    global DEBUG_PRINT_CNT
    if pad_to_max_seqlen and shift_label:
        pad_to_max_seqlen += 1

    assert len(batch) == 1, f"zzz len(batch) != 1, {len(batch)}"
    keys = list(batch[0].keys())
    if combine_batch > 1:
        _batch = []
        for group in [batch[i : i + combine_batch] for i in range(0, len(batch), combine_batch)]:
            item = {}
            for k in keys:
                if isinstance(group[0][k], (int, float)):
                    item[k] = np.stack([i[k] for i in group], 0)

                else:
                    item[k] = np.concatenate([i[k] for i in group])
            _batch.append(item)
        batch = _batch

    ret = {}
    for k in keys:
        if k == 'input_ids_batch':
            continue

        if isinstance(batch[0][k], (int, float, np.int64, np.float64)):
            ret[k] = np.stack([b[k] for b in batch], 0)
        elif k == 'grid_thw':
            to_concat = [b[k] for b in batch if b[k] is not None]
            ret[k] = np.concatenate(to_concat, axis=0)
            if pad_to_max_seqlen:
                tmp = max(0, pad_to_max_seqlen * len(batch) - ret[k].shape[0])
                if tmp > 0:
                    ret[k] = np.concatenate([ret[k], np.zeros([tmp, 3])], axis=0).astype("int64")
        elif k in ["data_id", "part_id", "src_id"]:
            ret[k] = np.concatenate([b[k] for b in batch])
        elif k == "data_type":
            ret[k] = np.array(batch[0][k])
        elif k == "images":
            to_concat = [b[k] for b in batch if b[k] is not None]
            if len(to_concat) != 0:
                assert image_dtype != "bfloat16", f"Currently, not support {image_dtype} for numpy"
                # ret[k] = np.concatenate([b[k] for b in batch if b[k] is not None])
                ret[k] = np.concatenate(to_concat, axis=0).astype(image_dtype)
            else:
                ret[k] = None
        elif k == "cumulative_indices":
            if batch[0][k] is not None:
                cur_cumulative_indices = cumulative_indices_merging([b[k] for b in batch])
                cur_cumulative_indices = np.array(cur_cumulative_indices)
                ret[k] = cur_cumulative_indices
            else:
                ret[k] = None
        else:
            if k == "input_ids":
                pad_value = tokenizer.pad_token_id
            elif k in ["labels", "image_type_ids"]:
                pad_value = tokenizer.ignored_index
            elif k == 'image_position_ids':
                pad_value = -1
            elif k in ['position_ids']:
                pad_value = [0, 0, 0]
            elif k in ["token_type_ids"]:
                pad_value = 0
            else:
                pad_value = 0

            if batch[0][k] is not None:
                if k in ['tmp_images', 'image_position_ids']:
                    tmp = [i for b in batch for i in b[k]]
                else:
                    tmp = [b[k] for b in batch]
                try:
                    if k == "token_type_ids":
                        ret[k] = pad_sequence(tmp, padding_value=pad_value, fix_len=pad_to_max_seqlen + 1)
                    else:
                        ret[k] = pad_sequence(tmp, padding_value=pad_value, fix_len=pad_to_max_seqlen)
                except Exception as e:
                    logger.info(f"k: {k}, tmp: {tmp}, original: {[b[k] for b in batch]}")
                    logger.info(f"e: {e}")
                    # exit()

                if k == 'image_position_ids':
                    ret['image_attention_mask'] = ret[k] != pad_value
                    ret[k][ret[k] == pad_value] = 0

    inbatch_pack_offset = [0]
    if not need_multiround:  # 0.3是伪多轮的概率，TODO：改成超参设置
        for item in batch[0]['input_ids_batch']:
            inbatch_pack_offset.append(inbatch_pack_offset[-1] + len(item))
        # fix position-ids
        if "position_ids" in ret:
            ret["position_ids"] = [unmerge_rope_3d_position(ret["position_ids"][0], batch[0]['input_ids_batch'])]

            # logger.debug(f"unmerge position_ids: ret['position_ids']")
        inbatch_pack_offset[-1] = pad_to_max_seqlen  # include padding in the last interval
    else:
        inbatch_pack_offset.append(pad_to_max_seqlen)
    padded_inbatch_pack_offset = np.reshape(
        np.array(inbatch_pack_offset + [-1] * (pad_to_max_seqlen + 1 - len(inbatch_pack_offset)), dtype=np.int64),
        [1, -1],
    )
    ret["inbatch_pack_offset"] = padded_inbatch_pack_offset
    batch = ret

    if DEBUG_PRINT_CNT < debug_print:
        DEBUG_PRINT_CNT += 1
        for k, v in batch.items():
            logger.debug(
                f"""Example={DEBUG_PRINT_CNT} key={k},
                len={len(v[0])if isinstance(v, np.ndarray) and v.ndim > 1 else 0},
                value={v[0] if isinstance(v, np.ndarray) else v}"""
            )
        logger.debug(f"Example={DEBUG_PRINT_CNT} text={fancy_print(batch, tokenizer, im_prefix_length)}")
    # for key, value in batch.items():
    #     logger.info(f"shape-of-{key}: {np.array(value).shape}")
    #     if key in ["grid_thw", "position_ids"]:
    #         logger.info(f"{key}: {np.array(value).tolist()}")
    # logger.info(f"{key}: {np.array(value).tolist()}")
    if shift_label:
        batch["labels"] = batch["labels"][:, 1:]
        batch["input_ids"] = batch["input_ids"][:, :-1]
    return batch


def get_in_batch_mask(batch):
    """
    获取批量数据的掩码。

    Args:
        batch (list): 包含多个序列的列表，每个序列由相同长度的元素组成。

    Returns:
        numpy.ndarray: 掩码矩阵，形状为 (1, len(batch), max_len, max_len)，其中 max_len 是 batch 中序列的最大长度。

    """
    length_list = [len(b) for b in batch]
    scale = np.concatenate([np.full(length_list[i], i, dtype=batch[0].dtype) for i in range(len(batch))], 0)
    mask = np.expand_dims(np.tril(np.expand_dims(scale, axis=0) == np.expand_dims(scale, axis=-1)), axis=0).astype(
        batch[0].dtype
    )
    return mask


def merge_fn_in_batch(
    tokenizer,
    batch,
    pad_to_max_seqlen=None,
    debug_print=1,
    im_prefix_length=32,
    use_attn_mask=False,
    shift_label=False,
):
    """
    in-batch merge策略，将数据横向连接。目前没有处理attention-mask.
    `pad_to_max_seqlen`指定后， **seqlen超长时会从右边开始阶段，seqlen不够时会从右边开始pad**
        不指定的话，没有padding。
    """
    global DEBUG_PRINT_CNT
    if pad_to_max_seqlen and shift_label:
        pad_to_max_seqlen += 1
    keys = list(batch[0].keys())
    if "data_id" in keys:
        data_id = np.stack([b.pop("data_id") for b in batch], 0)

    ret = {}
    for k in keys:
        if k == "data_id":
            continue
        if k == "input_ids":
            pad_value = tokenizer.pad_token_id
        elif k == "labels":
            pad_value = tokenizer.ignored_index
        else:
            pad_value = 0

        if batch[0][k] is not None:
            array_list = [b[k] for b in batch]
            for a in array_list:
                if k != "images":
                    assert len(a.shape) == 1, (a.shape, k)
                else:
                    assert len(a.shape) == 4, (a.shape, k)
            concated = np.concatenate(array_list, 0)
            if pad_to_max_seqlen and k != "images":
                concated = concated[:pad_to_max_seqlen]
                pad_len = pad_to_max_seqlen - concated.shape[0]
                if pad_len > 0:
                    concated = np.pad(concated, (0, pad_len), constant_values=pad_value)

            if k != "images":
                concated = np.expand_dims(concated, 0)
            ret[k] = concated

            if use_attn_mask and k == "input_ids":
                attn_mask = get_in_batch_mask(array_list)
                if pad_to_max_seqlen:
                    attn_mask = attn_mask[:pad_to_max_seqlen]
                    pad_len = pad_to_max_seqlen - attn_mask.shape[1]
                    if pad_len > 0:
                        attn_mask = np.pad(attn_mask, ((0, 0), (0, pad_len), (0, pad_len)), constant_values=0)
                ret["attention_mask"] = attn_mask

    batch = ret
    batch.update(ignored_index=tokenizer.ignored_index)
    if "data_id" in keys:
        batch.update(data_id=data_id)

    if DEBUG_PRINT_CNT < debug_print:
        DEBUG_PRINT_CNT += 1
        for k, v in batch.items():
            logger.debug(
                f"""Example={DEBUG_PRINT_CNT} key={k},
                len={len(v[0])if isinstance(v, np.ndarray) else 0},
                value={v[0] if isinstance(v, np.ndarray) else v}"""
            )
        logger.debug(f"Example={DEBUG_PRINT_CNT} text={fancy_print(batch, tokenizer, im_prefix_length)}")

    if shift_label:
        batch["labels"] = batch["labels"][:, 1:]
        batch["input_ids"] = batch["input_ids"][:, :-1]
    return batch


def mix_merge_fn_in_batch(
    tokenizer,
    multi_source,
    pad_to_max_seqlen=None,
    multimodal_max_seqlen=None,
    use_attn_mask=False,
    debug_print=4,
    im_prefix_length=32,
):
    """
    collate_fn
    """
    global DEBUG_PRINT_CNT

    new_batch = {}
    for source in multi_source:
        keys = list(source[0].keys())
        if "data_id" in keys:
            data_id = np.stack([b.pop("data_id") for b in source], 0)

        if "text_input_ids" in source[0]:
            max_seqlen = pad_to_max_seqlen
        else:
            max_seqlen = multimodal_max_seqlen

        ret = {}
        for k in keys:
            if "data_id" in k:
                continue
            if "input_ids" in k:
                pad_value = tokenizer.pad_token_id
            elif "labels" in k:
                pad_value = tokenizer.ignored_index
            else:
                pad_value = 0

            if source[0][k] is not None:
                if "text_" in k:
                    if max_seqlen is None or k == "text_attention_mask":
                        ret[k] = np.array([b[k] for b in source])
                    else:
                        ret[k] = pad_sequence([b[k] for b in source], padding_value=pad_value, fix_len=max_seqlen)
                else:
                    array_list = [b[k] for b in source]
                    for cur_array in array_list:
                        if k != "images":
                            assert len(cur_array.shape) == 1, (cur_array.shape, k)
                        else:
                            assert len(cur_array.shape) == 4, (cur_array.shape, k)
                    concated = np.concatenate(array_list, 0)
                    if max_seqlen is not None and k != "images":
                        concated = concated[:max_seqlen]
                        pad_len = max_seqlen - concated.shape[0]
                        if pad_len > 0:
                            concated = np.pad(concated, (0, pad_len), constant_values=pad_value)

                    if k != "images":
                        concated = np.expand_dims(concated, 0)
                    ret[k] = concated

                    if use_attn_mask and k == "input_ids":
                        attn_mask = get_in_batch_mask(array_list)
                        if max_seqlen:
                            attn_mask = attn_mask[:max_seqlen]
                            pad_len = max_seqlen - attn_mask.shape[1]
                            if pad_len > 0:
                                attn_mask = np.pad(attn_mask, ((0, 0), (0, pad_len), (0, pad_len)), constant_values=0)
                        ret["attention_mask"] = attn_mask

        new_batch.update(ret)
    new_batch.update(ignored_index=tokenizer.ignored_index)

    if DEBUG_PRINT_CNT < debug_print:
        DEBUG_PRINT_CNT += 1
        for k, v in new_batch.items():
            logger.debug(
                f"""Example={DEBUG_PRINT_CNT} key={k},
                len={len(v[0])if isinstance(v, np.ndarray) else 0},
                value={v[0] if isinstance(v, np.ndarray) else v}"""
            )
        text_debug = {"input_ids": new_batch["text_input_ids"], "labels": new_batch["text_labels"]}
        logger.debug(f"TextExample={DEBUG_PRINT_CNT} text={fancy_print(text_debug, tokenizer, im_prefix_length)}")
        multimodal_debug = {"input_ids": new_batch["input_ids"], "labels": new_batch["labels"]}
        logger.debug(
            f"MultimodalExample={DEBUG_PRINT_CNT} text={fancy_print(multimodal_debug, tokenizer, im_prefix_length)}"
        )

    return new_batch


def normalize_quad_old(quad_str, width, height):
    """normalize_quad"""
    matches = re.findall(PATTERN_QUAD, quad_str)
    if matches is None:
        return None
    assert len(matches) == 1, f"quad may be wrong {quad_str}"

    macth = matches[0]
    x_list, y_list = macth[0::2], macth[1::2]
    x_list = list(map(lambda x: int(int(x) / width * 1000), x_list))
    y_list = list(map(lambda x: int(int(x) / height * 1000), y_list))

    assert len(x_list) == len(y_list), f"quad points may be wrong {quad_str}"
    quad_str_normalized = ""
    for x, y in zip(x_list, y_list):
        quad_str_normalized += f"({x},{y}),"
    quad_str_normalized = quad_str_normalized.rstrip(",")
    return quad_str_normalized


def normalize_ocr_old(text, width, height):
    """normalize_ocr"""
    matches = re.findall(PATTERN, text)

    if matches is None:
        return None

    text_noramlize = ""
    for phrase, quad_str in matches:
        text_noramlize += f"<ref>{phrase}</ref>"
        text_quad = normalize_quad(quad_str, width, height)
        if text_quad is None:
            return None
        text_noramlize += f"<quad>{text_quad}</quad>"
    return text_noramlize


def normalize_quad(quad_str, width, height):
    """normalize_quad"""
    matches = re.findall(PATTERN_QUAD, quad_str)
    if matches is None:
        return None
    assert len(matches) == 1, f"quad may be wrong {quad_str}"

    macth = matches[0]
    x_list, y_list = macth[0::2], macth[1::2]
    x_list = list(map(lambda x: int(int(x) / width * 1000), x_list))
    y_list = list(map(lambda x: int(int(x) / height * 1000), y_list))

    assert len(x_list) == len(y_list), f"quad points may be wrong {quad_str}"
    quad_str_normalized = ""
    for x, y in zip(x_list[0::2], y_list[0::2]):
        quad_str_normalized += f"({x},{y}),"
    quad_str_normalized = quad_str_normalized.rstrip(",")
    return quad_str_normalized


def normalize_ocr(text, width, height):
    """normalize_ocr"""
    matches = re.findall(PATTERN, text)

    if matches is None:
        return None

    text_noramlize = ""
    for phrase, quad_str in matches:
        text_noramlize += f"<ref>{phrase}</ref>"
        text_quad = normalize_quad(quad_str, width, height)
        if text_quad is None:
            return None
        text_noramlize += f"<box>{text_quad}</box>"
    return text_noramlize


def find_all_indexes(lst, value):
    """find_all_indexes"""
    return [index for index, item in enumerate(lst) if item == value]


def ref_mask(lst, ref_index, _ref_index, ignore_indx=-100):
    """ref_mask"""
    ref_index_lst = find_all_indexes(lst, ref_index)
    _ref_index_lst = find_all_indexes(lst, _ref_index)
    for index in ref_index_lst + _ref_index_lst:
        lst[index] = ignore_indx
    return lst


def box_mask(lst, box_index, _box_index, ignore_index_lst, ignore_index=-100):
    """
    根据给定的索引对列表中的元素进行屏蔽处理。

    Args:
        lst (list): 待处理的列表。
        box_index (int): 起始索引。
        _box_index (int): 结束索引。
        ignore_index_lst (list): 需要被屏蔽的元素列表。
        ignore_index (int, optional): 屏蔽后元素的替换值，默认为-100。

    Returns:
        list: 处理后的列表。

    Raises:
        AssertionError: 如果起始索引大于结束索引，则抛出断言错误。
    """
    box_index_lst = find_all_indexes(lst, box_index)
    _box_index_lst = find_all_indexes(lst, _box_index)
    if len(box_index_lst) != len(_box_index_lst):
        print("dataset uncomplete")
        return lst
    # import pdb;pdb.set_trace()
    for box_index, _box_index in zip(box_index_lst, _box_index_lst):
        assert box_index < _box_index, "box list index out of bounds"
        for index in range(box_index, _box_index + 1):
            if lst[index] in ignore_index_lst:
                lst[index] = ignore_index
    return lst


def loss_mask_index_replace_for_grounding(
    lst, ref_index, _ref_index, box_index, _box_index, ignore_index_lst, ignore_index=-100
):
    """loss_mask_index_replace"""

    lst = ref_mask(lst, ref_index, _ref_index, ignore_indx=ignore_index)

    lst = box_mask(lst, box_index, _box_index, ignore_index_lst, ignore_index=ignore_index)
    return lst


PATTERN_DETECT = r"<ref>(.*?)</ref>((?:<box>.*?</box>)+)(?=<ref>|$)"
PATTERN_DETECT_BOX = r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"


def normalize_box(box_str, width, height):
    """normalize_box"""
    matches = re.findall(PATTERN_DETECT_BOX, box_str)

    if matches is None:
        return None

    text = ""
    for box in matches:
        x1, y1, x2, y2 = box
        x1 = int(int(x1) / width * 1000)
        x2 = int(int(x2) / width * 1000)
        y1 = int(int(y1) / height * 1000)
        y2 = int(int(y2) / height * 1000)
        text += f"<box>({x1},{y1}),({x2},{y2})</box>"
    return text


def normalize_detect(text, width, height):
    """normalize_ocr"""
    split_token = "> and <"
    text_list = text.split(split_token)
    text = ">#and#<".join(text_list)
    text_list = text.split("#and#")

    text_noramlize = []
    for text_ in text_list:
        text_new = ""
        matches = re.findall(PATTERN_DETECT, text_)
        if matches is None:
            return None
        phrase, box_str = matches[0]

        text_new += f"<ref>{phrase}</ref>"
        box_str_new = normalize_box(box_str, width, height)
        if box_str_new is None:
            return None
        text_new += box_str_new
        text_noramlize.append(text_new)

    return " and ".join(text_noramlize)


def pad_images_to_square(images, pad_value=127):
    """
    将图像填充为正方形。

    Args:
        images (list of PIL.Image.Image): 要填充的图像列表。
        pad_value (int, optional): 填充使用的颜色值，默认为127。

    Returns:
        list of PIL.Image.Image: 填充后的图像列表。

    """
    new_images = []
    for image in images:
        width, height = image.size
        new_size = max(width, height)

        new_image = Image.new("RGB", (new_size, new_size), color=pad_value)

        h_padding = (new_size - width) // 2
        v_padding = (new_size - height) // 2

        new_image.paste(image, (h_padding, v_padding))
        new_images.append(new_image)
    return new_images


def normalize_box_fn(box_match, width, height):
    """normalize_box"""

    def process_coordinates(coord_match):
        x1, y1, x2, y2 = map(int, coord_match.groups())
        x1 = int(int(x1) / width * 1000)
        x2 = int(int(x2) / width * 1000)
        y1 = int(int(y1) / height * 1000)
        y2 = int(int(y2) / height * 1000)
        return f"({x1},{y1}),({x2},{y2})"

    box_content = box_match.group(1)
    processed_content = re.sub(r"\((\d+),(\d+)\),\((\d+),(\d+)\)", process_coordinates, box_content)

    return f"<box>{processed_content}</box>"


def normalize_box_for_str(text, width, height):
    """normalize_box_for_str"""
    process_text = re.sub(r"<box>(.*?)</box>", lambda match: normalize_box_fn(match, width, height), text)
    return process_text


def normalize_quad_fn(box_match, width, height):
    """normalize_box"""

    def process_coordinates(coord_match):
        x1, y1, x2, y2, x3, y3, x4, y4 = map(int, coord_match.groups())
        x1 = int(int(x1) / width * 1000)
        x2 = int(int(x2) / width * 1000)
        x3 = int(int(x3) / width * 1000)
        x4 = int(int(x4) / width * 1000)
        y1 = int(int(y1) / height * 1000)
        y2 = int(int(y2) / height * 1000)
        y3 = int(int(y3) / height * 1000)
        y4 = int(int(y4) / height * 1000)
        return f"({x1},{y1}),({x2},{y2}),({x3},{y3}),({x4},{y4})"

    box_content = box_match.group(1)
    processed_content = re.sub(
        r"\((\d+),(\d+)\),\((\d+),(\d+)\),\((\d+),(\d+)\),\((\d+),(\d+)\)", process_coordinates, box_content
    )

    return f"<quad>{processed_content}</quad>"


def normalize_quad_for_str(text, width, height):
    """normalize_quad_for_str"""
    process_text = re.sub(r"<quad>(.*?)</quad>", lambda match: normalize_quad_fn(match, width, height), text)
    return process_text


def convert_quad_to_box_fn(text):
    """convert_quad_to_box"""
    # 正则表达式匹配<quad>标签及其内容
    quad_pattern = r"<quad>\((\d+,\d+)\),\((\d+,\d+)\),\((\d+,\d+)\),\((\d+,\d+)\)</quad>"
    # 定义一个处理函数

    def process_quad(match):
        # 获取第一个和第三个坐标
        coord1, _, coord3, _ = match.groups()
        # 返回替换后的<box>标签
        return f"<box>({coord1}),({coord3})</box>"

    # 使用正则表达式替换<quad>为<box>
    return re.sub(quad_pattern, process_quad, text)


def convert_quad_to_box(data):
    """convert_quad_to_box"""
    for text_info in data["text_info"]:
        text = text_info["text"]
        if "<quad>" in text and "</quad>" in text:
            text = convert_quad_to_box_fn(text)
            # logger.info("quad to box:\nbefore:\n{}\nafter:\n{}".format(text_info["text"], text))
        text_info["text"] = text

    return data


LOSS_MASK_TEXT_DICT = dict(
    ref_token="<ref>",
    _ref_token="</ref>",
    box_token="<box>",
    _box_token="</box>",
    quad_token="<quad>",
    _quad_token="</quad>",
    ignore_tokens=["(", ")", ","],
)


def find_paired_tag_individual_indices(text, mask_flag, specal_token_start, special_token_end, ignore_tokens=None):
    """
    在给定文本中查找配对的HTML标签，并返回标签的起始和结束索引。

    Args:
        text (str): 要查找的文本。
        mask_flag (list): 一个布尔值列表，表示文本中每个字符是否被掩码。
        specal_token_start (str): 起始标签的标记，例如 "<tag>"。
        special_token_end (str): 结束标签的标记，例如 "</tag>"。
        ignore_tokens (list, optional): 要忽略的字符列表。默认为None。

    Returns:
        list: 更新后的mask_flag列表，其中标签及其内容被标记为True。

    """
    # 正则表达式匹配配对的 <tag>...</tag>
    pattern = f"{specal_token_start}.*?{special_token_end}"
    matches = re.finditer(pattern, text, re.DOTALL)

    # 获取每个配对的 <tag> 和 </tag> 的起始和结束索引
    paired_indices = []
    for match in matches:
        start_index = match.start()
        end_index = match.end()

        # 查找内部的 <tag> 和 </tag>
        internal_match = re.search(f"{specal_token_start}|{special_token_end}", match.group())
        if internal_match:
            tag_start_index = start_index + internal_match.start()
            tag_end_index = start_index + internal_match.end()

            # 计算 </tag> 的位置
            tag_close_start_index = text.find(special_token_end, tag_end_index)
            tag_close_end_index = tag_close_start_index + len(special_token_end)

            paired_indices.append(((tag_start_index, tag_end_index), (tag_close_start_index, tag_close_end_index)))

    ignore_token_index = []
    if ignore_tokens is not None:
        for (tag_start_index, tag_end_index), (tag_close_start_index, tag_close_end_index) in paired_indices:
            for i in range(tag_end_index, tag_close_start_index + 1):
                if text[i] in ignore_tokens:
                    ignore_token_index.append(i)

    for (tag_start_index, tag_end_index), (tag_close_start_index, tag_close_end_index) in paired_indices:
        mask_flag[tag_start_index:tag_end_index] = [True] * (tag_end_index - tag_start_index)
        mask_flag[tag_close_start_index:tag_close_end_index] = [True] * (tag_close_end_index - tag_close_start_index)

    for i in ignore_token_index:
        mask_flag[i] = True

    return mask_flag


def loss_mask_with_origin_word_fn(lst, text, text_mask, tokenizer, ignore_index):
    """loss_mask_with_origin_word_fn"""
    # import pdb; pdb.set_trace()
    spm_token = tokenizer.sp_model.encode_as_immutable_proto(text)
    mask_list = []
    for p, token_index in zip(spm_token.pieces, lst):
        # debug
        # assert p.piece == tokenizer._convert_id_to_token(token_index), ValueError("spm_token and lst not equal")
        mask_signal = text_mask[p.begin : p.end]
        if any(mask_signal):
            mask_list.append(ignore_index)
        else:
            mask_list.append(token_index)
    return mask_list


def loss_mask_special_tokens_fn(lst, text, loss_mask_text_dict, tokenizer, ignore_index):
    """loss_mask_special_tokens_fn"""
    mask_flag = [False] * len(text)
    if loss_mask_text_dict["ref_token"] in text:
        mask_flag = find_paired_tag_individual_indices(
            text, mask_flag, loss_mask_text_dict["ref_token"], loss_mask_text_dict["_ref_token"]
        )
    if loss_mask_text_dict["box_token"] in text:
        mask_flag = find_paired_tag_individual_indices(
            text,
            mask_flag,
            loss_mask_text_dict["box_token"],
            loss_mask_text_dict["_box_token"],
            ignore_tokens=loss_mask_text_dict["ignore_tokens"],
        )
    if loss_mask_text_dict["quad_token"] in text:
        mask_flag = find_paired_tag_individual_indices(
            text,
            mask_flag,
            loss_mask_text_dict["quad_token"],
            loss_mask_text_dict["_quad_token"],
            ignore_tokens=loss_mask_text_dict["ignore_tokens"],
        )

    mask_list = loss_mask_with_origin_word_fn(lst, text, mask_flag, tokenizer, ignore_index)
    return mask_list, mask_flag


# def filter_error_file(meta):
#     """filter_error_files"""
#     for item in meta["image_info"]:
#         if not bos.exists(item["bos_url"]):
#             raise FileNotFoundError(f"image_info bos_url {item['bos_url']} not found")


def redbook_add_prompt_on_start(meta):
    """redbook_add_prompt_on_start"""
    meta["text_info"].insert(0, {"text": "来源：小红书\n", "tag": "mask"})
    for item in meta["image_info"]:
        item["matched_text_index"] += 1


def adaptive_partition(img_width, img_height, width=448, height=448, max_tile=15):
    """压缩多少倍 切多少块 for LlaVa-UHD"""

    select_col = 0
    select_row = 0
    select_score = -float("inf")

    for n in range(1, max_tile + 1):
        for m in range(1, max_tile + 1):
            # 最终crop数量 范围 [max_tile-2, max_tile]
            if n * m == max_tile or n * m == max_tile - 1 or n * m == max_tile - 2:
                score = -abs(math.log((img_width * n) / (img_height * m)) - math.log(width / height))

                if score > select_score:
                    select_score = score
                    select_row = n
                    select_col = m

    return select_row, select_col


# NOTE (chw) crop 策略
def partition_image(img, split_type=None):
    """切分高分辨率图片"""
    width, height = img.size
    if split_type is None:
        split_type = [5, 3]

    # 哪边长，哪边切多些
    if width >= height:
        num_width, num_height = max(split_type), min(split_type)
    else:
        num_width, num_height = min(split_type), max(split_type)

    # 计算每一部分的宽度和高度
    part_width = width / num_width
    part_height = height / num_height

    parts = []
    # 遍历图片的每一个部分，并进行切割
    for i in range(num_height):
        one = []
        for j in range(num_width):
            # 计算当前部分的坐标
            left = j * part_width
            top = i * part_height
            right = left + part_width
            bottom = top + part_height

            # 切割图片
            part = img.crop((left, top, right, bottom))
            one.append(part)
        parts.append(one)

    return parts, [num_width, num_height]


def get_rounded(width, height, patch_size=448, all_resize=False):
    """
    根据给定的宽度、高度和块大小，返回调整后的宽度和高度。

    Args:
        width (int): 原始图像的宽度。
        height (int): 原始图像的高度。
        patch_size (int, optional): 块大小，默认为448。
        all_resize (bool, optional): 是否对调整后的宽度和高度都进行块大小调整，默认为False。

    Returns:
        tuple: 调整后的宽度和高度，格式为(width_rounded, height_rounded)。

    """

    def get_rounded_helper(length, patch_size):
        ret = int(round(length / patch_size, 0))
        ret = max(1, ret)
        return ret * patch_size

    if width < height:
        width_rounded = get_rounded_helper(width, patch_size)
        height_rounded = round(height * (width_rounded / width), 0)
        if all_resize:
            height_rounded = get_rounded_helper(height_rounded, patch_size)
    else:
        height_rounded = get_rounded_helper(height, patch_size)
        width_rounded = round(width * (height_rounded / height), 0)
        if all_resize:
            width_rounded = get_rounded_helper(width_rounded, patch_size)

    return width_rounded, height_rounded


### SMART Resize ######
def round_by_factor(number: int, factor: int):
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int):
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int, max_ratio: int):
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > max_ratio:
        logger.info(
            f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
        )
        return height, width

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    if h_bar == 0 or w_bar == 0:
        return height, width
    return h_bar, w_bar
