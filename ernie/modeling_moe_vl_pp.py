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

""" ErnieMoEVLForCausalLMPipe """

import contextlib
import functools
import heapq
import json
import logging
import math
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial, reduce
from itertools import accumulate
from types import MethodType
from typing import Dict, List, Optional, Union

import numpy as np
import paddle
import paddle.distributed as dist
from paddle import framework, nn
from paddle.distributed.communication.batch_isend_irecv import _coalescing_manager
from paddle.distributed.fleet import get_hybrid_communicate_group as get_hcg
from paddle.distributed.fleet.layers.mpu.mp_layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from paddle.distributed.fleet.layers.mpu.random import get_rng_state_tracker
from paddle.distributed.fleet.meta_parallel import LayerDesc, PipelineLayer, SharedLayerDesc
from paddle.distributed.fleet.utils import recompute
from paddle.nn import functional as F
from paddle.utils.layers_utils import flatten, map_structure, pack_sequence_as
from paddleformers.transformers.model_utils import PretrainedModel

from .comm_utils import (
    all_gather_varlen,
    gather_varlen,
    mp_slice,
)
from .configuration import Ernie4_5_MoeConfig, Ernie4_5_VLMoeConfig
from .dfnrope.modeling import (
    DFNRopeVisionTransformerConfig,
    DFNRopeVisionTransformerPretrainedModel,
)
from .distributed import ColumnSequenceParallelLinear, RowSequenceParallelLinear
from .modeling import Ernie4_5_MLP, LayerNorm, RMSNorm
from .modeling_moe import Ernie4_5_DecoderLayer as ErnieMoEDecoderLayer
from .modeling_moe import Ernie4_5_MoeLMHead, _parse_moe_group
from .modeling_moe_pp import (
    EmptyLayer,
    Ernie4_5_EmbeddingPipe,
    create_skip_config_for_refined_recompute,
    get_pp_vp_split_layers,
)
from .modeling_moe_vl import (
    Ernie4_5_VLMoeForConditionalGeneration,
    Ernie4_5_MoeVLHead,
    ErniePretrainingCriterion,
    ModalityDetach,
    TokenType,
    VariableResolutionResamplerModel,
    create_freeze_hook,
    get_backbone_lm_param_regex,
    monkey_patch_param_hook,
)
from .moe.moe_all_gather_layer import MOEAllGatherLayerV2
from .moe.moe_layer import MOELayer
from .moe.topk_gate import TopKGate
from .sequence_parallel_utils import (
    AllGatherVarlenOpV2,
    ScatterOp,
    SliceVarlenOp,
    mark_as_sequence_parallel_parameter,
)

try:
    from ernie.utils.misc import global_training_logs
except ModuleNotFoundError:
    global_training_logs = {}

logger = logging.getLogger(__name__)


def moe_statedict_cherry_pick(state_dict: Dict[str, paddle.Tensor], config: Ernie4_5_MoeConfig):
    """
    pick pamater from state_dict
    """
    moe_num_experts = (
        sum(config.moe_num_experts) if isinstance(config.moe_num_experts, (list, tuple)) else config.moe_num_experts
    )
    if moe_num_experts <= 1:
        return state_dict
    moe_world_size = config.moe_world_size
    if moe_world_size <= 1:
        moe_world_size = 1
    moe_world_size_per_device = moe_num_experts // moe_world_size
    for key in list(state_dict.keys()):
        if "mlp.experts" in key:
            imoe = int(re.search(r"mlp\.experts\.(\d+)", key).group(1))
            maybe_moe_name = key.replace(
                f"mlp.experts.{imoe}", f"mlp.experts.{config.moe_rank * moe_world_size_per_device + imoe}"
            )
            if maybe_moe_name != key and maybe_moe_name in state_dict:
                logger.info(f"moe auto changed state-dict using {maybe_moe_name} as {key}")
                state_dict[key] = state_dict.pop(maybe_moe_name)
    return state_dict


class PipelinePretrainedModel(PretrainedModel):
    """_summary_

    Args:
        PretrainedModel (_type_): _description_
    """

    def __init__(self, config, *args, **kwargs):  # no call
        self.config = config
        super().__init__(config, *args, **kwargs)

    def init(self, config, *args, **kwargs):  # no call
        """Add a single layer to the sequential module."""
        self._sequential_layers = []
        self._pipeline_name_mapping = None
        self._pp_to_single_mapping = None

    def add_sequential_layer(self, layer_desc, name_prefix=""):
        """addd a single layer to the sequential module."""
        self._sequential_layers.append({"layer": layer_desc, "name_prefix": name_prefix})

    def get_sequential_layers(self):
        """get_sequential_layers"""
        return [x["layer"] for x in self._sequential_layers]

    def get_sequential_name_prefixs(self):
        """get sequential_name_prefix"""
        return {str(index): x["name_prefix"] for index, x in enumerate(self._sequential_layers)}

    def get_shardlayer_prefix(self, name_splited):
        """_summary_
            This function retrieves the prefix of a shared layer. The process involves:
            1. Identifying all key names of shared layers, like 'shared_weight01', 'shared_weight02', etc.
            2. For instance, given name_splited = ['shared_layers', 'shared_weight01', 'weight'],
                the 'shared_layer_key' would be name_splited[1], which is 'shared_weight01'.
            3. By traversing through all layers, the function checks if the specified
                shared_layer is present in the current stage. If found, it returns the corresponding prefix.

            Note: For retrieving all SharedLayer instances in Paddle, you can refer to the following Paddle code.
            https://github.com/PaddlePaddle/Paddle/blob/2cf724d055679a1a0e48766dfb1708b920273078/python/paddle/distributed/fleet/meta_parallel/parallel_layers/pp_layers.py#L460-L513
        Args:
            name_splited (_type_): _description_

        Returns:
            _type_: _description_
        """
        shared_layer_names = {s.layer_name for s in self._layers_desc if isinstance(s, SharedLayerDesc)}
        assert name_splited[1] in shared_layer_names, f"The shared layer name {name_splited[1]} must be in prefixes!"
        shared_layer_key = name_splited[1]
        for idx, layer in enumerate(self._layers_desc):
            if isinstance(layer, SharedLayerDesc) and layer.layer_name == shared_layer_key:
                if self.get_stage_from_index(idx) == self._stage_id:
                    return self.get_sequential_name_prefixs()[str(idx)]

        # the prefix must be in the current stage, else raise error
        raise ValueError(f"The shared layer {shared_layer_key} must be in the current stage!")

    def _set_pipeline_name_mapping(self, mappings=None):
        """set pipeline_name_mapping"""
        if mappings is not None:
            self._pipeline_name_mapping = mappings
        else:
            single_to_pp_mapping = {}
            pp_to_single_mapping = {}

            state_dict_keys = list(super().state_dict().keys())
            first_key = ""
            for k in state_dict_keys:
                if "shared_layers" not in k:
                    first_key = k
                    break
            first_key = first_key.split(".")
            # if use virtual pp_degree, the prefix is like 0.0.xxx
            # else it will be like 0.xxx
            use_virtual_pp_degree = first_key[0].isdigit() and first_key[1].isdigit()

            prefixes = self.get_sequential_name_prefixs()
            for k in state_dict_keys:
                name_splited = k.split(".")
                if use_virtual_pp_degree:
                    if name_splited[0].isdigit():
                        if name_splited[1].isdigit():
                            idx = str(int(name_splited[0]) + int(name_splited[1]))
                            single_name = [prefixes[idx]]
                            single_name.extend(name_splited[2:])
                        else:
                            single_name = [prefixes[str(len(prefixes) - 1)]]
                            single_name.extend(name_splited[2:])
                            logger.warning(
                                f"Please check! we treat this key as last layer, get {k}, \
                                        set origin name as {'.'.join(single_name)}"
                            )
                    elif name_splited[0] == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue
                else:
                    idx = name_splited[0]
                    # for normal pp layer
                    if idx.isdigit():
                        # allow empty prefix
                        single_name = [] if prefixes[idx] == "" else [prefixes[idx]]
                        single_name.extend(name_splited[1:])
                    elif idx == "shared_layers":
                        single_name = [self.get_shardlayer_prefix(name_splited)]
                        single_name.extend(name_splited[2:])
                    else:
                        single_to_pp_mapping[k] = k
                        pp_to_single_mapping[k] = k
                        continue

                single_to_pp_mapping[".".join(single_name)] = k
                pp_to_single_mapping[k] = ".".join(single_name)

            self._pipeline_name_mapping = single_to_pp_mapping
            self._pp_to_single_mapping = pp_to_single_mapping

        return self._pipeline_name_mapping

    def state_dict(self, *args, **kwargs):
        """get state_dict"""
        state_dict = super().state_dict(*args, **kwargs)

        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        # assert len(self._pipeline_name_mapping) > 0, "The pipeline stage must have parameters!"

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict[self._pp_to_single_mapping[k]] = v

        return state_dict

    def _init_weights(self, layer):
        """Initialization hook"""
        if self.config.tensor_parallel_degree > 1:
            rng_tracker = get_rng_state_tracker().rng_state
        else:
            rng_tracker = contextlib.nullcontext

        if isinstance(
            layer,
            (
                ColumnParallelLinear,
                RowParallelLinear,
                ColumnSequenceParallelLinear,
                RowSequenceParallelLinear,
                VocabParallelEmbedding,
                # TEFP8Linear,
                Ernie4_5_MoeLMHead,
                nn.Embedding,
                # NativeLinear,
                paddle.incubate.nn.FusedLinear,
            ),
        ):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            # logger.info(f'initializing pp:{type(layer)}')

            # if isinstance(layer, RoundRobinGate):
            #     return

            is_moe = getattr(layer.weight, "no_sync", False)
            with rng_tracker("local_seed" if is_moe else "model_parallel_rng"):
                dtype = paddle.get_default_dtype()  # layer.weight.dtype  #
                # dtype = str(dtype).replace("paddle.", "")
                paddle.set_default_dtype("float32")
                # if isinstance(layer, TEFP8Linear):
                #     layer.weight.set_value(
                #         paddle.transpose(
                #             paddle.randn(layer.weight.shape[::-1], dtype=dtype).scale(self.config.initializer_range),
                #             perm=[1, 0],
                #         )
                #     )
                # else:
                layer.weight.set_value(
                    paddle.randn(layer.weight.shape, dtype=dtype).scale(self.config.initializer_range)
                )
                paddle.set_default_dtype(dtype)
                logger.info(
                    f"dist-init-fc: shape={layer.weight.shape}, range={self.config.initializer_range}, "
                    f"dtype={layer.weight.dtype} "
                    f'type={type(layer)},norm={layer.weight.astype("float32").norm().item()},is_moe={is_moe}'
                )

        elif isinstance(layer, TopKGate):
            if not hasattr(layer, "weight"):  # no weight to initialie : moe-round-robin-gate
                return
            with rng_tracker("model_parallel_rng"):
                dtype = paddle.get_default_dtype()  # layer.weight.dtype  #
                paddle.set_default_dtype("float32")
                moe_inter = self.config.moe_intermediate_size
                moe_num_experts = self.config.moe_num_experts
                if isinstance(moe_inter, (list, tuple)):
                    moe_inter = moe_inter[0]
                if isinstance(moe_num_experts, (list, tuple)):
                    moe_num_experts = moe_num_experts[0]
                if self.config.moe_group_experts:
                    layer.weight.set_value(
                        paddle.randn(layer.weight.shape, dtype=layer.weight.dtype).scale(self.config.initializer_range)
                    )
                else:
                    granularity = 1 if moe_inter == 0 else self.config.intermediate_size // moe_inter
                    layer.weight.set_value(
                        paddle.randn(
                            [self.config.hidden_size, moe_num_experts // granularity],
                            dtype="float32",
                        )
                        .scale(self.config.initializer_range)
                        .repeat_interleave(granularity, axis=-1)
                    )
                logger.info(
                    f"dist-init-moe_gate: shape={layer.weight.shape}, dtype={layer.weight.dtype} "
                    f"range={self.config.initializer_range},type={type(layer)}, "
                    f'norm={layer.weight.astype("float32").norm().item()}'
                )
                if isinstance(self.config.moe_num_experts, (tuple, list)):
                    for i in range(1, len(self.config.moe_num_experts)):
                        layer_weight = getattr(layer, f"weight_{i}")
                        layer_weight.set_value(
                            paddle.randn(layer_weight.shape, dtype=layer_weight.dtype).scale(
                                self.config.initializer_range
                            )
                        )
                        logger.info(
                            f"dist-init-moe_gate: shape={layer_weight.shape}, dtype={layer_weight.dtype} "
                            f"range={self.config.initializer_range},type={type(layer)}, "
                            f'norm={layer_weight.astype("float32").norm().item()}'
                        )
                paddle.set_default_dtype(dtype)

        # elif isinstance(layer, RotaryEmbedding):
        #     head_dim = self.config.hidden_size // self.config.num_attention_heads
        #     inv_freq = 1.0 / (layer.base ** (np.arange(0, head_dim, 2).astype("float32") / head_dim))
        #     # self.register_buffer("inv_freq", inv_freq.cast(dtype))

        #     # higher acc using float32
        #     t = np.arange(layer.max_position_embeddings, dtype="float32")
        #     freqs = np.einsum("i,j->ij", t, inv_freq)
        #     # Different from paper, but it uses a different permutation in order to obtain the same calculation
        #     emb = np.concatenate([freqs, freqs], axis=-1)
        #     # [bs, seqlen, nhead, head_dim]
        #     cos_cached = np.cos(emb)[:, :]  # .astype(dtype)
        #     sin_cached = np.sin(emb)[:, :]  # .astype(dtype)

        #     layer.cos_cached.set_value(cos_cached)  # model后续会被cast成half/bfloat16
        #     layer.sin_cached.set_value(sin_cached)


class ErniePretrainingCriterionPipe(ErniePretrainingCriterion):
    """
    ErniePretrainingCriterionPipe
    """

    def __init__(self, config):
        if config.use_recompute_loss_fn:
            config = deepcopy(config)
            config.sequence_parallel = False  # Do GatherOp in LMHead
        super().__init__(config)

    def forward(self, logits, labels):
        """forward"""
        assert len(labels) in {2, 3}, labels
        assert len(logits) in {4, 8}, logits
        if len(labels) == 2:
            token_type_ids_untouched, labels = labels
            # audio_labels = None
        else:
            token_type_ids_untouched, labels, audio_labels = labels
        if self.config.use_recompute_loss_fn:
            token_type_ids, logits_text, logits_image, logits_audio, *head_and_bias = logits
            # token_type_ids, logits_text, logits_image, *head_and_bias = logits
        else:
            token_type_ids, logits_text, logits_image, logits_audio = logits
            # token_type_ids, logits_text, logits_image = logits
            head_and_bias = ()
        token_type_ids_shifted = token_type_ids[:, 1:]
        loss, _ = super().forward(
            logits_text,
            logits_image,
            labels,
            token_type_ids_shifted,
            token_type_ids_untouched,
            # logits_audio,
            # audio_labels,
            *head_and_bias,
        )
        return loss


@dataclass
class _DtypeSndShape:
    dtype: paddle.dtype
    shape: list

    def size(self):
        """size"""
        return reduce(lambda x, y: x * y, self.shape)


def gather_tensors_list_in_pp_group(inputs, offload_pp_data_chunk_size=0, merge_output=True):
    """
    gather `inputs` from all pp group, send to pp 0 and pp -1
    Args:
        `inputs`: (nested) lists of tensor
        `offload_pp_data_chunk_size`
        `merge_output`:
            if specified, return tensors.
            if no specified, return list of results gather from each pp rank.
    """
    hcg = get_hcg()
    dp_group = hcg.get_pipe_parallel_group()
    dp_worldsize = hcg.get_pipe_parallel_world_size()
    dp_src_rank = dp_group.ranks[0]
    dp_src_rank_last = dp_group.ranks[-1]
    this_rank = dist.get_rank()
    if dp_worldsize <= 1:
        return inputs

    template = map_structure(
        lambda x: (
            _DtypeSndShape(dtype=x.dtype, shape=x.shape) if x is not None else _DtypeSndShape(dtype="", shape=(0,))
        ),
        inputs,
    )
    tensor_flat = flatten(inputs)

    all_template = []
    dist.all_gather_object(all_template, template, group=dp_group)

    gather_meta = []
    dtype = [temp.dtype for temp in flatten(all_template) if temp.dtype != ""]
    if len(dtype) == 0:  # world all none
        nones = sum(map_structure(lambda i: None, all_template), [])
        if this_rank in (dp_src_rank, dp_src_rank_last):
            return nones
        return None

    assert len(set(dtype)) == 1, dtype
    dtype = dtype[0]
    for temps_per_rank in all_template:
        if all([temp.dtype == "" for temp in flatten(temps_per_rank)]):
            gather_meta.append((None, None))
        else:
            gather_meta.append(
                (
                    [
                        sum([np.prod(temp.shape) for temp in flatten(temps_per_rank)]),
                    ],
                    dtype,
                )
            )

    if all([t is None for t in tensor_flat]):
        tensor = None
    else:
        tensor = paddle.concat([t.reshape([-1]) for t in tensor_flat if t is not None], 0)

    gathered_tensor_first = gather_varlen(
        tensor, dp_src_rank, dp_group, offload_pp_data_chunk_size, all_shape_and_dtype=gather_meta
    )
    gathered_tensor_last = gather_varlen(
        tensor, dp_src_rank_last, dp_group, offload_pp_data_chunk_size, all_shape_and_dtype=gather_meta
    )

    gathered_tensor = gathered_tensor_first if len(gathered_tensor_first) > 0 else gathered_tensor_last
    if not len(gathered_tensor):
        return None

    start, end = 0, 0
    ret = []
    for template in all_template:
        ret_per_rank = []
        for temp in flatten(template):
            if np.prod(temp.shape) == 0:
                ret_per_rank.append(None)
                continue
            end += np.prod(temp.shape)
            r = gathered_tensor[start:end].clone().reshape(temp.shape)  # remove clone will trigger 719 error
            ret_per_rank.append(r)
            start = end
        ret_per_rank = pack_sequence_as(template, ret_per_rank)
        ret.append(ret_per_rank)
    if merge_output:
        return sum(ret, [])
    return ret


def exchange_images_meta(images, group):
    """
    exchange_images_meta
    """
    batch_size = paddle.to_tensor(images.shape[0], dtype=paddle.int32)
    batch_size_list = []
    dist.stream.all_gather(batch_size_list, batch_size, group=group, use_calc_stream=True)
    total_batch_size = sum(batch_size_list)
    avg = total_batch_size // len(batch_size_list)
    remain = total_batch_size % len(batch_size_list)
    unbalanced_rank2size = dict(enumerate(batch_size_list))
    sorted_unbalanced_rank2size = dict(sorted(unbalanced_rank2size.items(), key=lambda item: item[1]))
    balanced_rank2size = {key: avg for key in sorted_unbalanced_rank2size.keys()}
    if remain > 0:
        for key in list(sorted_unbalanced_rank2size.keys())[-remain:]:
            balanced_rank2size[key] += 1
    diff_rank2size = {key: sorted_unbalanced_rank2size[key] - balanced_rank2size[key] for key in balanced_rank2size}
    return diff_rank2size


def reshard_images(send_recv_pairs, group, images, reshard_size):
    """
    reshard_images
    """
    rank_in_group = group.rank
    if reshard_size > 0:
        send_meta = list()
        for pair in send_recv_pairs:
            if rank_in_group == pair[0]:
                send_meta.append((pair[1], pair[2]))
        sections = [images.shape[0] - reshard_size]
        for tup in send_meta:
            sections.append(tup[1])
        images_list = paddle.split(images, num_or_sections=sections)
        tasks = []
        with _coalescing_manager(group, tasks):
            for i in range(1, len(images_list)):
                task = dist.isend(images_list[i], group.ranks[send_meta[i - 1][0]], group=group)
                tasks.append(task)
        for task in tasks:
            task.wait()
        return images_list[0]
    elif reshard_size < 0:
        recv_images_list = [images]
        tasks = []
        with _coalescing_manager(group, tasks):
            for pair in send_recv_pairs:
                if rank_in_group == pair[1]:
                    data_shape = images.shape
                    data_shape[0] = pair[2]
                    data = paddle.empty(data_shape, dtype=images.dtype)
                    task = dist.irecv(data, group.ranks[pair[0]], group=group)
                    tasks.append(task)
                    recv_images_list.append(data)
        for task in tasks:
            task.wait()
        return paddle.concat(recv_images_list)
    else:
        return images


def get_send_recv_pairs(diff_rank2size):
    """
    get_send_recv_pairs
    """
    send_rank_size_pairs = list()
    recv_rank_size_pairs = list()
    for key in diff_rank2size:
        if diff_rank2size[key] > 0:
            send_rank_size_pairs.append((-diff_rank2size[key].item(), key))
        elif diff_rank2size[key] < 0:
            recv_rank_size_pairs.append((diff_rank2size[key].item(), key))
    # max heap
    heapq.heapify(send_rank_size_pairs)
    # min heap
    heapq.heapify(recv_rank_size_pairs)

    send_recv_pairs = list()
    while len(send_rank_size_pairs) > 0 and len(recv_rank_size_pairs) > 0:
        src_pair = heapq.heappop(send_rank_size_pairs)
        src_size = -src_pair[0]
        src_rank = src_pair[1]
        dst_pair = heapq.heappop(recv_rank_size_pairs)
        dst_size = -dst_pair[0]
        dst_rank = dst_pair[1]
        if src_size > dst_size:
            send_recv_pairs.append((src_rank, dst_rank, dst_size))
            heapq.heappush(send_rank_size_pairs, (dst_size - src_size, src_rank))
        elif src_size == dst_size:
            send_recv_pairs.append((src_rank, dst_rank, src_size))
        else:
            send_recv_pairs.append((src_rank, dst_rank, src_size))
            heapq.heappush(recv_rank_size_pairs, (src_size - dst_size, dst_rank))
    assert (
        len(send_rank_size_pairs) == 0 and len(recv_rank_size_pairs) == 0
    ), f"send={send_rank_size_pairs} and recv={recv_rank_size_pairs} heap should be empty"
    return send_recv_pairs


def partition_numbers(nums, m):
    """
    Args:
    nums: List[int] - Sorted list of positive integers
    m: int - Number of piles to partition into.

    Returns:
    List[List[int]] - A list containing m lists
    """
    heap = [(0, 0, i) for i in range(m)]
    heapq.heapify(heap)
    piles = [[] for _ in range(m)]
    for num in reversed(nums):
        sum_pile, count_pile, pile_index = heapq.heappop(heap)
        piles[pile_index].append(num)
        sum_pile += num[1] * num[2]
        count_pile += 1
        heapq.heappush(heap, (sum_pile, count_pile, pile_index))

    return piles


def shard_data_in_pp_group(
    fn,
    fwd_batch_size=128,
    scatter_size=2048,
    input_is_parallel=False,
    feature_shape=None,
    is_balanced=False,
    offload_pp_data_chunk_size=0,
    patches_per_image=256,
):
    """shard data in pipe parallel group"""

    @functools.wraps(fn)
    @paddle.no_grad()
    def _wrapper(*args):
        if len(args) == 2:
            images, grid_thw = args
        else:
            images, grid_thw = args[0], None
        if grid_thw is not None:
            assert input_is_parallel, "input_is_parallel must be true when grid_thw is not None"
            assert not is_balanced, "is_balanced must be false when grid_thw is not None"
        hcg = get_hcg()
        dp_group = hcg.get_pipe_parallel_group()
        dp_worldsize = hcg.get_pipe_parallel_world_size()
        dp_src_rank = dp_group.ranks[0]
        this_rank = dist.get_rank()

        if dp_worldsize <= 1:
            if images is None:
                return None
            with paddle.no_grad():
                out = fn(images)
            return out

        if is_balanced:
            pp_sd_group = hcg.pp_sd_group
            diff_rank2size = exchange_images_meta(images, pp_sd_group)
            send_recv_pairs = get_send_recv_pairs(diff_rank2size)
            images = reshard_images(send_recv_pairs, pp_sd_group, images, diff_rank2size[pp_sd_group.rank].item())

        if not input_is_parallel:
            if dp_src_rank == this_rank:
                assert images.ndim == 4, images.shape
                full_image_shape = paddle.shape(images).cuda().astype("int32")
            else:
                full_image_shape = paddle.empty([4], dtype="int32")
            dist.broadcast(full_image_shape, dp_src_rank, group=dp_group)
            full_image_shape = full_image_shape.tolist()
            assert scatter_size % dp_worldsize == 0 and scatter_size >= dp_worldsize, (scatter_size, dp_worldsize)
            # pad to multiply of `scatter_size`
            pad_size = (full_image_shape[0] + scatter_size - 1) // scatter_size * scatter_size - full_image_shape[0]
            # logger.info(f"full_image:{full_image_shape}, pad_size:{pad_size}")
            shareded_images = paddle.empty(
                [(full_image_shape[0] + pad_size) // dp_worldsize] + full_image_shape[1:], dtype="uint8"
            )  # images dtype bfloat16
            for ichunk in range((full_image_shape[0] + pad_size) // scatter_size):
                if images is not None:
                    i = images[ichunk * scatter_size : (ichunk + 1) * scatter_size]
                    assert len(i) <= scatter_size, (len(i), scatter_size)
                    if len(i) < scatter_size:
                        pad_len = int(scatter_size - len(i)) * int(np.prod(images.shape[1:]))
                        # shit hack
                        i = F.pad(i.astype("bfloat16").reshape([-1]), (0, pad_len)).astype("uint8")
                        i = i.reshape([scatter_size] + images.shape[1:])
                else:
                    i = None
                o = shareded_images[
                    ichunk * scatter_size // dp_worldsize : (ichunk + 1) * scatter_size // dp_worldsize
                ]
                dist.stream.scatter(o, i, dp_src_rank, dp_group, use_calc_stream=True)
            images = shareded_images  # release mem

        # logger.info(f'sharded image :{images.shape}')
        if images is not None and len(images) > 0:
            out = []
            with paddle.no_grad():
                if grid_thw is not None:
                    grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
                    grid_thw = F.pad(
                        paddle.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0), [0, 0, 1, 0], value=1
                    )
                    grid_thw_cumsum = F.pad(paddle.prod(grid_thw, -1).cumsum(0), [1, 0])

                    assert grid_thw_cumsum[-1] == len(images), (grid_thw_cumsum[-1], len(images))
                    # logger.info(f"GRID_THW_CUMSUM:{grid_thw_cumsum}")
                    s = 0
                    for i in range(1, len(grid_thw)):
                        if (grid_thw_cumsum[i] - grid_thw_cumsum[s]) >= fwd_batch_size * patches_per_image:
                            # logger.info(f"{patches_cumsum[s]}--{patches_cumsum[i]}")
                            # logger.info(f"{grid_thw_cumsum[s]}--{grid_thw_cumsum[i]}")
                            # logger.info(f"images----{images[grid_thw_cumsum[s]: grid_thw_cumsum[i]]}")
                            o = fn(images[grid_thw_cumsum[s] : grid_thw_cumsum[i]], grid_thw[s:i])
                            s = i
                            out.append(o)
                    if s < len(grid_thw):
                        # logger.info(f"final-{patches_cumsum[s]}--{patches_cumsum[-1]}")
                        # logger.info(f"final-{grid_thw_cumsum[s]}--{grid_thw_cumsum[-1]}")
                        o = fn(images[grid_thw_cumsum[s] :], grid_thw[s:])
                        out.append(o)
                else:
                    s = 0
                    while s < len(images):
                        i = images[s : s + fwd_batch_size]
                        assert len(i) > 0, i.shape
                        o = fn(
                            i,
                            None,
                        )
                        s += fwd_batch_size
                        out.append(o)
            if len(out) == 1:
                if is_balanced:
                    reverse_send_recv_pairs = [(p[1], p[0], p[2]) for p in send_recv_pairs]
                    out = reshard_images(
                        reverse_send_recv_pairs, pp_sd_group, out[0], -diff_rank2size[pp_sd_group.rank].item()
                    )
                else:
                    (out,) = out
            else:
                # I dont know why can not release GPU memory, so I using `_clear_data` to clear underlaying GPU memory
                if offload_pp_data_chunk_size > 0:
                    args[0]._clear_data()
                    images._clear_data()
                out = paddle.concat(out, 0)
                if is_balanced:
                    reverse_send_recv_pairs = [(p[1], p[0], p[2]) for p in send_recv_pairs]
                    out = reshard_images(
                        reverse_send_recv_pairs, pp_sd_group, out, -diff_rank2size[pp_sd_group.rank].item()
                    )
            # self.offload()
            out = out.contiguous()
        else:
            out = None

        if input_is_parallel:
            # gather var len
            gathered = gather_varlen(out, dp_src_rank, dp_group, offload_pp_data_chunk_size)
        else:
            gathered = []
            dist.stream.gather(out, gathered, dp_src_rank, dp_group, use_calc_stream=True)
            if gathered: 
                gathered = paddle.concat(gathered, 0)
                if pad_size > 0:
                    gathered = gathered[:-pad_size]
        if not len(gathered):
            return None
        return gathered

    return _wrapper


def modality_detach(wrapped_class):
    """
    activate `modality_detach` feature in the forward of the wrapped class.
    """
    old_fwd = wrapped_class.forward
    old_init = wrapped_class.__init__

    def new_init(self, *args, **kwargs):
        init = MethodType(old_init, self)
        ret = init(*args, **kwargs)
        self._modality_param_mapping = defaultdict(lambda: [])
        return ret

    def new_fwd(self, args):
        assert isinstance(args, tuple), f"only support wrap PP pipe: {type(self)}"
        assert hasattr(self, "config"), f"cannot get config from self:,type={type(self)}"
        bound_forward = MethodType(old_fwd, self)
        if not self.config.modality_detach:
            return bound_forward(args)
        assert self._modality_param_mapping, f"call `ErnieMoEVLForCausalLMPipe.freeze_lm()` first, self={self}"

        @contextlib.contextmanager
        def freeze_context():
            unfreeze_handler = []
            for _, p, hook in self._modality_param_mapping["lm"]:
                unfreeze_handler.append(p.register_hook(hook, 0))
            yield  # backward fun
            for h in unfreeze_handler:
                assert h.remove(), (p.name, self)

        has_dummy_input = False
        if all(a.stop_gradient for a in args if a is not None):
            fake_tensor = paddle.zeros([])  # add shit to make paddle happy
            fake_tensor.stop_gradient = False
            args = args + (fake_tensor,)
            has_dummy_input = True

        token_type_ids, *args = args

        is_first_fwd = not framework._dygraph_tracer()._has_grad
        ret = ModalityDetach.apply(
            token_type_ids,  # token-type-ids is alwasy the first argument
            *args,
            fn=lambda *args: bound_forward(
                args[:-1] if has_dummy_input else args
            ),  # `ModalityDetach`不接受 tuple 作为参数, 输出必然是tuple
            freeze_context=freeze_context,
        )
        if isinstance(ret, (tuple, list)) and len(ret) == 1:
            (ret,) = ret
        if ret[0] is not None and ret[0].dtype in {paddle.int64, paddle.int32}:
            ret[0].stop_gradient = True  # hack Pylayer的返回值似乎总是 stop_gradient = False, 需要手动改过来
        return ret

    wrapped_class.__init__ = new_init
    wrapped_class.forward = new_fwd
    return wrapped_class


@modality_detach
class ErnieMoELMHeadPipe(Ernie4_5_MoeVLHead):
    """
    support token-type-ids
    """

    def __init__(self, config):
        super().__init__(config)

    @property
    def embedding_weight(self):
        """embedding_weight property"""
        return self.weight

    def forward(self, args):
        """forward"""
        if len(args) == 2:
            token_type_ids, hidden_states = args
            inbatch_pack_offset = None
        else:
            token_type_ids, hidden_states, inbatch_pack_offset = args
        token_type_ids_shifted = token_type_ids[:, 1:]

        # logits_text, logits_image, logits_audio = super().forward(hidden_states, token_type_ids_shifted) note!!
        logits_text, logits_image = super().forward(hidden_states, token_type_ids_shifted)
        token_type_ids = token_type_ids.detach()
        token_type_ids.stop_gradient = True
        if self.config.use_recompute_loss_fn:
            mm_head_weight = self.mm_head.weight if self.mm_head is not None else None
            mm_head_bias = self.mm_head.bias if self.mm_head is not None else None
            return (
                token_type_ids,
                logits_text,
                logits_image,
                None,
                self.weight,
                self.bias,
                mm_head_weight,
                mm_head_bias,
            )
        return token_type_ids, logits_text, logits_image, None


class DFNRopeVisionTransformerPipe(DFNRopeVisionTransformerPretrainedModel):
    """
    DFNRopeVisionTransformerPipe
    """

    def __init__(self, config, use_full_recompute=False):
        self.sorted_thw = None
        self.sorted_idx = None
        self.seq_list = None
        self.new_thw = []
        self.pp_data_balance = getattr(config.vision_config, "pp_data_balance", False)
        self.attn_sep = getattr(config.vision_config, "attn_sep", False)
        self.use_full_recompute = use_full_recompute
        if self.use_full_recompute:
            logger.info("use full recompute, vision model will NOT use recompute inner")
            config.vision_config.recompute = False
        super().__init__(config.vision_config)
        if self.config.tensor_parallel_degree > 1:
            logger.info("use sp extract feature, vit parameter will be marked as sequence parallel")
            for p in self.parameters():
                mark_as_sequence_parallel_parameter(p)

    def extract_feature(self, images, grid_thw, second_fwd=False):
        """extract feature"""
        if self.config.tensor_parallel_degree <= 1:
            return self._extract_feature(images, grid_thw)
        else:
            grid_thw = grid_thw.clone()
            # logger.info("use sp extract feature")
            images_indices = []
            parallelism = self.config.tensor_parallel_degree
            capacity = (grid_thw.prod(-1).sum(-1) + parallelism - 1) // parallelism
            crop_sizes = grid_thw.prod(-1)
            crop_offset = crop_sizes.cumsum(0)
            rank_per_crop = paddle.maximum((crop_offset - 1) // capacity, paddle.to_tensor(0))
            image_size_per_rank = paddle.zeros([parallelism], dtype="int64")
            num_crop_per_rank = paddle.bincount(rank_per_crop, minlength=parallelism)
            image_size_per_rank = paddle.scatter(image_size_per_rank, rank_per_crop, crop_sizes, overwrite=False)

            thw_indices = num_crop_per_rank
            images_indices = image_size_per_rank

            num_pad = 0
            if self.attn_sep:
                seqlen = images.shape[0]
                num_pad = math.ceil(seqlen / parallelism) * parallelism - seqlen
                images = paddle.nn.functional.pad(images, [0, num_pad, 0, 0], value=0)
                images_indices = [images.shape[0] // parallelism for _ in range(parallelism)]
                images = SliceVarlenOp.apply(images, images_indices)
            else:
                images = SliceVarlenOp.apply(images, images_indices)
                images = images.detach()
                grid_thw = mp_slice(grid_thw, thw_indices)

            if len(images):
                image_features = self._extract_feature(images, grid_thw, num_pad=num_pad)
            else:
                image_features = paddle.empty([0, self.config.hidden_size], dtype=self.patch_embed.proj.weight.dtype)
                image_features.stop_gradient = self.patch_embed.proj.weight.stop_gradient
            # sanity check
            if not second_fwd:
                image_features = AllGatherVarlenOpV2.apply(image_features, images_indices)
                if self.attn_sep:
                    image_features = image_features[:seqlen, :]
            # diff = (feas-image_features).abs().mean()
            # logger.info(f'shard vs not shard : {image_features.dtype} {image_features.stop_gradient} {diff}')
            if second_fwd:
                return image_features, images_indices
            return image_features

    def _extract_feature(self, images, grid_thw, num_pad=0):
        """extract feature"""
        ctx = paddle.no_grad if getattr(self.config, "freeze_vision", False) else contextlib.nullcontext
        with ctx():
            image_features = super().forward(images, grid_thw, num_pad)
        return image_features

    def forward(self, args):
        """_summary_

        Args:
            args (_type_): _description_

        Returns:
            _type_: _description_
        """
        raise NotImplementedError


@modality_detach
class ErnieVLEmbeddingPipe(Ernie4_5_EmbeddingPipe):
    """Embedding + Resampler"""

    def __init__(self, config, use_full_recompute=False):
        config = deepcopy(config)
        sequence_parallel = config.sequence_parallel
        config.sequence_parallel = False  # disable inner`ScatterOp`
        self.use_full_recompute = use_full_recompute
        self.offload_resamler = False  # config.pp_recompute_offload_resampler
        # out_dim = config.hidden_size
        super().__init__(config)
        if config.mm_vocab_size > 0:
            self.mm_embed_tokens = VocabParallelEmbedding(config.mm_vocab_size, config.hidden_size)
        else:
            self.mm_embed_tokens = None
        self.resampler_model = VariableResolutionResamplerModel(
            config.pixel_hidden_size,
            config.hidden_size,
            config.spatial_conv_size,
            config.temporal_conv_size,
            config=config,
        )
        self.config = config
        self.scatter_output = sequence_parallel  # outer `ScatterOp`
        self.use_mem_eff_attn = False  # config.use_mem_eff_attn
        # self.use_mem_eff_attn = config.use_mem_eff_attn
        # self.use_mem_eff_attn = True # in train

    def forward(self, args):
        """forward lm embedding + mm embedding + resampler"""
        # assert len(args) == 4, args
        super_forward = super().forward
        token_type_ids, input_ids, *args = args

        def get_args(args, need_inbatch, need_image, need_varres, need_pos):
            """
            get args: inbatch, position-id, image, image_type_ids, grid_thw
            """
            assert isinstance(args, (tuple, list)), type(args)
            keys = [
                i
                for i, j in zip(
                    ["inbatch", "images", "image_type_ids", "grid_thw", "position_ids"],  # args 的出现顺序
                    [need_inbatch, need_image, need_image, need_varres, need_pos],
                )
                if j
            ]
            args = dict(zip(keys, args))
            return (
                args.get("inbatch"),
                args.get("images"),
                args.get("image_type_ids"),
                args.get("grid_thw"),
                args.get("position_ids"),
                # args.get("audio"),
            )

        # inbatch_pack_offset, image_features, image_type_ids, grid_thw, position_ids, audio_ids = get_args(
        inbatch_pack_offset, image_features, image_type_ids, grid_thw, position_ids = get_args(
            args,
            self.use_mem_eff_attn,  # inbatch,
            self.config.vision_config is not None,  # image-type-ids
            getattr(self.config.vision_config, "variable_resolution", False),  # varres
            self.config.rope_3d,  # position-ids
        )
        if inbatch_pack_offset is not None:
            inbatch_pack_offset.stop_gradient = True
        if position_ids is not None:
            position_ids.stop_gradient = True

        token_type_ids_input = token_type_ids[..., :-1]
        token_type_ids_input_ori = token_type_ids_input.clone()
        image_mask = input_ids == self.config.im_patch_id

        token_type_ids_input = token_type_ids_input.flatten()
        input_ids = input_ids.flatten()

        token_type_ids_input[token_type_ids_input == TokenType.video] = TokenType.image
        input_ids.stop_gradient = False  # make recompute happy
        if image_features is not None:
            image_features.stop_gradient = False

        lm_input_ids = input_ids.clone()
        mm_input_ids = input_ids.clone()
        if self.mm_embed_tokens is not None:
            lm_input_ids[token_type_ids_input == TokenType.image] = 0
            mm_input_ids[token_type_ids_input == TokenType.text] = self.config.max_text_id

        def fwd(image_features, _):
            nonlocal input_ids, lm_input_ids, mm_input_ids, token_type_ids_input, image_type_ids, image_mask
            """recompute 段"""
            assert lm_input_ids.max() < self.config.vocab_size, lm_input_ids.tolist()

            inputs_embeds = super_forward(lm_input_ids)
            if isinstance(inputs_embeds, tuple):
                inputs_embeds = inputs_embeds[0]
            if image_features is not None:  # text sample will pass through vit
                # mapping_forward
                image_features = self.resampler_model(
                    image_features,
                    image_mask,
                    token_type_ids_input_ori,
                    image_type_ids,
                    grid_thw,
                )
                # B, N, C = image_features.shape
                # image_features = image_features.reshape([B * N, C])

                if self.mm_embed_tokens is not None:
                    mm_ids_features = self.mm_embed_tokens(mm_input_ids - self.config.max_text_id)
                    mm_ids_features = mm_ids_features.astype(inputs_embeds.dtype)
                    image_indices = paddle.nonzero(token_type_ids_input == TokenType.image).flatten()
                    inputs_embeds = paddle.scatter_(
                        inputs_embeds,
                        image_indices,
                        paddle.gather(mm_ids_features, image_indices, axis=0),
                        overwrite=True,
                    )
                # else:
                # assert (mm_input_ids <= self.config.max_text_id).all().item(), (
                #     f"found vistual token in ids, but `mm_vocab_size` == 0, "
                #     f"ids:{input_ids}, max_text_id={self.config.max_text_id} "
                # )

                image_indices = paddle.nonzero(image_mask.flatten()).flatten()
                image_features = image_features.reshape([-1, image_features.shape[-1]])
                inputs_embeds = paddle.scatter_(
                    inputs_embeds,
                    image_indices,
                    image_features.astype(inputs_embeds.dtype),
                    overwrite=True,
                )

            # if audio_ids is not None:
            #     audio_features = self.audio_embed_tokens(audio_ids)
            #     audio_features = paddle.mean(audio_features, axis=1)
            #     audio_features = self.audio_after_norm(audio_features.astype(self.audio_after_norm.weight.dtype))
            #     audio_indices = paddle.nonzero(token_type_ids_input == TokenType.audio).flatten()
            #     audio_features = audio_features.reshape([-1, audio_features.shape[-1]])
            #     inputs_embeds = paddle.scatter_(
            #         inputs_embeds, audio_indices, audio_features.astype(inputs_embeds.dtype), overwrite=True
            #     )

            if self.scatter_output:
                inputs_embeds = inputs_embeds.reshape([-1, inputs_embeds.shape[-1]])
                inputs_embeds = ScatterOp.apply(inputs_embeds)
            else:
                inputs_embeds = inputs_embeds.reshape(token_type_ids_input_ori.shape + [inputs_embeds.shape[-1]])

            return inputs_embeds

        # `image_features` could be none, add fake tensor to make recompute happy
        fake_tensor = paddle.zeros([])
        fake_tensor.stop_gradient = False

        if self.use_full_recompute and self.training:
            inputs_embeds = recompute(
                fwd,
                image_features,
                fake_tensor,
                offload_indices=[0, 1] if self.offload_resamler else [],
            )
        else:
            inputs_embeds = fwd(image_features, fake_tensor)

        # modify video token type to image token type for expert gating
        token_type_ids[token_type_ids == TokenType.video] = TokenType.image
        ret = (token_type_ids, inputs_embeds)
        if position_ids is not None:
            ret += (position_ids,)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset,)
        return ret


@modality_detach
class ErnieDecoderLayerPipe(ErnieMoEDecoderLayer):
    """_summary_

    Args:
        ErnieDecoderLayer (_type_): _description_
    """

    def __init__(self, config, layer_idx, use_full_recompute=False):
        """initialize"""
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.use_full_recompute = use_full_recompute
        # self.use_mem_eff_attn = False  # config.use_mem_eff_attn
        self.use_meme_eff_attn = config.use_mem_eff_attn  # fix by liaojincheng
        self.sequence_parallel = config.sequence_parallel
        self.rope_3d = config.rope_3d

    def forward(self, args):
        """forward"""
        # assert len(args) == 2, len(args)
        if len(args) == 2:
            token_type_ids, hidden_states = args
            inbatch_pack_offset = None
            position_ids = None
        elif len(args) == 3:
            if self.rope_3d:
                token_type_ids, hidden_states, position_ids = args
                inbatch_pack_offset = None
            else:
                token_type_ids, hidden_states, inbatch_pack_offset = args
                position_ids = None
                inbatch_pack_offset.stop_gradient = True
        token_type_ids = token_type_ids.clone()
        # logger.info(f'hidden_states: {hidden_states.shape}, {hidden_states.astype("float32").norm()}')
        # token_type_ids = token_type_ids.clone().detach()
        # token_type_ids.stop_gradient = True
        if self.training and self.use_full_recompute:
            decoderlayer_act_offload_settings = self.config.get(
                "decoderlayer_act_offload_settings", {"type": "", "value": ""}
            )
            setting_type = decoderlayer_act_offload_settings["type"]
            offload_value = decoderlayer_act_offload_settings["value"]
            offload_kwargs = {}
            if "mod" == setting_type:
                assert isinstance(offload_value, (list, tuple))
                v1, v2 = offload_value
                offload_kwargs["offload_indices"] = [0] if self.layer_idx % v1 == v2 else []
            elif "layer_idxs" == setting_type:
                offload_kwargs["offload_indices"] = [0] if self.layer_idx in offload_value else []

            hidden_states = recompute(
                super().forward,
                hidden_states,
                None,  # attention_mask,
                None,  # attn_mask_start_row_indices
                position_ids,  # position_ids,
                token_type_ids.clone(),  # token-type
                False,  # output-attention
                None,  # past key_value
                False,  # use-cache
                # inbatch_pack_offset,  # inbatch_pack_offset,
                False,  # output_gate_logits
                # **offload_kwargs,
            )
        else:
            hidden_states = super().forward(
                hidden_states,
                None,  # attention_mask,
                None,  # attn_mask_start_row_indices
                position_ids,  # position_ids,
                token_type_ids.clone(),  # token-type
                False,  # output-attention
                None,  # past key_value
                False,  # use-cache
                # inbatch_pack_offset,  # inbatch_pack_offset,
                False,  # output_gate_logits
            )
        ret = (token_type_ids, hidden_states)
        if position_ids is not None:
            ret += (position_ids.clone(),)
        if inbatch_pack_offset is not None:
            ret += (inbatch_pack_offset.clone(),)
        return ret


@modality_detach
class LayerNormPipe(LayerNorm):
    """LayerNormPipe"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        mark_as_sequence_parallel_parameter(self.weight)
        mark_as_sequence_parallel_parameter(self.bias)

    def forward(self, args):
        """forward"""
        token_type_ids, hidden_states, *_ = args
        hidden_states = super().forward(hidden_states)
        token_type_ids.stop_gradient = True
        return token_type_ids, hidden_states


@modality_detach
class RMSNormPipe(RMSNorm):
    """RMSNormPipe"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        mark_as_sequence_parallel_parameter(self.weight)

    def forward(self, args):
        """forward"""
        token_type_ids, hidden_states, *_ = args
        hidden_states = super().forward(hidden_states)
        token_type_ids.stop_gradient = True
        return token_type_ids, hidden_states


def multimodal_data_provider(
    inputs, labels, split_image: Optional[List[int]] = None, use_async=False, image_fea_concated=True
):
    """multimodal data provider"""
    hcg = get_hcg()
    pp_stages = hcg.get_pipe_parallel_world_size()
    pp_stage_id = hcg.get_stage_id()
    is_first_stage = pp_stage_id == 0
    is_last_stage = pp_stage_id == pp_stages - 1

    def check_len(list_of_ten, is_input, num_sample_per_pp_data=1):
        if not image_fea_concated and is_input:
            valid_lens = []
            for i, l in enumerate(list_of_ten):
                if isinstance(l, list):
                    valid_lens.append(len(l) * num_sample_per_pp_data if i == 3 else len(l))
        else:
            valid_lens = [len(l) for l in list_of_ten if isinstance(l, list)]
        assert len(set(valid_lens)) == 1, valid_lens

    if image_fea_concated:
        check_len(inputs, is_input=True)
        check_len(labels, is_input=False)
        acc_steps = len(inputs[0])
    else:
        acc_steps = len(inputs[0])
        num_sample_per_pp_data = acc_steps // len(inputs[3])
        check_len(inputs, is_input=True, num_sample_per_pp_data=num_sample_per_pp_data)
        check_len(labels, is_input=False, num_sample_per_pp_data=num_sample_per_pp_data)

    if is_first_stage:
        labels = None
    if is_last_stage:
        inputs = None

    if not split_image:
        for micro_step in range(acc_steps):
            micro_inputs = (
                tuple(x[micro_step] if isinstance(x, list) else x for x in inputs) if inputs is not None else None
            )
            micro_labels = (
                tuple(l[micro_step] if isinstance(l, list) else l for l in labels) if labels is not None else None
            )
            yield micro_inputs, micro_labels
    else:

        def slice_image(x, start, end):
            if start == end:
                return None
            if image_fea_concated:
                return x.slice((0,), start, end).clone().cuda()
            return x.cuda()._slice(start, end)

        if image_fea_concated:
            split_offset = [
                0,
            ] + list(accumulate(split_image))
            for micro_step in range(acc_steps):
                if inputs is None:
                    micro_inputs = None
                else:
                    micro_inputs = tuple(
                        (
                            slice_image(x, split_offset[micro_step], split_offset[micro_step + 1])
                            if i == 2
                            else x[micro_step] if isinstance(x, list) else x
                        )
                        for i, x in enumerate(inputs)
                    )
                micro_labels = (
                    tuple(l[micro_step] if isinstance(l, list) else l for l in labels) if labels is not None else None
                )
                yield micro_inputs, micro_labels
        else:
            for micro_step in range(acc_steps):
                if inputs is None:
                    micro_inputs = None
                else:
                    micro_inputs = []
                    for i, x in enumerate(inputs):
                        if i == 2:
                            pp_data_idx = micro_step // num_sample_per_pp_data
                            pp_data_idx_offset = micro_step % num_sample_per_pp_data
                            start = pp_data_idx * num_sample_per_pp_data
                            end = start + num_sample_per_pp_data
                            split_offset = [0] + list(accumulate(split_image[start:end]))
                            micro_inputs.append(
                                slice_image(
                                    x[pp_data_idx],
                                    split_offset[pp_data_idx_offset],
                                    split_offset[pp_data_idx_offset + 1],
                                )
                            )
                        elif isinstance(x, list):
                            micro_inputs.append(x[micro_step])
                        else:
                            micro_inputs.append(x)
                    micro_inputs = tuple(micro_inputs)

                micro_labels = (
                    tuple(l[micro_step] if isinstance(l, list) else l for l in labels) if labels is not None else None
                )
                yield micro_inputs, micro_labels


def exchange_pp_imgs_with_thw(
    images, img_thw, img_idx, recv_thw, recv_idx, cur_rank, src_rank_index, dst_rank_index, group
):
    """exchange_pp_imgs_with_thw"""
    tasks = []
    with _coalescing_manager(group, tasks):
        for thw, idx in zip(img_thw, img_idx):
            if thw[src_rank_index] == cur_rank and thw[dst_rank_index] != cur_rank:
                size = thw[1] * thw[2]
                task = dist.isend(images[idx : (idx + size), :], group.ranks[thw[dst_rank_index]], group=group)
                tasks.append(task)
        new_images = []
        new_thw = []
        new_idx = [0]
        old_idx = []
        for thw, idx in zip(recv_thw, recv_idx):
            new_thw.append(thw)
            old_idx.append(idx)
            if thw[src_rank_index] != cur_rank:
                data_shape = thw[1] * thw[2]
                data = paddle.empty([data_shape, images.shape[1]], dtype=images.dtype)
                # dist.stream.recv(data, dp_group.ranks[src_rank], group=dp_group, use_calc_stream=True)
                task = dist.irecv(data, group.ranks[thw[src_rank_index]], group=group)
                tasks.append(task)
                new_images.append(data)
                new_idx.append(new_idx[-1] + data_shape)
            else:
                new_images.append(images[idx : (idx + thw[1] * thw[2]), :])
                new_idx.append(new_idx[-1] + thw[1] * thw[2])
    for task in tasks:
        task.wait()
    new_idx.pop()
    new_images = paddle.concat(new_images, axis=0)

    return new_images, new_thw, new_idx, old_idx


def get_len_and_offset(input_len, group):
    """get length and offset"""
    input_len = paddle.to_tensor(input_len, dtype=paddle.int64)
    length_list = []
    dist.stream.all_gather(length_list, input_len, group=group)
    offset_list = [0]
    for length in length_list:
        offset_list.append(offset_list[-1] + length.item())
    offset_list.pop()
    return length_list, offset_list


class ErnieMoEVLForCausalLMPipe(PipelinePretrainedModel, PipelineLayer):
    """support Pipeline Parallel ERNIE4 """

    config_class = Ernie4_5_VLMoeConfig
    _get_tensor_parallel_mappings = Ernie4_5_VLMoeForConditionalGeneration._get_tensor_parallel_mappings
    _resolve_prefix_keys = Ernie4_5_VLMoeForConditionalGeneration._resolve_prefix_keys
    _init_weights = Ernie4_5_VLMoeForConditionalGeneration._init_weights
    _keep_in_fp32_modules = Ernie4_5_VLMoeForConditionalGeneration._keep_in_fp32_modules

    def _prepare_pipeline_inputs_func(self, data: Union[List, Dict]):
        """
        Convert input data into a format acceptable by the model, including image processing, text processing, etc.

        Args:
            data (Union[List, Dict]): Input data, which can be a list or a dictionary.
            If it is a list, each element should be a dictionary containing all the inputs required by the model.
            The keys in the dictionary include:
            'images', 'grid_thw', 'input_ids', 'audio_ids',
            'token_type_ids', 'image_type_ids', 'labels', 'audio_labels', 'position_ids'.
            'images' represents image data, 'grid_thw'
            represents size and position information of the image,
            'input_ids' represents text ID, 'audio_ids' represents audio ID,
            'token_type_ids' represents the text type ID,
            'image_type_ids' represents the image type ID,
            'labels' represents labels, 'audio_labels' represents audio labels,
            'position_ids' represents position ID.

        Returns:
            Tuple[Dict, Dict]: Returns two dictionaries.
            The first dictionary contains all the input information for the model,
            including 'token_type_ids', 'input_ids', 'image_fea',
            'image_type_ids', 'global_grid_thw', 'position_ids', 'audio_ids';
            the second dictionary contains label information,
            including 'token_type_ids_shifted', 'labels', 'audio_labels'.

        Raises:
            AssertionError: If data is not a list or a dictionary, an AssertionError will be raised .
        """
        assert isinstance(data, list), type(data)
        if getattr(self.config.vision_config, "variable_resolution", False):
            assert (
                not self.balanced_image_preprocess
            ), "balanced_image_preprocess is not supported in variable_resolution"

        all_keys = [
            "images",
            "grid_thw",
            "input_ids",
            "audio_ids",
            "token_type_ids",
            "image_type_ids",
            "labels",
            "audio_labels",
            "position_ids",
        ]
        inputs = []
        for k in all_keys:
            temp = []
            for d in data:
                if k not in d:
                    temp.append(None)
                else:
                    temp.append(d[k])
            inputs.append(temp)

        hcg = get_hcg()
        dp_group = hcg.get_pipe_parallel_group()
        dp_worldsize = hcg.get_pipe_parallel_world_size()
        dp_src_rank = dp_group.ranks[0]
        dp_rank = hcg._get_pipe_parallel_id()
        this_rank = dist.get_rank()

        images, grid_thw, *other_inputs = inputs
        if self.pp_need_data_ranks:
            send_args = [
                grid_thw,
            ] + other_inputs
            recv_args = gather_tensors_list_in_pp_group(send_args, merge_output=False)
            if recv_args is not None:
                recv_args = list(zip(*recv_args))
                global_grid_thw, ids, audio_ids, token_type_ids, image_type_ids, labels, audio_labels, position_ids = (
                    sum(args_from_all_pp, []) for args_from_all_pp in recv_args
                )
            else:
                # middle pp
                global_grid_thw = ids = audio_ids = token_type_ids = image_type_ids = labels = audio_labels = (
                    position_ids
                ) = None
        else:
            ids, audio_ids, token_type_ids, image_type_ids, labels, audio_labels, position_ids = other_inputs
            global_grid_thw = grid_thw
        if ids is not None:  # pp0, pp, -1
            token_type_ids = [t.astype("int32") for t in token_type_ids]
            token_type_ids_shifted = [t[:, 1:] for t in token_type_ids]
        else:
            ids = audio_ids = token_type_ids = image_type_ids = token_type_ids_shifted = labels = audio_labels = None

        if self.vision_model is None:
            images = None
            global_grid_thw = None
            return multimodal_data_provider(
                (token_type_ids, ids, images, image_type_ids, global_grid_thw, position_ids, audio_ids),
                (token_type_ids_shifted, labels, audio_labels),
            )

        if (self.pp_need_data_ranks and dp_rank not in self.pp_need_data_ranks) or (
            not self.pp_need_data_ranks and dp_rank != 0
        ):
            images = []

        image_len_before_concat = paddle.to_tensor(
            [len(n) if n is not None else 0 for i, n in enumerate(images)], dtype="int32"
        )
        images_is_all_none = paddle.to_tensor(all(i is None for i in images), dtype="int32")
        dist.broadcast(images_is_all_none, src=dp_src_rank, group=dp_group)
        if images_is_all_none.item():
            images = None  # no images
            global_grid_thw = None
            return multimodal_data_provider(
                (token_type_ids, ids, images, image_type_ids, global_grid_thw, position_ids, audio_ids),
                (token_type_ids_shifted, labels, audio_labels),
            )

        images = [i for i in images if i is not None]
        images = paddle.concat(images) if len(images) else None  # list -> tensor
        grid_thw = [i for i in grid_thw if i is not None]
        grid_thw = paddle.concat(grid_thw) if len(grid_thw) else None  # list -> tensor

        # start pp data balance
        pp_data_balance = getattr(self.vision_model, "pp_data_balance", False)

        if self.balanced_image_preprocess or self.config.offload_pp_data_chunk_size > 0 or pp_data_balance:
            # to initial group of batch send recv, early do alltoall
            if not hasattr(get_hcg(), "pp_sd_group"):
                # pp_sd_group = create_pp_sd_group()
                pp_sd_group = get_hcg().get_pipe_parallel_group()
                # alltoall to make p2p eager
                fake_data = paddle.ones([pp_sd_group.nranks, 1])
                fake_out = paddle.empty([pp_sd_group.nranks, 1])
                dist.alltoall(fake_out, fake_data, pp_sd_group)
                get_hcg().pp_sd_group = pp_sd_group

        if pp_data_balance:
            # step1: get some infos, like seqlen, grid_thw, for current sort and later restore
            seq_list, seq_idx_list = get_len_and_offset(images.shape[0], dp_group)
            self.vision_model.seq_list = seq_idx_list

            grid_thw = grid_thw[grid_thw > 0].reshape([-1, 3])
            grid_thw = F.pad(paddle.repeat_interleave(grid_thw[:, 1:], grid_thw[:, 0], 0), [0, 0, 1, 0], value=1)

            # get offset
            img_idx = paddle.cumsum(grid_thw[:, 1] * grid_thw[:, 2])
            thwsum = img_idx[-1]
            assert thwsum == images.shape[0], f"thwsum {thwsum}, images.shape {images.shape}"
            img_idx = img_idx[:-1]
            img_idx = F.pad(img_idx, [1, 0], value=0)
            assert (
                img_idx.shape[0] == grid_thw.shape[0]
            ), f"img_idx.shape {img_idx.shape} , grid_thw.shape {grid_thw.shape}"

            # add rank for thw
            rank_column = paddle.full(shape=[grid_thw.shape[0], 1], fill_value=dp_rank, dtype=grid_thw.dtype)
            gridthw_withid = paddle.concat([grid_thw, rank_column], axis=-1)

            # get offset for thw and img of all pp
            thw_len = paddle.to_tensor(gridthw_withid.shape[0], dtype=paddle.int32)
            thw_len_list = []
            dist.stream.all_gather(thw_len_list, thw_len, group=dp_group)
            gathered_gridthw_withid = all_gather_varlen(gridthw_withid, thw_len_list, dp_group)
            gathered_img_idx = all_gather_varlen(img_idx, thw_len_list, dp_group)
            gridthw_withid = gathered_gridthw_withid
            img_idx = gathered_img_idx

            # sort by image size
            gridthw_withid = np.array(gridthw_withid, dtype=np.int64)
            img_idx = np.array(img_idx, dtype=np.int64)
            # products = gridthw_withid[:, 1] * gridthw_withid[:, 2]
            # sorted_indices = np.argsort(products)
            sorted_indices = sorted(
                range(gridthw_withid.shape[0]), key=lambda i: gridthw_withid[i, 1] * gridthw_withid[i, 2]
            )
            sorted_thw = gridthw_withid[sorted_indices]
            sorted_idx = img_idx[sorted_indices]

            indices = np.arange(sorted_thw.shape[0]) % dp_worldsize
            indices = np.expand_dims(indices, axis=-1)
            sorted_thw = np.concatenate((sorted_thw, indices), axis=-1)
            sorted_thw = paddle.to_tensor(sorted_thw, dtype=gridthw_withid.dtype)
            sorted_idx = paddle.to_tensor(sorted_idx, dtype=img_idx.dtype)

            assert sorted_thw.shape[1] == 5, f"{sorted_thw.shape}"
            self.vision_model.sorted_thw = sorted_thw.clone()
            self.vision_model.sorted_idx = sorted_idx.clone()
            # data exchange 
            new_images, new_thw, new_idx, old_idx = exchange_pp_imgs_with_thw(
                images,
                sorted_thw[sorted_thw[:, -2] == dp_rank],
                sorted_idx[sorted_thw[:, -2] == dp_rank],
                sorted_thw[sorted_thw[:, -1] == dp_rank],
                sorted_idx[sorted_thw[:, -1] == dp_rank],
                dp_rank,
                src_rank_index=-2,
                dst_rank_index=-1,
                group=dp_group,
            )

            # data for vit
            images = new_images
            # record old rank and sort rank
            grid_thw_5column = paddle.stack(new_thw, axis=0)
            new_idxes = paddle.to_tensor(new_idx, dtype=img_idx.dtype)
            old_idxes = paddle.to_tensor(old_idx, dtype=img_idx.dtype)
            grid_thw = grid_thw_5column[:, :-2]

        # I dont know why can not release GPU memory, so I using `_clear_data` to clear underlaying GPU memory
        if self.config.offload_pp_data_chunk_size > 0:
            for img in inputs[0]:
                if img is not None:
                    img._clear_data()
        image_len_before_concat_gathered = gather_varlen(image_len_before_concat, dst=dp_src_rank, group=dp_group)
        # logger.info(f"image_len_before_concat_gathered:{image_len_before_concat_gathered}")
        # image_len_before_concat_gathered = paddle.concat(image_len_before_concat_gathered, 0)

        def create_pp_sd_group():
            cur_rank = dist.get_rank()
            cur_world_size = dist.get_world_size()
            hcg = get_hcg()
            pp_world_size = hcg.get_pipe_parallel_world_size()
            sd_world_size = hcg.get_sharding_parallel_world_size()
            pp_sd_world_size = pp_world_size * sd_world_size
            tp_world_size = hcg.get_model_parallel_world_size()
            list_of_ranks = []
            for i in range(tp_world_size):
                ranks = np.arange(i, cur_world_size, tp_world_size)
                list_of_ranks.append(ranks)

            cur_group = None
            for i, ranks in enumerate(list_of_ranks):
                group = dist.new_group(ranks)
                if cur_rank % tp_world_size == i:
                    cur_group = group
            # alltoall to make p2p eager
            fake_data = paddle.ones([cur_group.nranks, 1])
            fake_out = paddle.empty([cur_group.nranks, 1])
            dist.alltoall(fake_out, fake_data, cur_group)
            return cur_group

        @partial(
            shard_data_in_pp_group,
            fwd_batch_size=getattr(self.config.vision_config, "vit_first_fwd_bsz", 128),
            input_is_parallel=len(self.pp_need_data_ranks) > 1,
            is_balanced=self.balanced_image_preprocess,
            offload_pp_data_chunk_size=self.config.offload_pp_data_chunk_size,
        )
        def fwd_image(images, grid_thw):
            # logger.info(f"# image inside shard : {images.shape}")
            if self.image_preprocess is not None:
                assert images.dtype == paddle.uint8, images.dtype
                images = self.image_preprocess.rescale_factor * images.astype("float32")
                images = (images - self.image_preprocess.image_mean_tensor) / self.image_preprocess.image_std_tensor
                images = images.astype("bfloat16")
            else:
                assert images.dtype == paddle.bfloat16, images.dtype

            image_fea = self.vision_model.extract_feature(images, grid_thw)
            if self.config.tensor_parallel_degree > 1:
                if getattr(self.config.vision_config, "variable_resolution", False):
                    S, C = image_fea.shape
                    image_fea = image_fea.reshape([-1, C * self.config.spatial_conv_size**2])
                image_fea = ScatterOp.apply(image_fea, axis=-1)  # mp 切 Fea
                if getattr(self.config.vision_config, "variable_resolution", False):
                    image_fea = image_fea.reshape([S, -1])
            # logger.info(f"# image-fea inside shard : {image_fea.shape}")

            return image_fea

        if self.balanced_image_preprocess:
            # broadcast image shape if needed
            if len(self.pp_need_data_ranks) < dp_worldsize:
                if self.balanced_image_shape is None:
                    pp_sd_group = get_hcg().pp_sd_group
                    src_rank = pp_sd_group.ranks[0]
                    this_rank = dist.get_rank()
                    if src_rank == this_rank:
                        assert images.ndim == 4, images.shape
                        full_image_shape = paddle.shape(images).cuda().astype("int32")
                    else:
                        full_image_shape = paddle.empty([4], dtype="int32")
                    dist.broadcast(full_image_shape, src_rank, group=pp_sd_group)
                    full_image_shape = full_image_shape.tolist()
                    self.balanced_image_shape = full_image_shape
                if dp_rank not in self.pp_need_data_ranks:
                    assert images is None, "pp rank exceed partial pp_need_data must be None"
                    full_image_shape = self.balanced_image_shape
                    full_image_shape[0] = 0
                    images = paddle.empty(full_image_shape, dtype=paddle.uint8)
                else:
                    # check [c, h, w] must be equal
                    assert (
                        images.shape[1:] == self.balanced_image_shape[1:]
                    ), "image shape is not equal to the previous cache shape"

        image_fea = fwd_image(images, grid_thw)
        # if image_fea is not None:
        #     logger.info(f"# image-fea outside shard : {image_fea.shape}")

        if pp_data_balance:
            new_seq_list, new_seq_idx_list = get_len_and_offset(images.shape[0], dp_group)
            new_thw_len_list, new_thw_idx_list = get_len_and_offset(grid_thw_5column.shape[0], dp_group)

            new_gathered_gridthw_withid = all_gather_varlen(grid_thw_5column, new_thw_len_list, dp_group)
            new_gathered_img_idx = all_gather_varlen(new_idxes, new_thw_len_list, dp_group)
            new_gathered_old_idx = all_gather_varlen(old_idxes, new_thw_len_list, dp_group)
            assert (
                new_gathered_gridthw_withid.shape[0] == new_gathered_img_idx.shape[0]
            ), f"{new_gathered_gridthw_withid.shape[0]} != {new_gathered_img_idx.shape[0]}"
            # gather each pp img seq
            if image_fea is not None:
                new_gathered_gridthw_withid = np.array(new_gathered_gridthw_withid, dtype=np.int64)
                new_gathered_img_idx = np.array(new_gathered_img_idx, dtype=np.int64)
                new_gathered_old_idx = np.array(new_gathered_old_idx, dtype=np.int64)
                new_seq_idx_list = np.array(new_seq_idx_list, dtype=np.int64)

                new_fea = []
                for rank in range(dp_group.nranks):
                    # get thw and offset
                    cur_thw = new_gathered_gridthw_withid[new_gathered_gridthw_withid[:, -2] == rank]
                    cur_idx = new_gathered_img_idx[new_gathered_gridthw_withid[:, -2] == rank]
                    old_idx = new_gathered_old_idx[new_gathered_gridthw_withid[:, -2] == rank]

                    sorted_indices = np.argsort(old_idx)
                    sorted_fea_idx = cur_idx[sorted_indices]
                    sorted_fea_thw = cur_thw[sorted_indices]

                    # according to the original offset, restore the order of fea
                    start_offset = new_seq_idx_list[sorted_fea_thw[:, -1]] + sorted_fea_idx
                    end_offset = (
                        new_seq_idx_list[sorted_fea_thw[:, -1]]
                        + sorted_fea_idx
                        + sorted_fea_thw[:, 1] * sorted_fea_thw[:, 2]
                    )
                    index_list = [np.arange(start_offset[i], end_offset[i]) for i in range(len(start_offset))]
                    index_list = paddle.to_tensor(np.concatenate(index_list, axis=-1), dtype=paddle.int64)
                    fea = paddle.gather(image_fea, index_list)
                    new_fea.append(fea)
                new_fea = paddle.concat(new_fea, axis=0)
                image_fea = new_fea

        if image_fea is not None:  # pp 0 or LM batch
            return multimodal_data_provider(
                (token_type_ids, ids, image_fea, image_type_ids, global_grid_thw, position_ids, audio_ids),
                (token_type_ids_shifted, labels, audio_labels),
                split_image=image_len_before_concat_gathered.tolist(),
                image_fea_concated=isinstance(image_fea, paddle.Tensor),
            )
        image_fea = None
        return multimodal_data_provider(
            (token_type_ids, ids, image_fea, image_type_ids, global_grid_thw, position_ids, audio_ids),
            (token_type_ids_shifted, labels, audio_labels),
        )

    def __init__(self, config, recompute=False):
        new_initializer_range = math.sqrt(0.3333 / config.hidden_size)
        logger.info(f"change initializer-range from {config.initializer_range} to {new_initializer_range}")
        config.initializer_range = new_initializer_range
        if config.moe_group in {"mp", "model", "tp", "mpdp"}:
            assert config.sequence_parallel
            logger.info(f"disable FFN tensor model parallel, moe-group={config.moe_group}")
            config.disable_ffn_model_parallel = True

        # add
        config.moe_group_origin = config.moe_group
        config.moe_group = _parse_moe_group(config.moe_group)
        config.moe_world_size = dist.get_world_size(config.moe_group)
        if config.moe_world_size < 0:
            config.moe_world_size = 1
        config.moe_rank = dist.get_rank(config.moe_group)
        hcg = get_hcg()

        self.config = config
        self.image_preprocess = None
        self.pp_need_data_ranks = []  # default to all need data
        self.balanced_image_preprocess = (
            config.balanced_image_preprocess if hasattr(config, "balanced_image_preprocess") else False
        )
        self.balanced_image_shape = None

        tensor_parallel_degree = max(hcg.get_model_parallel_world_size(), 1)
        tensor_parallel_rank = max(hcg.get_model_parallel_rank(), 0)
        logger.info(f"using vpp={config.virtual_pp_degree}")
        if config.sequence_parallel:
            logger.info(f"using sequence_parallel, input seqlen={config.max_sequence_length}")
            assert config.max_sequence_length is not None
            assert (
                config.tensor_parallel_degree > 1
            ), f"sequence-parallel needs mp>1, got mp={config.tensor_parallel_degree}"
        config.tensor_parallel_degree = tensor_parallel_degree
        config.tensor_parallel_rank = tensor_parallel_rank

        if isinstance(config.vision_config, DFNRopeVisionTransformerConfig):
            logger.info("variable resolution vision model")
            config.vision_config.variable_resolution = True
        else:
            raise RuntimeError(f"unknown vision_config: {config.vision_config}")

        # DON'T call PipelinePretrainedModel `__ini__` due to mro issue
        PipelinePretrainedModel.init(self, config=config)

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    key="embed_weight_share",
                    layer_func=ErnieVLEmbeddingPipe,
                    shared_weight_attr="embedding_weight",
                    use_full_recompute=config.use_recompute,
                    config=config,
                ),
                "ernie",
            )
        else:
            self.add_sequential_layer(
                LayerDesc(ErnieVLEmbeddingPipe, config=config, use_full_recompute=config.use_recompute), "ernie"
            )

        no_recompute_layers = get_pp_vp_split_layers(config)

        def _need_full_recompute(layer_idx):
            return layer_idx not in no_recompute_layers and config.recompute

        num_empty_layers = config.remove_tail_layer if isinstance(config.remove_tail_layer, int) else 1

        for i in range(config.num_hidden_layers - num_empty_layers):
            self.add_sequential_layer(
                LayerDesc(
                    ErnieDecoderLayerPipe,
                    config=create_skip_config_for_refined_recompute(i, config),
                    layer_idx=i,
                    use_full_recompute=_need_full_recompute(i),
                ),
                f"ernie.layers.{i}",
            )

        if config.remove_tail_layer:
            for n in range(num_empty_layers):
                self.add_sequential_layer(
                    LayerDesc(
                        EmptyLayer,
                    ),
                    f"empty.layers.{n}",
                )
        else:
            for n in range(num_empty_layers):
                self.add_sequential_layer(
                    LayerDesc(
                        ErnieDecoderLayerPipe,
                        config=create_skip_config_for_refined_recompute(i, config),
                        layer_idx=i,
                        use_full_recompute=_need_full_recompute(i),
                    ),
                    f"ernie.layers.{n + config.num_hidden_layers - num_empty_layers}",
                )

        self.add_sequential_layer(
            LayerDesc(RMSNormPipe if config.use_rmsnorm else LayerNormPipe, config=config), "ernie.norm"
        )

        if config.tie_word_embeddings:
            self.add_sequential_layer(
                SharedLayerDesc(
                    key="embed_weight_share",
                    layer_func=ErnieMoELMHeadPipe,
                    shared_weight_attr="embedding_weight",
                    config=config,
                ),
                "lm_head",
            )
        else:
            self.add_sequential_layer(LayerDesc(ErnieMoELMHeadPipe, config=config), "lm_head")
        recompute_interval = 0

        if 0:  # self.config.pp_first_stage_layers: 
            assert self.config.pp_first_stage_layers >= 2
            _num_layers = len(self.get_sequential_layers())
            _num_stages = get_hcg().topology().get_dim_size("pipe")
            part_size = (_num_layers - self.config.pp_first_stage_layers) // (_num_stages - 1)
            seg_method = [0, self.config.pp_first_stage_layers] + [part_size for i in range(_num_stages - 1)]
            seg_method = list(accumulate(seg_method))
            seg_method[-1] = _num_layers
        else:
            seg_method = "layer:ErnieDecoderLayer|EmptyLayer"
        logger.info(f"using recompute_interval={recompute_interval}, seg_method={seg_method}")

        PipelineLayer.__init__(
            self,
            layers=self.get_sequential_layers(),
            loss_fn=ErniePretrainingCriterionPipe(config),
            topology=get_hcg().topology(),
            seg_method=seg_method,
            recompute_interval=recompute_interval,
            recompute_ctx={
                "mp_group": get_hcg().get_model_parallel_group(),
                "offload": False,
                "partition": False,
            },
            num_virtual_pipeline_stages=config.virtual_pp_degree,
        )
        self.vision_model = None
        self._modality_param_mapping = None

    def add_vision_model(
        self,
        encoder: nn.Layer,
    ):
        """add_vision_model"""
        self.vision_model = encoder
        self._set_modality_param_mapping()

    def add_image_preprocess(self, preprocess):
        """add image_preprocess"""
        logger.info("image preprocess is set")
        self.image_preprocess = preprocess

    def set_pp_need_data_degree(self, p):
        """set pp need data degree"""
        if p == 1:
            logger.warning("you are trying to disable pp-need-data")
            return
        pp_world_size = get_hcg().get_pipe_parallel_world_size()
        no_need_data_range = list(range(p - 1, pp_world_size - 1))
        ranks = [i for i in range(pp_world_size) if i not in no_need_data_range]
        logger.info(f"set `pp_need_data_ranks` to {p}, {ranks}")
        self.pp_need_data_ranks = ranks

    def _set_modality_param_mapping(self, use_stop_grad=True):
        self._set_pipeline_name_mapping()  # TODO
        assert len(self._pipeline_name_mapping) > 0, "The pipeline stage must have parameters!"
        pp_to_single_mapping = {v: k for k, v in self._pipeline_name_mapping.items()}
        lm_pattern = get_backbone_lm_param_regex(self.config)
        self._modality_param_mapping = defaultdict(lambda: [])
        for name, param in self.named_parameters():
            name_split = name.split(".")
            if not name_split[0].isdigit():
                pipe = None
            else:
                if self.config.virtual_pp_degree > 1:
                    pipe = self._sub_layers[name_split[0]]._sub_layers[name_split[1]]
                else:
                    pipe = self._sub_layers[name_split[0]]
            expert_type = getattr(param, "expert_type", None)
            if not use_stop_grad:  # use hook
                monkey_patch_param_hook(param)
            name = pp_to_single_mapping[name]
            if "vision_model" in name:
                self._modality_param_mapping["vit"].append((name, param))
                pipe and pipe._modality_param_mapping["vit"].append((name, param, None))
                param.color = "vit"
            elif expert_type == "expert_type_3":
                self._modality_param_mapping["audio"].append((name, param))
                pipe and pipe._modality_param_mapping["audio"].append((name, param, create_freeze_hook(name, param)))
                param.color = "audio"
            elif lm_pattern.match(name) or expert_type == "expert_type_0":
                self._modality_param_mapping["lm"].append((name, param))
                pipe and pipe._modality_param_mapping["lm"].append((name, param, create_freeze_hook(name, param)))
                param.color = "lm"
            else:
                self._modality_param_mapping["mm"].append((name, param))
                pipe and pipe._modality_param_mapping["mm"].append((name, param, create_freeze_hook(name, param)))
                param.color = "mm"
        debug_msg = {k: [i[0] for i in v] for k, v in self._modality_param_mapping.items()}
        logger.info(f"modality_param_mapping: {json.dumps(debug_msg, ensure_ascii=False, indent=2)}")

    def update_params_stat(self, param_group, stop_gradient):
        """freeze mm"""
        assert param_group in (
            "lm",
            "mm",
            "audio",
            "vit",
        ), "param_group must be in ('lm', 'mm', 'audio', 'vit')"
        if self._modality_param_mapping is None:
            self._set_modality_param_mapping()
        for name, param in self._modality_param_mapping.get(param_group, []):
            # logger.info(f"{param_group}: {name} set_stop_gradient to {stop_gradient}")
            param.stop_gradient = stop_gradient

    def freeze_vision(self):
        """freeze_vision"""
        if self._modality_param_mapping is None:
            self._set_modality_param_mapping()
        for name, param in self._modality_param_mapping.get("vit", []):
            logger.info(f"Freezing vision parameter: {name}")
            param.stop_gradient = True
        self.vision_model.config.freeze_vision = True

    # Initialize weights and apply final processing
    def _post_init(self, original_init, *args, **kwargs):
        """_post_init"""
        super()._post_init(self, original_init, *args, **kwargs)
        with paddle.no_grad():
            if self.config.virtual_pp_degree > 1:
                pipe_layers = (
                    self._sub_layers[i]._sub_layers[j]
                    for i in self._sub_layers
                    for j in self._sub_layers[i]._sub_layers
                )
            else:
                pipe_layers = (self._sub_layers[i] for i in self._sub_layers)

            for i, layer in enumerate(pipe_layers):
                if isinstance(layer, ErnieVLEmbeddingPipe):
                    if getattr(layer, "mm_embed_tokens", None) is not None:
                        layer.mm_embed_tokens.weight.expert_type = "expert_type_1"
                        if getattr(layer.mm_embed_tokens, "bias", None) is not None:
                            layer.mm_embed_tokens.bias.expert_type = "expert_type_1"
                    if getattr(layer, "audio_embed_tokens", None) is not None:
                        layer.audio_embed_tokens.weight.expert_type = "expert_type_3"
                        if getattr(layer.audio_embed_tokens, "bias", None) is not None:
                            layer.audio_embed_tokens.bias.expert_type = "expert_type_3"
                    if getattr(layer, "audio_after_norm", None) is not None:
                        layer.audio_after_norm.weight.expert_type = "expert_type_3"
                        if getattr(layer.audio_after_norm, "bias", None) is not None:
                            layer.audio_after_norm.bias.expert_type = "expert_type_3"

                if isinstance(layer, ErnieMoELMHeadPipe):
                    if getattr(layer, "mm_head", None) is not None:
                        layer.mm_head.weight.expert_type = "expert_type_1"
                        if getattr(layer.mm_head, "bias", None) is not None:
                            layer.mm_head.bias.expert_type = "expert_type_1"
                    if getattr(layer, "audio_out_module", None) is not None:
                        layer.audio_out_module.weight.expert_type = "expert_type_3"
                        if getattr(layer.audio_out_module, "bias", None) is not None:
                            layer.audio_out_module.bias.expert_type = "expert_type_3"

                if isinstance(layer, ErnieDecoderLayerPipe):
                    layer_id = layer.layer_idx  # skip_vocab
                    factor = 1 / math.sqrt(2 * self.config.num_hidden_layers)
                    logger.info(f"using post init div: layer[{layer_id}].factor={factor}")
                    layer.self_attn.o_proj.weight.scale_(factor)
                    if isinstance(layer.mlp, (MOELayer, MOEAllGatherLayerV2)):
                        for e in layer.mlp.experts:
                            if isinstance(e, Ernie4_5_MLP):
                                e.down_proj.weight.scale_(factor)
                        if getattr(layer.mlp, "dense_experts", None) and isinstance(layer.mlp.dense_experts, Ernie4_5_MLP):
                            layer.mlp.dense_experts.down_proj.weight.scale_(factor)
                    else:
                        layer.mlp.down_proj.weight.scale_(factor)

        if not self.config.disable_pipeline_warmup:
            self.pipeline_warmup()

    def pipeline_warmup(self):
        """pipeline_warmup"""
        # warmup moe layer for pp
        try:
            input = None
            hcg = get_hcg()
            mp_size = hcg.get_model_parallel_world_size()
            with paddle.no_grad():
                ori_state = self.training
                self.eval()
                mbs = max(self.config.micro_batch_size, 1)
                token_type_ids = paddle.zeros([mbs, self.config.max_sequence_length + 1]).astype("int32")
                hidden_states = paddle.randn(
                    [mbs * self.config.max_sequence_length // mp_size, self.config.hidden_size]
                )
                logger.info(f"mm-moe pp warmup w/ shape: {hidden_states.shape}")
                input_ids = paddle.zeros([mbs, self.config.max_sequence_length]).astype("int32")
                input = (token_type_ids, hidden_states)
                if self._num_virtual_pipeline_stages <= 1:
                    chunk_id = None
                    if hcg.get_stage_id() == 0:
                        input = (token_type_ids, input_ids, None, None)
                else:
                    if hcg.get_stage_id() == 0:
                        chunk_id = 1
                    else:
                        chunk_id = 0
                output = self.forward(input, chunk_id=chunk_id)
            paddle.device.synchronize()
            if ori_state:
                self.train()
            logger.info("warmup moe layer for pp successfully")
        except Exception as e:
            logger.info(f"failed to warmup moe layer for pp: {e}")
        finally:
            if isinstance(global_training_logs, dict):
                global_training_logs.clear()
            else:
                global_training_logs.reset()

    def set_state_dict(self, state_dict, *args, **kwargs):
        """set state dict"""
        if self._pipeline_name_mapping is None:
            self._set_pipeline_name_mapping()
        assert len(self._pipeline_name_mapping) > 0, "The pipeline stage must have parameters!"

        layer_idxs = []
        if self.config.virtual_pp_degree == 1:
            _layers = iter(self.run_function)
        else:
            _layers = (cc for c in self._model_chunks for cc in c.run_function)

        for layer in _layers:
            if isinstance(layer, ErnieDecoderLayerPipe):
                layer_idxs.append(layer.layer_idx)
        logger.info(f"this pipeline stage has ErnieDecoderLayers: {layer_idxs}")
        state_dict = moe_statedict_cherry_pick(state_dict, self.config)

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if k not in self._pipeline_name_mapping:
                if f"ernie.{k}" in self._pipeline_name_mapping:
                    state_dict[self._pipeline_name_mapping[f"ernie.{k}"]] = v
                continue
            state_dict[self._pipeline_name_mapping[k]] = v
        res = super().set_state_dict(state_dict, *args, **kwargs)
        logger.info(f"ERNIE-MM-MOE-PP - set-state_dict- {res}")
        return res
