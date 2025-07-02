# !/usr/bin/env python3

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
definition of the tokenizer class.
"""
import os
import random

import numpy as np

from ernie.tokenizer_vl import Ernie4_5_VLTokenizer

coor_num = 1001
NOT_FOUND_TOKEN_ID = -101
NUM_IMAGE_SPECIAL_TOKEN = 2048
NUM_AUDIO_SPECIAL_TOKEN = 1024

SFT_IMAGE_START_TOKEN = "<|IMAGE_START|>"
SFT_IMAGE_END_TOKEN = "<|IMAGE_END|>"
SFT_VIDEO_START_TOKEN = "<|VIDEO_START|>"
SFT_VIDEO_END_TOKEN = "<|VIDEO_END|>"
SFT_ASR_START_TOKEN = "<|ASR_START|>"
SFT_ASR_END_TOKEN = "<|ASR_END|>"

special_tokens_info = {
    "image_placeholder": "<|IMAGE_PLACEHOLDER|>",
    "audio_placeholder": "<|AUDIO_PLACEHOLDER|>",
    "crop": ["<|CROP_COL_SEP|>", "<|CROP_ROW_SEP|>", "<|IMAGE_SEP|>"],
    "loc_coor": [f"<|LOC_{i}|>" for i in range(coor_num)],
    "loc_begin_end": ["<|LOC_BEGIN|>", "<|LOC_END|>", "<|LOC_SEP|>"],
    "image_begin_end": ["<|BOI|>", "<|EOI|>"],
    "video_begin_end": ["<|BOV|>", "<|EOV|>"],
    "sft_video_begin_end": [SFT_VIDEO_START_TOKEN, SFT_VIDEO_END_TOKEN],
}


_tokenizer_cache = {}


def _get_serialized_args(args):
    resulted_args = (
        ("random_seed", args.random_seed),
        ("use_loc_specialtoken", False),  # args.use_loc_specialtoken),
        ("use_crop_specialtoken", False),  # args.use_crop_specialtoken),
        ("tokenizer", args.model_name_or_path),  # args.tokenizer
    )
    return resulted_args


def get_image_special_tokens(
    tokenizer,
    special_tokens_info,
    use_loc_specialtoken=False,
    use_crop_specialtoken=False,
):
    """
    add image special token

    placeholder [<|IMAGE_PLACEHOLDER|>, <|AUDIO_PLACEHOLDER|>, <|VIDEO_PLACEHOLDER|>]

    special tokens [<|BOI|> <|EOI|> <|BOA|> <|EOA|> <|BOV|> <|EOV|>]

    location special tokens [<|LOC_0|> <|LOC_1|> ... <|LOC_1000|>] 1001

    crop special tokens [<|CROP_COL_SEP|>, <|CROP_ROW_SEP|>, <|CROP_IMAGE_SEP|>] 3
        <|CROP_COL_SEP|> for col width
        <|CROP_ROW_SEP|> for row height
        <|CROP_IMAGE_SEP|> for crop and width

    other <|IMAGE_UNUSE:xx|>

    Args:
        tokenizer (ErnieTokenizer): tokenizer
        special_token_ids_start (int, optional): special token begin ids. Defaults to 254208.
        special_token_ids_end (int, optional): special token end ids. Defaults to 256256.
    """
    special_tokens = [
        special_tokens_info["image_placeholder"],
        special_tokens_info["audio_placeholder"],
    ]

    if use_loc_specialtoken:
        special_tokens.extend(special_tokens_info["loc_coor"])
        special_tokens.extend(special_tokens_info["loc_begin_end"])

    if use_crop_specialtoken:
        special_tokens.extend(special_tokens_info["crop"])

    ori_vocab_size = len(tokenizer.get_vocab())
    image_unuse_token_num = max(0, NUM_IMAGE_SPECIAL_TOKEN - len(special_tokens))
    for i in range(image_unuse_token_num):
        special_tokens.append(f'<|IMAGE_UNUSE:{i}|>')

    assert NUM_IMAGE_SPECIAL_TOKEN == len(
        special_tokens
    ), f'Image Special Tokens number is not as expected. expected={NUM_IMAGE_SPECIAL_TOKEN}, now={len(special_tokens)}'
    return special_tokens


def get_tokenizer(args):
    """
    return tokenizer
    """

    serialized_args = _get_serialized_args(args)
    if serialized_args in _tokenizer_cache:
        return _tokenizer_cache[serialized_args]
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    # 1 for use 0 for unuse
    use_loc_specialtoken = False  # args.use_loc_specialtoken == 1
    use_crop_specialtoken = False  # args.use_crop_specialtoken == 1
    if os.path.isdir(args.model_name_or_path):
        tokenizer = Ernie4_5_VLTokenizer.from_pretrained(
            args.model_name_or_path, verbose=False, padding_side="right", model_max_length=args.max_seq_length
        )
    else:
        tokenizer = Ernie4_5_VLTokenizer(
            args.model_name_or_path, verbose=False, padding_side="right", model_max_length=args.max_seq_length
        )
        origin_vocab_size = len(tokenizer.get_vocab())
        image_special_tokens = get_image_special_tokens(
            tokenizer,
            special_tokens_info=special_tokens_info,
            use_loc_specialtoken=use_loc_specialtoken,
            use_crop_specialtoken=use_crop_specialtoken,
        )
        # check vocab_size
        expect_final_vocab_size = origin_vocab_size + NUM_IMAGE_SPECIAL_TOKEN + NUM_AUDIO_SPECIAL_TOKEN
        real_final_vocab_size = len(tokenizer.get_vocab())
        assert (
            real_final_vocab_size == expect_final_vocab_size
        ), f"[ERROR] vocab_size = {real_final_vocab_size} != {expect_final_vocab_size} add too many special tokens!"

        # check image
        image_first_special_tokens = tokenizer.encode(image_special_tokens[0])["input_ids"]
        image_last_special_tokens = tokenizer.encode(image_special_tokens[-1])["input_ids"]

        assert (
            image_first_special_tokens[0] == origin_vocab_size
        ), f"[ERROR] image_first_special_tokens={image_first_special_tokens}"
        assert (
            image_last_special_tokens[0] == origin_vocab_size + NUM_IMAGE_SPECIAL_TOKEN - 1
        ), f"[ERROR] image_last_special_tokens={image_last_special_tokens}"

    tokenizer.ignored_index = -100
    _tokenizer_cache[serialized_args] = tokenizer

    return tokenizer
