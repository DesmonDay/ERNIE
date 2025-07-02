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

"""Base prompting class"""

import re
from abc import ABC
from typing import List, Tuple


class BasePrompt(ABC):
    """Base class for prompting"""

    def __init__(self, tokenizer, break_token, break_turn_token):
        self.tokenizer = tokenizer
        self.break_token = break_token
        self.break_turn_token = break_turn_token

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def flatten(s):
        """Return an instance of this string that is marked as shortenable"""
        tokens = []
        for x in s:
            if isinstance(x, tuple):
                tokens.extend(x[0])
            else:
                tokens.extend(x)

        return tokens

    @staticmethod
    def _seq_length(parts: List[Tuple[List, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[List, bool]], just_begin: bool = False, truncate_first: bool = True):
        # print(parts)
        first_idx = min(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)

        idx = first_idx if truncate_first else last_idx
        if just_begin:
            parts[idx] = (parts[idx][0][1:], parts[idx][1])
        else:
            parts[idx] = (parts[idx][0][:-1], parts[idx][1])

    def truncate(self, parts_a: List[Tuple[List, bool]], parts_b: List[Tuple[List, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)

        num_tokens_to_remove = total_len - max_length
        if num_tokens_to_remove <= 0:
            return False, False

        is_parts_a_truncated = False
        is_parts_b_truncated = False
        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
                is_parts_a_truncated = True
            else:
                self._remove_last(parts_b)
                is_parts_b_truncated = True
        return is_parts_a_truncated, is_parts_b_truncated

    def encode(self, src, tgt, max_seq_len):
        """
        对输入进行编码处理。

        Args:
            src (str or list of str): 源文本，可以是一个字符串或字符串列表。
            tgt (str or list of str): 目标文本，与src类型相同。
            max_seq_len (int): 序列的最大长度。

        Returns:
            tuple: 包含四个元素的元组，分别为：
                - flatten_parts_a (list of tuple): 编码后的源文本部分。
                - flatten_parts_b (list of tuple): 编码后的目标文本部分。
                - is_parts_a_truncated (bool): 源文本是否被截断。
                - is_parts_b_truncated (bool): 目标文本是否被截断。

        """
        tokenizer = self.tokenizer

        if isinstance(src, list):
            for i, (x, y) in enumerate(zip(src, tgt)):
                src[i] = x.strip()
                tgt[i] = y.strip()
        else:
            src, tgt = src.strip(), tgt.strip()

        raw_parts_a, raw_parts_b = self.prompt(src, tgt)

        raw_parts_a = [x if isinstance(x, tuple) else (x, False) for x in raw_parts_a]
        raw_parts_b = [x if isinstance(x, tuple) else (x, False) for x in raw_parts_b]

        def encode_input(raw_parts):
            parts = []
            for x, s in raw_parts:
                if isinstance(x, str):
                    x = tokenizer.tokenize(x)
                else:
                    pass
                parts.append((x, s))
            return parts

        parts_a, parts_b = encode_input(raw_parts_a), encode_input(raw_parts_b)

        is_parts_a_truncated, is_parts_b_truncated = self.truncate(parts_a, parts_b, max_seq_len)

        flatten_parts_a = self.flatten(parts_a)
        flatten_parts_b = self.flatten(parts_b)
        return flatten_parts_a, flatten_parts_b, is_parts_a_truncated, is_parts_b_truncated

    def prompt(self, src, tgt):
        """
        生成提示词。

        Args:
            src (str): 源文本。
            tgt (str): 目标文本。

        Returns:
            List[Tuple[str, str]]: 一个包含两个元素的列表，
                第一个元素是一个元组，包含源文本缩短后的结果和分隔符；
                第二个元素是一个包含目标文本缩短后的结果的列表。

        """
        return [self.shortenable(src), self.break_token], [self.shortenable(tgt)]


class SearchPrompt(BasePrompt):
    """Search Prompt"""

    def prompt(self, src, tgt):
        """
        根据输入的源文本和目标文本生成提示文本。

        Args:
            src (list): 源文本列表，包含两个元素，第一个元素为问题，第二个元素为参考文章。
            tgt (list): 目标文本列表，包含两个元素，第一个元素为问题，第二个元素为答案。

        Returns:
            tuple: 包含两个部分的元组，parts_a和parts_b。
                parts_a (list): 包含问题、参考文章和分隔符的列表。
                parts_b (list): 包含答案的列表。

        Raises:
            AssertionError: 如果src或tgt的长度不等于2，则引发断言错误。

        """
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(r"\[<search-res>\](.*?)\[<\/search-res>\]", src[1], re.DOTALL | re.MULTILINE)[0]
        # question_modified = re.findall(r"\[<search>\](.*?)\[<\/search>\]", tgt[0], re.S|re.M)[0]
        answer = tgt[1]

        parts_a = [
            self.shortenable(result),
            "\n根据以上参考文章回答问题，补全对话",
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class KGPrompt(BasePrompt):
    """
    Knowledge Graph Prompt
    """

    def prompt(self, src, tgt):
        """
        根据给定的源数据和目标数据生成提示信息。

        Args:
            src (list): 包含两个元素的列表，第一个元素为问题，第二个元素为知识库信息。
            tgt (list): 包含两个元素的列表，第一个元素为模板问题，第二个元素为答案。

        Returns:
            tuple: 包含两个列表的元组，第一个列表为问题部分，第二个列表为答案部分。

        Raises:
            AssertionError: 如果src或tgt的长度不为2，抛出异常。
        """
        assert len(src) == len(tgt) == 2
        question = src[0]
        result = src[1]
        for markup in [
            "[<kg-res>]",
            "[</kg-res>]",
            "[<kg-yes>]",
            "[</kg-yes>]",
            "[</kg-cs-yes>]",
            "[</kg-cs-yes>]",
            "[</kg-cs-no>]",
            "[</kg-cs-no>]",
            "[<image>]",
            "[</image>]",
        ]:
            result = result.replace(markup, "")

        # question_modified = re.findall(r"\[<kg>\](.*?)\[<\/kg>\]", tgt[0], re.S|re.M)[0]
        answer = tgt[1]

        parts_a = [
            "知识库：",
            self.shortenable(result),
            "\n根据所提供的知识库信息，回答问题并补全对话：",
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class ComputePrompt(BasePrompt):
    """Computation Prompt"""

    def prompt(self, src, tgt):
        """
        向用户提示问题和参考文章。

        Args:
            src (tuple): 一个元组，包含两个字符串元素，第一个元素是问题，第二个元素是参考文章。
            tgt (tuple): 一个元组，包含两个字符串元素，第一个元素是原始问题，第二个元素是答案。

        Returns:
            tuple: 一个元组，包含两个部分，第一个元素是包含参考文章和问题的字符串列表，第二个元素是答案的字符串列表。

        Raises:
            AssertionError: 如果src或tgt的长度不等于2，或者src和tgt的长度不相等，则抛出异常。
        """
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(r"\[<compute-res>\](.*?)\[<\/compute-res>\]", src[1], re.DOTALL | re.MULTILINE)[0]
        # question_modified = re.findall(r"\[<compute>\](.*?)\[<\/compute>\]", tgt[0], re.S|re.M)[0]
        answer = tgt[1]

        parts_a = [
            "参考文章1：",
            self.shortenable(result),
            "\n根据以上参考文章回答问题，补全对话",
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class PromptEngine(BasePrompt):
    """Prompt Engine"""

    def prompt(self, src, tgt):
        """
        根据输入的问题和答案生成提示文本。

        Args:
            src (list): 包含问题和提示文本的列表，其中src[0]为问题，src[1]为提示文本。
            tgt (list): 包含修改后的问题和答案的列表，其中tgt[0]为修改后的问题，tgt[1]为答案。

        Returns:
            tuple: 包含两部分文本的元组，parts_a为问题部分，parts_b为答案部分。

        Raises:
            AssertionError: 如果src或tgt的长度不等于2，或者src[1]中不包含"[<prompt-res>]"标签。

        """
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(r"\[<prompt-res>\](.*?)\[<\/prompt-res>\]", src[1], re.DOTALL | re.MULTILINE)[0]
        # question_modified = re.findall(r"\[<prompt>\](.*?)\[<\/prompt>\]", tgt[0], re.S|re.M)[0]
        answer = tgt[1]

        parts_a = [self.shortenable(result), self.break_turn_token, self.shortenable(question), self.break_token]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class CitationEngine(BasePrompt):
    """
    Citation Engine
    """

    def prompt(self, src, tgt):
        """
        向用户展示问题和搜索结果，并提示用户根据搜索结果回答问题并标注引用。

        Args:
            src (list of str): 包含问题的列表，其中src[0]为问题，src[1]为搜索结果。
            tgt (list of str): 包含答案的列表，其中tgt[0]为问题（此处未使用），tgt[1]为答案。

        Returns:
            tuple: 包含两个列表的元组，parts_a和parts_b。
                parts_a: 包含问题、搜索结果和提示用户回答问题的提示信息的列表。
                parts_b: 包含答案的列表。

        Raises:
            AssertionError: 如果src或tgt的长度不为2，则抛出异常。

        """
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(r"\[<citation-ref>\](.*?)\[<\/citation-ref>\]", src[1], re.DOTALL | re.MULTILINE)[0]
        # question_modified = re.findall(r"\[<citation>\](.*?)\[<\/citation>\]", tgt[0], re.S|re.M)[0]
        answer = tgt[1]

        parts_a = [
            """请参考搜索结果回答下面问题并使用引用标记来标注回答内容参考的搜索结果序号，
            例如^[1]^ (引用单个搜索结果）,^[1][2]^（引用多个搜索结果），其中方括号中的数字是搜索结果序号。
            引用标记只能出现在句尾标点符号前。
            \n以下是搜索结果（每行开头[1]、[2]、...是搜索结果序号），可以对答案中的核心部分进行markdown加粗（**加粗内容**）：\n""",
            self.shortenable(result),
            "\n根据以上搜索结果回答问题并标注引用，补全对话",
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class RetrieveEngine(BasePrompt):
    """Retrieve Engine 用来在构造伪多轮的时候计算拼接长度"""

    def prompt(self, src, tgt):
        """
        生成对话中的问题和答案部分。

        Args:
            src (list of str): 源对话列表，包含问题以及对应的搜索结果。
            tgt (list of str): 目标对话列表，包含修改后的问题以及对应的答案。

        Returns:
            tuple: 包含问题和答案部分的两个列表。

        Raises:
            AssertionError: 如果源对话列表和目标对话列表的长度不为2，则抛出异常。
        """
        assert len(src) == len(tgt) == 2

        question = src[0]
        result = re.findall(r"\[<retrieve-ref>\](.*?)\[<\/retrieve-ref>\]", src[1], re.DOTALL | re.MULTILINE)[0]
        question_modified = re.findall(r"\[<retrieve>\](.*?)\[<\/retrieve>\]", tgt[0], re.DOTALL | re.MULTILINE)[0]
        answer = tgt[1]

        parts_a = [
            """请你扮演一个专家，参考搜索结果中正确、可信、高质量的信息回答问题，并注明答案中引用的搜索结果，
            格式为^[2]^表示引用了第2条搜索结果，^[1][3]^表示引用第1和第3条搜索结果。
            每条搜索结果包含若干相关内容片段。同时你需要遵循以下原则回答问题：
            \n1. 严格遵循搜索结果作答，可以承认不知道答案，并尝试给出一些搜索结果中的相关背景信息。
            \n2. 如果搜索结果存在多种可能的答案，要罗列出每种情况。
            \n3. 如果问题涉及金融、医疗、法律等存在风险的领域，请在结尾提醒用户注意并进行免责说明。
            \n搜索结果：\n""",
            self.shortenable(result),
            "\n\n现在，请根据上面的搜索结果回答问题并标注引用，补全对话",
            self.break_turn_token,
            self.shortenable(question),
            self.break_token,
        ]
        parts_b = [self.shortenable(answer)]
        return parts_a, parts_b


class EBMarkUpRouter:
    """
    EBMarkUpRouter is a router that routes the request to the corresponding engine based on the input text.
    """

    def __init__(self, tokenizer, break_token, break_turn_token) -> None:
        self.search_prompt = SearchPrompt(tokenizer, break_token, break_turn_token)
        self.kg_prompt = KGPrompt(tokenizer, break_token, break_turn_token)
        self.compute_prompt = ComputePrompt(tokenizer, break_token, break_turn_token)
        self.prompt_engine = PromptEngine(tokenizer, break_token, break_turn_token)
        self.citation_engine = CitationEngine(tokenizer, break_token, break_turn_token)
        self.retrieve_engine = RetrieveEngine(tokenizer, break_token, break_turn_token)

    def encode(self, src, tgt, max_seq_len):
        """
        将输入的源和目标序列编码成模型可接受的格式。

        Args:
            src (tuple): 包含两个元素的元组，分别表示源序列和源序列的长度。
            tgt (tuple): 包含两个元素的元组，分别表示目标序列和目标序列的长度。
            max_seq_len (int): 序列的最大长度。

        Returns:
            dict: 编码后的结果，包含模型训练所需的输入数据和标签。

        Raises:
            AssertionError: 如果源序列和目标序列的长度不为2，或者无法识别目标序列的类型。
        """
        assert len(src) == len(tgt) == 2, f"src:{src}, tgt:{tgt}"
        if "[<search" in tgt[0]:
            return self.search_prompt.encode(src, tgt, max_seq_len)
        elif "[<kg" in tgt[0]:
            return self.kg_prompt.encode(src, tgt, max_seq_len)
        elif "[<compute" in tgt[0]:
            return self.compute_prompt.encode(src, tgt, max_seq_len)
        elif "[<prompt" in tgt[0]:
            return self.prompt_engine.encode(src, tgt, max_seq_len)
        elif "[<citation" in tgt[0]:
            return self.citation_engine.encode(src, tgt, max_seq_len)
        elif "[<retrieve" in tgt[0]:
            return self.retrieve_engine.encode(src, tgt, max_seq_len)
        else:
            assert False, f"src:{src}, tgt:{tgt}"
