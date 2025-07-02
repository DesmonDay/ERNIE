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
logging callback
"""

import logging
import os

from paddleformers.trainer.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


class StopperCallback(TrainerCallback):
    """
    主动根据外部信号Stop Training 。（防止kill作业炸鸡）
    """

    def on_substep_end(self, args, state, control, **kwargs):
        """
        在子步骤结束时调用。

        Args:
            args (dict): 参数字典，用于传递其他需要的参数。
            state (State): 当前环境的状态。
            control (Control): 控制对象，用于控制训练过程。
            **kwargs: 其他关键字参数。

        Returns:
            None

        """
        if os.path.exists("/root/stop"):
            control.should_training_stop = True
