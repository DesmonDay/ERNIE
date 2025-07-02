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


class LoggingCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

    def on_train_begin(self, *args, **kwargs):
        """
        在训练开始时执行的回调方法。

        Args:
            *args: 不定长位置参数，实际使用中未使用。
            **kwargs: 不定长关键字参数，实际使用中未使用。

        Returns:
            None

        """

        def _log(x):
            return os.popen(x).read()

        logger.info(
            f"""
                HOSTNAME: {os.environ.get('HOSTNAME')}
                CLUSTER_NAME: {os.environ.get('CLUSTER_NAME')}

                CPU INFO
                    [ ] BASIC: \n{_log('lscpu')}

                GPU INFO
                    [ ] NVIDIA-SMI: \n{_log('nvidia-smi |grep NVIDIA-SMI')}
                    [ ] GPU MEMORY: \n{_log('nvidia-smi |grep Default')}

                DISK INFO
                    [ ] BASIC: \n{_log('lsblk')}

                MEMORY INFO
                    [ ] BASIC: \n{_log('free -h')}

                ENVS:
                    [ ] BASIC: \n{_log('env')}
                """
        )
        logger.info(os.popen("git status").read())
        logger.info(os.popen("git log -1").read())

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        处理日志信息，包括将数据ID和源ID转换为字符串并添加到日志中。
            如果传入了`metrics_dumper`参数，则将日志信息添加到其中。

            Args:
                args (Any, optional): 可选参数，默认值为None。
                state (Any, optional): 可选参数，默认值为None。
                control (Any, optional): 可选参数，默认值为None。
                logs (List[Dict], optional): 可选参数，默认值为None，日志列表，格式为字典，每个字典包含一组键值对。
                kwargs (Dict, optional): 可选参数，默认值为空字典，其他额外的关键字参数。包含`inputs`键，其值是一个字典，包含`data_id`和`src_id`键，分别表示数据ID和源ID。

            Returns:
                None: 该函数没有返回值。

            Raises:
                None: 该函数没有引发任何异常。
        """
        _ = logs.pop("total_flos", None)
        if "inputs" in kwargs:
            data_id = kwargs["inputs"].get("data_id", None)
            src_id = kwargs["inputs"].get("src_id", None)
            data_type = kwargs["inputs"].get("data_type", None)

            if data_id is not None:
                logs = dict(logs, data_id="-".join(map(str, (data_id.numpy().tolist()))))
            if src_id is not None:
                logs = dict(logs, src_id="-".join(map(str, (src_id.numpy().tolist()))))
            if data_type is not None:
                logs.update(data_type="-".join(map(str, (data_type.numpy().tolist()))))

        if type(logs) is dict:
            logger.info(
                ", ".join(
                    (
                        (f"{k}: {v}" if k == "loss" or "cur_dp" in k else f"{k}: {v:e}" if v < 1e-3 else f"{k}: {v:f}")
                        if isinstance(v, float)
                        else f"{k}: {v}"
                    )
                    for k, v in logs.items()
                )
            )
            metrics_dumper = kwargs.get("metrics_dumper", None)
            if metrics_dumper is not None:
                metrics_dumper.append(logs)
        else:
            logger.info(logs)
