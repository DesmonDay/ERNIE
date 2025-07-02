# !/usr/bin/env python3
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""image preprocessor adaptive"""

import base64
import logging
import math
import os
import time
from functools import lru_cache
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn.functional as F
import PIL
import requests
from paddleformers.transformers.feature_extraction_utils import BatchFeature
from paddleformers.transformers.image_processing_utils import BaseImageProcessor
from paddleformers.transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from paddleformers.transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from paddleformers.transformers.tokenizer_utils_base import (
    TensorType,
)
from PIL import Image

logger = logging.getLogger(__name__)


OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


VideoInput = Union[
    List["PIL.Image.Image"],
    "np.ndarray",
    "paddle.Tensor",
    List["np.ndarray"],
    List["paddle.Tensor"],
    List[List["PIL.Image.Image"]],
    List[List["np.ndarrray"]],
    List[List["paddle.Tensor"]],
]


def is_scaled_image(image: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].
    """
    if image.dtype == np.uint8:
        return False

    # It's possible the image has pixel values in [0, 255] but is of floating type
    return np.min(image) >= 0 and np.max(image) <= 1


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)) and isinstance(images[0], (list, tuple)) and is_valid_image(images[0][0]):
        return [img for img_list in images for img in img_list]

    elif isinstance(images, (list, tuple)) and is_valid_image(images[0]):
        return images

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched images from {images}")


# Copied from transformers.models.llava_next_video.image_processing_llava_next_video.make_batched_videos
def make_batched_videos(videos) -> List[VideoInput]:
    """dummy"""
    if isinstance(videos, (list, tuple)) and isinstance(videos[0], (list, tuple)) and is_valid_image(videos[0][0]):
        return videos

    elif isinstance(videos, (list, tuple)) and is_valid_image(videos[0]):
        if isinstance(videos[0], Image.Image):
            return [videos]
        elif len(videos[0].shape) == 4:
            return [list(video) for video in videos]

    elif is_valid_image(videos) and len(videos.shape) == 4:
        return [list(videos)]

    raise ValueError(f"Could not make batched video from {videos}")


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
):
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        if height > width:
            new_width = max(factor, round_by_factor(width, factor))
            new_height = floor_by_factor(new_width * MAX_RATIO, factor)
        else:
            new_height = max(factor, round_by_factor(height, factor))
            new_width = floor_by_factor(new_height * MAX_RATIO, factor)

        logger.info(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)},\
              resize to {max(new_height, new_width) / min(new_height, new_width)}"
        )

        height = new_height
        width = new_width

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

    if min_pixels > h_bar * w_bar or h_bar * w_bar > max_pixels:
        raise ValueError(f"encounter invalid h_bar: {h_bar}, w_bar: {w_bar}")

    return h_bar, w_bar


def fetch_image(ele: Dict[str, Union[str, Image.Image]], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    """dummy"""
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        data = image.split(";", 1)[1]
        if data.startswith("base64,"):
            data = base64.b64decode(data[7:])
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    # resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size  # Image, not tensor
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image


def smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    if "nframes" in ele:
        nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        fps = ele.get("fps", FPS)
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        nframes = total_frames / video_fps * fps
        nframes = min(max(nframes, min_frames), max_frames)
        nframes = round_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def is_decord_available() -> bool:
    """dummy"""
    import importlib.util

    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
    ele: dict,
) -> paddle.Tensor:
    import decord

    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)
    if "video_start" in ele or "video_end" in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")
    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = paddle.linspace(0, total_frames - 1, nframes).round().astype("int64").tolist()
    video = vr.get_batch(idx).asnumpy()
    video = paddle.to_tensor(video).transpose([0, 3, 1, 2])  # Convert to TCHW format
    return video


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "paddlevision": None,
}

FORCE_VIDEO_READER = os.getenv("FORCE_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    """dummy"""
    if FORCE_VIDEO_READER is not None:
        video_reader_backend = FORCE_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "paddlevision"
    return video_reader_backend


def custom_resize(video, size, interpolation="bicubic", antialias=True):
    """
    Custom resize function for PaddlePaddle to mimic PyTorch's functionality.

    Args:
        video (paddle.Tensor): Input video tensor of shape [T, C, H, W]
        size (list[int]): Target size [H, W]
        interpolation (str): Interpolation method, default is 'bicubic'
        antialias (bool): Whether to use anti-aliasing, default is True

    Returns:
        paddle.Tensor: Resized video tensor
    """
    # 确保输入是4D张量 [T, C, H, W]
    if video.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {video.ndim}D tensor")

    # 转换为浮点类型
    video = video.astype("float32")

    # 获取原始尺寸
    T, C, H, W = video.shape

    # 设置插值模式
    if interpolation == "bicubic":
        mode = "bicubic"
    elif interpolation == "bilinear":
        mode = "bilinear"
    elif interpolation == "nearest":
        mode = "nearest"
    else:
        raise ValueError(f"Unsupported interpolation mode: {interpolation}")

    # 重塑张量以便于处理
    video = video.reshape([-1, C, H, W])

    # 执行resize操作
    if antialias and mode in ["bicubic", "bilinear"]:
        # PaddlePaddle目前没有直接支持antialias的选项，我们可以通过先下采样再上采样来模拟
        if H > size[0] or W > size[1]:
            # 下采样
            scale_factor = min(size[0] / H, size[1] / W, 1)
            if scale_factor < 1:
                video = F.interpolate(video, scale_factor=scale_factor, mode=mode, align_corners=False)
        # 上采样到目标尺寸
        video = F.interpolate(video, size=size, mode=mode, align_corners=False)
    else:
        video = F.interpolate(video, size=size, mode=mode, align_corners=False)

    # 恢复原始形状
    video = video.reshape([T, C, size[0], size[1]])

    return video


def gaussian_kernel_1d(size, sigma):
    """生成1D高斯核"""
    x = np.arange(-(size // 2), size // 2 + 1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> Union[paddle.Tensor, List[Image.Image]]:
    """dummy"""
    if isinstance(ele["video"], str):
        video_reader_backend = get_video_reader_backend()

        video = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        nframes, _, height, width = video.shape

        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        video = F.interpolate(
            video.astype("float32"),
            size=[resized_height, resized_width],
            mode="bicubic",
            align_corners=False,
            data_format="NCHW",
        )

        video = paddle.clip(video, 0, 255)

        return video

    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images


def extract_vision_info(conversations: Union[List[dict], List[List[dict]]]) -> List[dict]:
    """dummy"""
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: Union[List[dict], List[List[dict]]],
) -> Tuple[Union[List[Image.Image], None, List[Union[paddle.Tensor, List[Image.Image]]]]]:
    """dummy"""
    vision_infos = extract_vision_info(conversations)
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs
