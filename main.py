#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
像素画处理脚本

功能说明：
1. 自动检测图片中每个像素块的实际尺寸，将图片缩小到像素块级别并裁剪透明边界。
2. 放大图片（scale_factor 参数指定，每个像素扩展为 scale_factor×scale_factor），
   然后在放大后的图像上进行补边（padding），使最终输出的宽高为 pad_unit 的倍数。
3. 如果 scale_factor 和 pad_unit 不成整倍数关系，则报错退出。
4. 对 GIF 文件均匀提取指定帧数进行处理。
5. 支持输入文件夹递归查找，并在输出文件夹中创建对应目录结构，多线程并发处理。

使用示例：
    python pixel_processor.py --input_dir ./input --output_dir ./output \
       --pad_mode wrap --pad_unit 32 --scale_factor 8 --gif_frames 1 --max_workers 5
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def detect_pixel_unit_size(image_path):
    """
    检测像素画中单个像素块的实际尺寸。
    若检测失败则返回 1。
    """
    try:
        img = Image.open(image_path)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img is None:
            print(f"警告：无法读取图片 {image_path}")
            return 1
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diff_x = np.abs(np.diff(gray, axis=1))
        diff_y = np.abs(np.diff(gray, axis=0))
        edges_x = np.where(diff_x > 0)[1]
        edges_y = np.where(diff_y > 0)[0]
        if len(edges_x) == 0 or len(edges_y) == 0:
            return 1
        distances_x = np.diff(np.sort(edges_x))
        distances_y = np.diff(np.sort(edges_y))
        distances_x = distances_x[distances_x > 0]
        distances_y = distances_y[distances_y > 0]
        if len(distances_x) == 0 or len(distances_y) == 0:
            return 1
        pixel_size_x = np.bincount(distances_x).argmax()
        pixel_size_y = np.bincount(distances_y).argmax()
        pixel_size = min(pixel_size_x, pixel_size_y)
        return pixel_size if pixel_size > 0 else 1
    except Exception as e:
        print(f"处理图片 {image_path} 时发生错误: {str(e)}")
        return 1

def get_pad(old_size, pad_unit):
    """
    计算使得 old_size 补齐到 pad_unit 的倍数时，
    在两侧各需要补充的数量（尽量均分）。
    """
    remainder = old_size % pad_unit
    if remainder == 0:
        return 0, 0
    needed = pad_unit - remainder
    pad_before = needed // 2
    pad_after = needed - pad_before
    return pad_before, pad_after

def process_image(image_path, output_path, pad_unit, pad_mode, scale_factor,
                  constant_fill_color=(255, 255, 255, 255)):
    """
    处理单张图片：
      1. 检测像素块尺寸，将图片缩小到像素块级别；
      2. 裁剪非透明区域；
      3. 放大图片（upsample）；
      4. 对放大后的图像进行补边，使其宽高为 pad_unit 的倍数；
      5. 保存至 output_path。
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGBA':
                img = img.convert('RGBA')

            # 检测像素块尺寸
            pixel_size = detect_pixel_unit_size(image_path)
            if pixel_size <= 0:
                pixel_size = 1

            width, height = img.size
            # 缩小到像素块级别
            new_width = width // pixel_size
            new_height = height // pixel_size
            if new_width < 1 or new_height < 1:
                new_width = max(1, new_width)
                new_height = max(1, new_height)
            img_min = img.resize((new_width, new_height), Image.NEAREST)

            # 裁剪至非透明区域
            img_np = np.array(img_min)
            alpha_channel = img_np[:, :, 3]
            non_transparent = np.where(alpha_channel != 0)
            if non_transparent[0].size == 0:
                print(f"图片 {image_path} 全透明，跳过处理。")
                return
            min_row = non_transparent[0].min()
            max_row = non_transparent[0].max()
            min_col = non_transparent[1].min()
            max_col = non_transparent[1].max()
            img_cropped = img_min.crop((min_col, min_row, max_col + 1, max_row + 1))

            # 放大（upsample）图像
            up_width = img_cropped.width * scale_factor
            up_height = img_cropped.height * scale_factor
            img_upsampled = img_cropped.resize((up_width, up_height), Image.NEAREST)

            # 在 upsample 后对图像补边，使尺寸为 pad_unit 的倍数
            up_np = np.array(img_upsampled)
            final_h, final_w = up_np.shape[0], up_np.shape[1]
            pad_top, pad_bottom = get_pad(final_h, pad_unit)
            pad_left, pad_right = get_pad(final_w, pad_unit)
            pad_width_cfg = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

            if pad_mode == "constant":
                new_h = final_h + pad_top + pad_bottom
                new_w = final_w + pad_left + pad_right
                padded = np.full((new_h, new_w, 4), constant_fill_color, dtype=up_np.dtype)
                padded[pad_top:pad_top + final_h, pad_left:pad_left + final_w, :] = up_np
            else:
                padded = np.pad(up_np, pad_width=pad_width_cfg, mode=pad_mode)

            # 对 alpha=0 的像素进行处理（填充为 constant_fill_color）
            alpha_channel = padded[:, :, 3]
            transparent_mask = (alpha_channel == 0)
            padded[transparent_mask] = constant_fill_color

            final_img = Image.fromarray(padded, mode='RGBA')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            final_img.save(output_path)
            # print(f"已处理 {image_path} -> {output_path}")

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")

def process_gif(gif_path, output_dir, pad_unit, pad_mode, scale_factor, gif_frames,
                constant_fill_color=(255, 255, 255, 255)):
    """
    处理 GIF 文件：
      1. 均匀提取 gif_frames 帧；
      2. 对每一帧调用 process_image 进行处理，输出文件名格式为 <原文件名>_frame<帧号>.png
    """
    try:
        with Image.open(gif_path) as img:
            frame_count = getattr(img, "n_frames", 1)
            frames_to_extract = min(frame_count, gif_frames)
            base_name = os.path.splitext(os.path.basename(gif_path))[0]

            if frame_count <= frames_to_extract:
                frame_indices = list(range(frame_count))
            else:
                intervals = np.linspace(0, frame_count - 1, frames_to_extract)
                frame_indices = [int(round(i)) for i in intervals]

            for idx, frame_number in enumerate(frame_indices):
                img.seek(frame_number)
                frame = img.copy().convert('RGBA')
                frame_filename = f"{base_name}_frame{idx}.png"
                frame_output_path = os.path.join(output_dir, frame_filename)
                frame.save(frame_output_path)
                process_image(frame_output_path, frame_output_path, pad_unit, pad_mode, scale_factor, constant_fill_color)
    except Exception as e:
        print(f"处理 GIF {gif_path} 时出错: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="像素画处理脚本：缩小、裁剪、放大后进行补边，使最终图片尺寸为 pad_unit 的倍数；支持 GIF 帧提取与多线程处理。"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="输入文件夹路径（支持递归搜索）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出文件夹路径")
    parser.add_argument("--pad_mode", type=str, default="wrap",
                        choices=["wrap", "constant", "edge", "reflect", "symmetric"],
                        help="补边方式，默认为 'wrap'。若选择 'constant' 则用常数颜色填充")
    parser.add_argument("--pad_unit", type=int, default=32,
                        help="补边的单位：最终输出图片的宽高将补齐到 pad_unit 的倍数，默认为 32")
    parser.add_argument("--scale_factor", type=int, default=8,
                        help="放大倍数，即像素画中每个像素点的大小（例如 2、4、8 等），默认为 8")
    parser.add_argument("--gif_frames", type=int, default=1,
                        help="对于 GIF 文件，提取的帧数，默认为 1")
    parser.add_argument("--max_workers", type=int, default=5,
                        help="线程池中最大的工作线程数，默认为 5")
    args = parser.parse_args()

    # 检查 scale_factor 和 pad_unit 是否成整倍数关系
    if args.pad_unit % args.scale_factor != 0:
        print("错误：pad_unit 必须是 scale_factor 的整倍数。")
        sys.exit(1)

    # 在输出文件夹中构建与输入文件夹相同的目录结构
    for root, dirs, files in os.walk(args.input_dir):
        rel_dir = os.path.relpath(root, args.input_dir)
        target_dir = os.path.join(args.output_dir, rel_dir)
        os.makedirs(target_dir, exist_ok=True)

    # 遍历所有待处理图片（目前支持 .png 与 .gif）
    image_files = []
    valid_extensions = ('.png', '.gif')
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, args.input_dir)
                out_path = os.path.join(args.output_dir, rel_path)
                image_files.append((full_path, out_path))

    # 多线程处理图片
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for in_path, out_path in image_files:
            if in_path.lower().endswith('.png'):
                futures.append(
                    executor.submit(process_image, in_path, out_path,
                                    args.pad_unit, args.pad_mode, args.scale_factor)
                )
            elif in_path.lower().endswith('.gif'):
                # 对 GIF 文件，输出路径设为不含扩展名的文件夹，用于存放各帧图片
                gif_out_dir = os.path.splitext(out_path)[0]
                os.makedirs(gif_out_dir, exist_ok=True)
                futures.append(
                    executor.submit(process_gif, in_path, gif_out_dir,
                                    args.pad_unit, args.pad_mode, args.scale_factor, args.gif_frames)
                )
        for future in tqdm(futures, desc="等待处理线程"):
            try:
                future.result()
            except Exception as e:
                print(f"线程处理出错: {e}")

if __name__ == "__main__":
    main()
