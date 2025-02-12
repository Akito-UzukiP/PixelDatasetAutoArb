# PIXEL DATASET ARB
用于对像素画数据集进行规整化、分桶的单文件脚本。
主要功能：
- 读取原始数据集（支持递归文件夹结构），自动检测源图片的像素点大小，并规整至指定的大小。
- 将规整后的图片数据集分桶，pad到指定桶分割大小。
- 支持PNG、GIF格式，输出为PNG格式，GIF图片会被拆分为指定数量的PNG图片。
- 支持多线程处理，加速处理速度。

## 使用方法
```
python main.py --input_dir <输入文件夹> --output_dir <输出文件夹> --pad_mode <填充模式> --pad_unit <填充最小单位> --scale_factor <像素块大小> --gif_frames <GIF拆分帧数> --max_workers <最大线程数>
```

## 参数说明
- `--input_dir`：输入文件夹路径，支持递归文件夹结构。
- `--output_dir`：输出文件夹路径。保持原有文件夹结构，将规整后的图片数据集输出到指定文件夹。
- `--pad_mode`：填充模式，可选值为["wrap", "constant", "edge", "reflect", "symmetric"]，默认为"wrap"。
- `--pad_unit`：填充最小单位，按照常见Lora训练脚本，该值应当为32或64，默认为32。
- `--scale_factor`：像素块大小，规整后的像素画的像素点大小，默认为8。
- `--gif_frames`：GIF拆分帧数，当输入文件为GIF格式时，将GIF拆分为指定数量的PNG图片，默认为1。
- `--max_workers`：最大线程数，默认为5。


JPEG、非原生以GIF动图为目的创作的GIF、MP4视频会引入压缩失真，不建议使用。