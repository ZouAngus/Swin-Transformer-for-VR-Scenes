import os
from tqdm import tqdm
import subprocess

def convert_videos_for_vscode(input_folder, output_folder):
    """
    遍历 input_folder 下所有 mp4 视频文件，
    转换为兼容 VSCode 预览的格式，
    并保存到 output_folder，保持文件名不变。
    使用 ffmpeg 命令行进行转换。
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_extensions = ('.mp4', '.MP4')

    files = [f for f in os.listdir(input_folder)
             if f.endswith(video_extensions) and os.path.isfile(os.path.join(input_folder, f))]

    if not files:
        print(f"{input_folder} 中未找到任何mp4视频文件。")
        return

    print(f"Processing {len(files)} videos in folder: {input_folder}")

    for file in tqdm(files, desc=f"Converting videos in {os.path.basename(input_folder)}"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        cmd = [
            'ffmpeg',
            '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-c:a', 'aac',
            '-movflags', '+faststart',
            output_path
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print(f"转换失败: {input_path}，错误信息:\n{e.stderr.decode('utf-8')}")

    print(f"{input_folder} 全部转换完成。")


if __name__ == '__main__':
    input_folder = "data/myvideo/videos_val/Stand"
    output_folder = "data/myvideo/videos_val/Stand_h264"

    if not os.path.exists(input_folder):
        print(f"输入文件夹不存在: {input_folder}，程序退出。")
    else:
        convert_videos_for_vscode(input_folder, output_folder)