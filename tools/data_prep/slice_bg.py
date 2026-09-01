import os
import cv2
import glob
from tqdm import tqdm

def split_bg_video_sliding_window(input_path, window_size=100, stride=100, output_dir="output_clips"):
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    parts = base_name.split('_')
    if len(parts) < 5:
        print(f"Filename {base_name} format unexpected, skipping")
        return

    orig_start_frame = int(parts[2])  # 文件名里的起始帧
    orig_end_frame = int(parts[3])    # 文件名里的结束帧
    total_frames = int(cv2.VideoCapture(input_path).get(cv2.CAP_PROP_FRAME_COUNT))

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open video file {input_path}, skip.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nProcessing _bg video: {input_path}")
    print(f"FPS: {fps}, Video total frames: {total_frames}, Filename frames: {orig_start_frame}-{orig_end_frame}, Resolution: {width}x{height}")

    if total_frames <= 200:
        print(f"Video length <= 200 frames, no splitting needed.")
        cap.release()
        return

    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    count = 1
    for offset in range(0, total_frames, stride):
        start_frame_in_video = offset
        end_frame_in_video = min(offset + window_size - 1, total_frames - 1)

        # 对应的“全局”帧数 = 文件名起始帧 + offset 及结束帧
        global_start = orig_start_frame + offset
        global_end = orig_start_frame + (end_frame_in_video)

        # 构造新文件名，将原文件名中的起止帧替换成新的
        # 这里把原文件名的第3和第4部分替换成global_start和global_end
        parts[2] = str(global_start)
        parts[3] = str(global_end)
        new_base_name = '_'.join(parts)

        final_output_path = os.path.join(output_dir, new_base_name + ".mp4")

        print(f"Saving clip {count}: video frames {start_frame_in_video}-{end_frame_in_video}, filename frames {global_start}-{global_end} -> {final_output_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_in_video)
        out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))

        for f in range(end_frame_in_video - start_frame_in_video + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Frame read failed at frame {start_frame_in_video + f}")
                break
            out.write(frame)
        out.release()

        count += 1

        if end_frame_in_video == total_frames - 1:
            break  # 到视频末尾了

    cap.release()
    print(f"Done splitting _bg video: {input_path}")

if __name__ == "__main__":
    import glob
    input_folder = "videos_train/Stand"
    output_dir = "videos_train/bg"
    window_size = 100
    stride = 100

    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(input_folder, "*_bg.mp4"))
    print(f"Found {len(video_files)} _bg mp4 videos in {input_folder}")

    for video_path in tqdm(video_files, desc="Processing videos"):
        split_bg_video_sliding_window(video_path, window_size, stride, output_dir)