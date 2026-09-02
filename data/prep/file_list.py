import os
import shutil

# 类别映射文件路径
classInd_path = 'classInd.txt'

# 训练和验证视频根目录（已经是划分好的目录）
train_root = 'videos_train'
val_root = 'videos_val'

# 读取classInd.txt，构建类别名到label的映射
class_to_idx = {}
with open(classInd_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue  # 跳过空行
        parts = line.split()
        if len(parts) != 2:
            print(f"警告: 第 {line_num} 行格式错误，跳过: {line}")
            continue
        label, cls_name = parts
        try:
            class_to_idx[cls_name] = int(label)
        except ValueError:
            print(f"警告: 第 {line_num} 行 label 不是数字，跳过: {line}")

train_samples = []
val_samples = []

def collect_samples(root_dir, samples_list, subset_name):
    for cls_name in class_to_idx.keys():
        cls_dir = os.path.join(root_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"警告：类别文件夹不存在: {cls_dir}, 跳过")
            continue
        videos = [f for f in os.listdir(cls_dir) if f.endswith('.mp4')]
        for v in videos:
            relative_path = os.path.join(os.path.basename(root_dir), cls_name, v).replace('\\', '/')
            samples_list.append((relative_path, class_to_idx[cls_name]))

# 收集训练集样本
collect_samples(train_root, train_samples, 'train')

# 收集验证集样本
collect_samples(val_root, val_samples, 'val')

# 写入annotation文件函数
def write_ann_file(samples, filepath):
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(filepath, 'w') as f:
        for path, label in samples:
            f.write(f"{path} {label}\n")

train_ann_file = 'train_1.txt'
val_ann_file = 'test_1.txt'

write_ann_file(train_samples, train_ann_file)
write_ann_file(val_samples, val_ann_file)

print("划分完成！")
print(f"训练集视频数: {len(train_samples)}")
print(f"验证集视频数: {len(val_samples)}")
print(f"训练集annotation文件: {train_ann_file}")
print(f"验证集annotation文件: {val_ann_file}")