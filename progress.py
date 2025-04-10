import time
import sys

def progress_bar(duration=5, resolution=100):
    total_length = 30  # 进度条总长度（字符数）
    
    for i in range(resolution + 1):
        # 计算进度比例
        progress = i / resolution
        # 计算已完成部分长度
        filled = int(total_length * progress)
        # 生成进度条字符串
        bar = '█' * filled + ' ' * (total_length - filled)
        # 计算百分比
        percent = progress * 100
        
        # 输出进度条（使用回车符覆盖前一行）
        sys.stdout.write(f'\r[{bar}] {percent:.1f}% ')
        sys.stdout.flush()
        
        time.sleep(duration / resolution)
    
    print("\n完成！")  # 结束后换行

# 使用示例：持续5秒的进度条
progress_bar(duration=5)