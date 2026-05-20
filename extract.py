import imageio.v2 as imageio
import os
from PIL import Image
from tqdm import tqdm

gif_dir = "/root/autodl-tmp/GameScene/multigames_rollouts"
output_dir = "/root/autodl-tmp/GameScene/multigames_datasets"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

gif_files = [f for f in os.listdir(gif_dir) if f.endswith('.gif')]

for gif_name in tqdm(gif_files, desc="按游戏分类拆解"):
    # 假设文件名是 coinrun_0.gif，提取出 "coinrun" 作为文件夹名
    game_label = gif_name.split('_')[0] 
    game_output_path = os.path.join(output_dir, game_label)
    
    if not os.path.exists(game_output_path):
        os.makedirs(game_output_path)
    
    gif_path = os.path.join(gif_dir, gif_name)
    frames = imageio.mimread(gif_path)
    
    # 获取该文件夹下已有的文件数，防止覆盖
    start_id = len(os.listdir(game_output_path))
    
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(os.path.join(game_output_path, f"{start_id + i:06d}.png"))