import os
import subprocess

games = ["coinrun", "starpilot", "caveflyer"]

# 每个游戏录制 5 次，但都固定同一个关卡
num_rollouts_per_game = 10

# 每次 rollout 录制 100 步
steps_per_rollout = 100

# 固定关卡，不再轮换
fixed_start_level = 0

output_base = "/root/autodl-tmp/GameScene/multigames_rollouts"
os.makedirs(output_base, exist_ok=True)

for game in games:
    print(f"\n🎮 正在录制游戏: {game}")

    for i in range(num_rollouts_per_game):
        out_path = os.path.join(output_base, f"{game}_{i}.gif")

        cmd = [
            "python", "record_procgen_rollout.py",
            "--env", game,
            "--out", out_path,
            "--steps", str(steps_per_rollout),
            "--start-level", str(fixed_start_level),
            "--num-levels", "1"
        ]

        print(f"  ▶ rollout {i}: level={fixed_start_level}, out={out_path}")
        subprocess.run(cmd, check=True)

print("\n✅ 全部录制完成！")