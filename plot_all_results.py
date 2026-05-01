import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_and_save_curves():
	summary_csv_path = "experiments_summary.csv"
	results_dir = "results"

	# 1. 如果结果文件夹不存在，则自动创建
	os.makedirs(results_dir, exist_ok=True)

	# 2. 检查汇总文件是否存在
	if not os.path.exists(summary_csv_path):
		print(f"未找到 {summary_csv_path}，请确认你位于项目根目录。")
		return

	# 3. 读取记录了每次运行参数和路径的汇总文件
	summary_df = pd.read_csv(summary_csv_path)

	# 4. 遍历每一次实验记录
	for index, row in summary_df.iterrows():
		run_id = str(row['run_id'])
		metrics_path = row['detailed_csv_path']

		# 设定该 run_id 对应图片的保存路径
		save_img_path = os.path.join(results_dir, f"curves_{run_id}.png")

		# 自动跳过已经绘制过的结果
		if os.path.exists(save_img_path):
			print(f"⏩ 跳过: {save_img_path} (已存在)")
			continue

		# 检查对应的 metrics_xxx.csv 文件是否存在
		if not os.path.exists(metrics_path):
			print(f"⚠️ 警告: 找不到指标文件 {metrics_path}，跳过此 run_id: {run_id}")
			continue

		print(f"📊 正在绘制 run_id: {run_id} 的曲线...")

		# 读取该次实验的具体指标数据
		try:
			df = pd.read_csv(metrics_path)
		except Exception as e:
			print(f"❌ 读取 {metrics_path} 失败: {e}")
			continue

		# 确保 MIA(%) 是数值类型
		if 'MIA(%)' in df.columns:
			df['MIA(%)'] = pd.to_numeric(df['MIA(%)'], errors='coerce')

		# 5. 开始绘制 2x2 的图表
		fig, axs = plt.subplots(2, 2, figsize=(14, 10))
		fig.suptitle(f"Training/Unlearning Curves for Run ID: {run_id}", fontsize=16, fontweight='bold')

		# ================= ① 遗忘客户端 ASR (左上) =================
		if 'ASR(%)' in df.columns:
			axs[0, 0].plot(df['Round'], df['ASR(%)'], marker='o', color='red', label='ASR')
		axs[0, 0].set_title('① Unlearned Client ASR vs Rounds')
		axs[0, 0].set_xlabel('Round')
		axs[0, 0].set_ylabel('ASR (%)')
		axs[0, 0].grid(True, linestyle='--', alpha=0.6)
		axs[0, 0].legend()

		# ================= ② 遗忘客户端 MIA Accuracy (右上) =================
		if 'MIA(%)' in df.columns:
			mia_df = df.dropna(subset=['MIA(%)'])
			if not mia_df.empty:
				axs[0, 1].plot(mia_df['Round'], mia_df['MIA(%)'], marker='d', color='purple', label='MIA Acc')
		axs[0, 1].axhline(y=50, color='gray', linestyle='--', label='Random Guess (50%)')  # MIA 完美遗忘基线
		axs[0, 1].set_title('② Unlearned Client MIA Accuracy vs Rounds')
		axs[0, 1].set_xlabel('Round')
		axs[0, 1].set_ylabel('MIA Accuracy (%)')
		axs[0, 1].grid(True, linestyle='--', alpha=0.6)
		axs[0, 1].legend()

		# ================= ③ Retained Client Local Test Acc (左下) =================
		if 'Retained_Acc(%)' in df.columns:
			axs[1, 0].plot(df['Round'], df['Retained_Acc(%)'], marker='^', color='green', label='Retained Local Acc')
		axs[1, 0].set_title('③ Retained Client Local Test Acc vs Rounds')
		axs[1, 0].set_xlabel('Round')
		axs[1, 0].set_ylabel('Local Accuracy (%)')
		axs[1, 0].grid(True, linestyle='--', alpha=0.6)
		axs[1, 0].legend()

		# ================= ④ Global Test Acc (右下) =================
		if 'Global_Acc(%)' in df.columns:
			axs[1, 1].plot(df['Round'], df['Global_Acc(%)'], marker='s', color='blue', label='Global Acc')
		axs[1, 1].set_title('④ Global Test Acc vs Rounds')
		axs[1, 1].set_xlabel('Round')
		axs[1, 1].set_ylabel('Global Accuracy (%)')
		axs[1, 1].grid(True, linestyle='--', alpha=0.6)
		axs[1, 1].legend()

		# 调整布局，防止主标题和子图重叠
		plt.tight_layout()
		plt.subplots_adjust(top=0.92)

		# 6. 保存图表并清理内存
		plt.savefig(save_img_path, dpi=300)
		plt.close()
		print(f"✅ 保存成功: {save_img_path}\n")


if __name__ == "__main__":
	plot_and_save_curves()