import os
import torch

class CheckpointManager:
    def __init__(self, save_dir="checkpoints", max_keep=5):
        self.save_dir = save_dir
        self.max_keep = max_keep
        os.makedirs(save_dir, exist_ok=True)
        self.checkpoints = []  # 存储 (score, path)

    def save(self, model, score, epoch):
        """
        保存模型并根据score判断是否进入前max_keep
        :param model: 要保存的模型
        :param score: 当前模型的评估指标（越大越好）
        :param epoch: 当前训练的epoch，方便区分文件
        """
        filename = os.path.join(self.save_dir, f"model_epoch{epoch}_score{score:.4f}.pth")

        # 如果还没满，直接存
        if len(self.checkpoints) < self.max_keep:
            torch.save(model.state_dict(), filename)
            self.checkpoints.append((score, filename))
            self.checkpoints.sort(key=lambda x: x[0], reverse=True)  # 按score排序
        else:
            # 如果比最差的还好，替换
            worst_score, worst_file = self.checkpoints[-1]
            if score > worst_score:
                torch.save(model.state_dict(), filename)
                self.checkpoints.append((score, filename))
                self.checkpoints.sort(key=lambda x: x[0], reverse=True)

                # 删除多余的第6个
                if len(self.checkpoints) > self.max_keep:
                    _, remove_file = self.checkpoints.pop(-1)
                    if os.path.exists(remove_file):
                        os.remove(remove_file)
            else:
                print(f"Epoch {epoch}: 当前模型(score={score:.4f})不优于已保存的权重，未保存。")

    def get_best_model_path(self):
        """获取当前最佳模型路径"""
        if not self.checkpoints:
            return None
        return self.checkpoints[0][1]
