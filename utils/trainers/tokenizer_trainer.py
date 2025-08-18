import os
import math
import torch
import logging
import torch.nn.functional as F
from torch.amp import autocast
from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class TokenizerTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = self.config["model"]["in_channels"]
        self.best_score = -math.inf

        self.commitment_loss_weight = self.config["training"].get(
            "commitment_loss_weight", 1.0
        )
        logger.info(f"Using commitment loss weight: {self.commitment_loss_weight}")

    def fit(self, num_epochs: int):
        """Common training loop with unsupervised validation"""
        end_epoch = self.start_epoch + num_epochs

        with self.train_logger:
            for epoch in range(self.start_epoch + 1, end_epoch + 1):
                self.current_epoch = epoch
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate()
                self._update_schedulers(epoch)
                self._log_metrics(train_metrics, val_metrics)
                self._save_if_best(epoch, val_metrics)
                self._save_last(epoch)
        self._vizualize()

    def train_epoch(self, epoch: int):
        self.model.train()
        total, running_loss, running_recon_loss, running_cmt_loss = 0, 0, 0, 0
        metrics = {}

        for idx, inputs in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                decoded, _, cmt_loss = self.model(inputs)
                recon_loss = self.criterion(decoded, inputs)
                loss = recon_loss + self.commitment_loss_weight * cmt_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.schedulers["warmup"] is not None and epoch <= self.warmup_epochs:
                self.schedulers["warmup"].step()

            running_loss += loss.item()
            running_recon_loss += recon_loss.item()
            running_cmt_loss += cmt_loss.item()
            total += 1

            batch_metrics = self.metric_handler.calculate_metrics(
                preds_patches=decoded.detach().float().cpu(),
                targets_patches=inputs.detach().float().cpu(),
            )

            if not metrics:
                metrics = {k: 0.0 for k in batch_metrics}

            for key, value in batch_metrics.items():
                metrics[key] += value

            self.train_logger.train_log_step(epoch, idx)

        metrics["Loss"] = running_loss / total
        return metrics

    def validate(self):
        self.model.eval()
        total, running_loss, running_recon_loss, running_cmt_loss = 0, 0, 0, 0
        metrics = {}

        with torch.no_grad():
            for idx, inputs in enumerate(self.val_loader):
                inputs = inputs.to(self.device)

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    decoded, _, cmt_loss = self.model(inputs)
                    recon_loss = self.criterion(decoded, inputs)
                    loss = recon_loss + self.commitment_loss_weight * cmt_loss

                running_loss += loss.item()
                running_recon_loss += recon_loss.item()
                running_cmt_loss += cmt_loss.item()
                total += 1

                batch_metrics = self.metric_handler.calculate_metrics(
                    preds_patches=decoded.detach().float().cpu(),
                    targets_patches=inputs.detach().float().cpu(),
                )

                if not metrics:
                    metrics = {k: 0.0 for k in batch_metrics}

                for key, value in batch_metrics.items():
                    metrics[key] += value

                self.train_logger.val_log_step(idx)

        metrics["Loss"] = running_loss / total
        return metrics

    def _save_if_best(self, epoch, val_metrics):

        if "SSIM" not in val_metrics or "PSNR" not in val_metrics:
            logger.warning(
                "SSIM or PSNR not found in validation metrics. Cannot determine best model."
            )

            score = -val_metrics.get("Loss", float("inf"))
            if score > self.best_score:
                return

        score = val_metrics["SSIM"] + 0.01 * val_metrics["PSNR"]
        if score > self.best_score:
            self.best_score = score
            logger.info(
                f"New best validation score: {self.best_score:.4f}. Saving model..."
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_score": self.best_score,
                "config": self.config,
            }
            os.makedirs(self.save_path, exist_ok=True)
            torch.save(checkpoint, os.path.join(self.save_path, "best_model.pth"))
            self.train_logger.resume()
