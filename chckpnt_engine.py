import threading
import time
import os
import torch
from datetime import datetime
import json
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class UniversalCheckpointManager:
    """Enhanced checkpoint manager with time-based features and snapshots"""
    
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')

    def save_checkpoint(self, epoch, model, optimizer, batch_idx, model_name,
                       loss, accuracy, elapsed_time, is_timeout=False):
        """Save checkpoint with exact training state"""
        try:
            checkpoint = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'model_name': model_name,
                'loss': float(loss) if loss is not None else None,
                'accuracy': float(accuracy) if accuracy is not None else None,
                'elapsed_time': elapsed_time,
                'is_timeout': is_timeout,
                'timestamp': datetime.now().isoformat()
            }

            # Save metadata as JSON
            checkpoint_file = os.path.join(self.checkpoint_dir, f'time_checkpoint_{model_name}.json')
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            # Save model weights separately (binary)
            model_file = os.path.join(self.checkpoint_dir, f'time_model_{model_name}.pth')
            torch.save(model.state_dict(), model_file)

            # Save optimizer state separately (binary)
            optimizer_file = os.path.join(self.checkpoint_dir, f'time_optimizer_{model_name}.pth')
            torch.save(optimizer.state_dict(), optimizer_file)

            status = "TIMEOUT" if is_timeout else "PROGRESS"
            print(f"[HOOK] ✓ {status} checkpoint saved: {model_name} at batch {batch_idx}")
            return True

        except Exception as e:
            print(f"[HOOK] ✗ Error saving checkpoint: {e}")
            return False

    def load_checkpoint(self, model, optimizer, model_name):
        """Load the latest checkpoint for a model"""
        try:
            checkpoint_file = os.path.join(self.checkpoint_dir, f'time_checkpoint_{model_name}.json')
            model_file = os.path.join(self.checkpoint_dir, f'time_model_{model_name}.pth')
            optimizer_file = os.path.join(self.checkpoint_dir, f'time_optimizer_{model_name}.pth')

            if not os.path.exists(checkpoint_file):
                return None

            # Load checkpoint info
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)

            # Load model weights
            if os.path.exists(model_file):
                model.load_state_dict(torch.load(model_file))

            # Load optimizer state
            if os.path.exists(optimizer_file):
                optimizer.load_state_dict(torch.load(optimizer_file))

            print(f"[HOOK] ✓ Loaded checkpoint: Epoch {checkpoint_data['epoch']}, "
                  f"Batch {checkpoint_data['batch_idx']}, {checkpoint_data['model_name']}")
            return checkpoint_data

        except Exception as e:
            print(f"[HOOK] ✗ Error loading checkpoint: {e}")
            return None

    def save_snapshot(self, images, outputs, labels, model_name, batch_idx, elapsed_time):
        """Save visual snapshot of current progress"""
        try:
            if len(images) == 0:
                return False

            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.ravel()

            def denormalize(img):
                img = img.cpu().numpy().transpose(1, 2, 0)
                img = (img * 0.5) + 0.5
                return np.clip(img, 0, 1)

            batch_correct = 0
            batch_total = min(8, len(images))

            for i in range(8):
                if i < len(images):
                    img = denormalize(images[i])
                    pred_class = torch.argmax(outputs[i]).item()
                    actual_class = labels[i].item()

                    axes[i].imshow(img)

                    pred_name = self.classes[pred_class]
                    actual_name = self.classes[actual_class]

                    color = 'green' if pred_class == actual_class else 'red'
                    axes[i].set_title(f'Pred: {pred_name}\nActual: {actual_name}',
                                    color=color, fontsize=9)
                    axes[i].axis('off')

                    if pred_class == actual_class:
                        batch_correct += 1
                else:
                    axes[i].axis('off')

            batch_accuracy = 100.0 * batch_correct / batch_total

            minutes = elapsed_time // 60
            seconds = elapsed_time % 60

            plt.suptitle(f'{model_name} - Batch {batch_idx}\n'
                        f'Elapsed: {int(minutes)}m {int(seconds)}s | '
                        f'Batch Accuracy: {batch_accuracy:.1f}%',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)

            snapshot_file = os.path.join(self.checkpoint_dir, f'time_snapshot_{model_name}.png')
            plt.savefig(snapshot_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[HOOK] ✓ Snapshot saved: {snapshot_file}")
            return True

        except Exception as e:
            print(f"[HOOK] ✗ Error saving snapshot: {e}")
            return False

    def save_session_info(self, current_model_index, start_time, total_elapsed):
        """Save session information"""
        session_info = {
            'current_model_index': current_model_index,
            'session_start_time': start_time,
            'total_elapsed_time': total_elapsed,
            'last_save_time': datetime.now().isoformat()
        }

        session_file = os.path.join(self.checkpoint_dir, 'session_info.json')
        with open(session_file, 'w') as f:
            json.dump(session_info, f, indent=2)

    def load_session_info(self):
        """Load session information"""
        try:
            session_file = os.path.join(self.checkpoint_dir, 'session_info.json')
            with open(session_file, 'r') as f:
                return json.load(f)
        except:
            return None

    def get_time_until_checkpoint(self, last_checkpoint_time, checkpoint_interval):
        """Calculate time until next checkpoint"""
        current_time = time.time()
        time_since_last = current_time - last_checkpoint_time
        return max(0, checkpoint_interval - time_since_last)


class StateTracker:
    """Enhanced state tracker with session management"""
    def __init__(self):
        self.epoch = 0
        self.batch_idx = 0
        self.running_loss = 0.0
        self.running_accuracy = 0.0
        self.session_start_time = time.time()
        self.total_elapsed_time = 0
        self.snapshot_images = []
        self.snapshot_outputs = []
        self.snapshot_labels = []
        self.lock = threading.Lock()

    def update_state(self, epoch, batch_idx, running_loss, running_accuracy):
        with self.lock:
            self.epoch = epoch
            self.batch_idx = batch_idx
            self.running_loss = running_loss
            self.running_accuracy = running_accuracy

    def get_state(self):
        with self.lock:
            return (self.epoch, self.batch_idx, self.running_loss, 
                   self.running_accuracy, self.session_start_time, 
                   self.total_elapsed_time)

    def update_session_time(self, total_elapsed):
        with self.lock:
            self.total_elapsed_time = total_elapsed

    def add_snapshot_data(self, image, output, label):
        with self.lock:
            if len(self.snapshot_images) < 8:
                self.snapshot_images.append(image)
                self.snapshot_outputs.append(output)
                self.snapshot_labels.append(label)

    def get_snapshot_data(self):
        with self.lock:
            return (self.snapshot_images.copy(), self.snapshot_outputs.copy(), 
                   self.snapshot_labels.copy())


class CheckpointTimer(threading.Thread):
    """Enhanced timer with session time limits and snapshot capabilities"""
    
    def __init__(self, interval, kind, save_dir, models_optimizers, manager, max_session_time=900):
        super().__init__(daemon=True)
        self.interval = interval
        self.kind = kind
        self.save_dir = save_dir
        self.models_optimizers = models_optimizers  # {'name': (model, optimizer, tracker)}
        self.manager = manager
        self.max_session_time = max_session_time
        self.is_running = True
        self.session_start_time = time.time()
        os.makedirs(save_dir, exist_ok=True)

    def run(self):
        print(f"[HOOK] 🤖 Enhanced checkpoint timer started. Interval: {self.interval}s, Max session: {self.max_session_time}s")
        
        # Load session info
        session_info = self.manager.load_session_info()
        if session_info:
            total_elapsed = session_info['total_elapsed_time']
            print(f"[HOOK] ✓ Resumed session with total time: {total_elapsed/60:.1f} minutes")
        else:
            total_elapsed = 0
            print("[HOOK] ✓ Starting new training session")

        while self.is_running:
            time.sleep(1)  # Check every second for better time monitoring
            
            current_time = time.time()
            current_session_time = current_time - self.session_start_time
            total_elapsed_so_far = total_elapsed + current_session_time
            time_remaining = self.max_session_time - current_session_time

            # Update tracker with current total elapsed time
            for _, (_, _, tracker) in self.models_optimizers.items():
                tracker.update_session_time(total_elapsed_so_far)

            # Check for regular checkpoint interval
            if int(current_time) % self.interval == 0:
                print(f"\n[HOOK] ⏰ Regular checkpoint triggered...")
                self._save_progress_checkpoint(total_elapsed_so_far, False)

            # Check for session timeout (with 15-second buffer)
            if time_remaining <= 15 and time_remaining > 0:
                print(f"\n[HOOK] 🚨 TIME LIMIT APPROACHING: {time_remaining:.1f}s remaining")
                self._save_progress_checkpoint(total_elapsed_so_far, True)
                
                # Save session info
                current_model_index = 0  # You might want to track this per model
                self.manager.save_session_info(current_model_index, 
                                             self.session_start_time, 
                                             total_elapsed_so_far)
                
                print(f"[HOOK] 💾 Final checkpoint saved. Session completed.")
                self.is_running = False
                break

    def _save_progress_checkpoint(self, total_elapsed, is_timeout=False):
        """Save checkpoint for all models"""
        for model_name, (model, optimizer, tracker) in self.models_optimizers.items():
            epoch, batch_idx, running_loss, running_accuracy, _, _ = tracker.get_state()
            
            self.manager.save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                batch_idx=batch_idx,
                model_name=model_name,
                loss=running_loss,
                accuracy=running_accuracy,
                elapsed_time=total_elapsed,
                is_timeout=is_timeout
            )
            
            # Save snapshot if we have data
            snapshot_images, snapshot_outputs, snapshot_labels = tracker.get_snapshot_data()
            if snapshot_images:
                self.manager.save_snapshot(
                    snapshot_images, 
                    torch.stack(snapshot_outputs) if snapshot_outputs else [],
                    torch.stack(snapshot_labels) if snapshot_labels else [],
                    model_name,
                    batch_idx,
                    total_elapsed
                )

    def stop(self):
        self.is_running = False
