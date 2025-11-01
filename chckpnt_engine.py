import threading
import time
import os
import torch
from datetime import datetime
import json
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import yaml


class UniversalCheckpointManager:
    """Complete checkpoint system - handles all save/load operations"""
    
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck')

    def save_checkpoint(self, epoch, model, optimizer, batch_idx, model_name,
                       loss, accuracy, elapsed_time, is_timeout=False):
        """Save complete training state"""
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

            # Save metadata
            checkpoint_file = os.path.join(self.checkpoint_dir, f'checkpoint_{model_name}.json')
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            # Save model weights
            model_file = os.path.join(self.checkpoint_dir, f'model_{model_name}.pth')
            torch.save(model.state_dict(), model_file)

            # Save optimizer state
            optimizer_file = os.path.join(self.checkpoint_dir, f'optimizer_{model_name}.pth')
            torch.save(optimizer.state_dict(), optimizer_file)

            status = "TIMEOUT" if is_timeout else "PROGRESS"
            print(f"✓ {status} checkpoint saved: {model_name} at batch {batch_idx}")
            return True

        except Exception as e:
            print(f"✗ Error saving checkpoint: {e}")
            return False

    def load_checkpoint(self, model, optimizer, model_name):
        """Load complete training state"""
        try:
            checkpoint_file = os.path.join(self.checkpoint_dir, f'checkpoint_{model_name}.json')
            model_file = os.path.join(self.checkpoint_dir, f'model_{model_name}.pth')
            optimizer_file = os.path.join(self.checkpoint_dir, f'optimizer_{model_name}.pth')

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

            print(f"✓ Loaded checkpoint: Epoch {checkpoint_data['epoch']}, "
                  f"Batch {checkpoint_data['batch_idx']}")
            return checkpoint_data

        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            return None

    def save_snapshot(self, images, outputs, labels, model_name, batch_idx, elapsed_time):
        """Save visual progress snapshot"""
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

            snapshot_file = os.path.join(self.checkpoint_dir, f'snapshot_{model_name}.png')
            plt.savefig(snapshot_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Snapshot saved: {snapshot_file}")
            return True

        except Exception as e:
            print(f"✗ Error saving snapshot: {e}")
            return False

    def save_session_info(self, current_model_index, start_time, total_elapsed):
        """Save session management info"""
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
        """Load session management info"""
        try:
            session_file = os.path.join(self.checkpoint_dir, 'session_info.json')
            with open(session_file, 'r') as f:
                return json.load(f)
        except:
            return None


class StateTracker:
    """Tracks and shares training state between main thread and checkpoint system"""
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
        self.timers = []

    def update_state(self, epoch, batch_idx, running_loss, running_accuracy):
        """Update current training state (called from main training loop)"""
        with self.lock:
            self.epoch = epoch
            self.batch_idx = batch_idx
            self.running_loss = running_loss
            self.running_accuracy = running_accuracy

    def get_state(self):
        """Get current training state (called by checkpoint system)"""
        with self.lock:
            return (self.epoch, self.batch_idx, self.running_loss, 
                   self.running_accuracy, self.session_start_time, 
                   self.total_elapsed_time)

    def update_session_time(self, total_elapsed):
        """Update total training time"""
        with self.lock:
            self.total_elapsed_time = total_elapsed

    def add_snapshot_data(self, image, output, label):
        """Collect data for visual snapshots"""
        with self.lock:
            if len(self.snapshot_images) < 8:
                self.snapshot_images.append(image)
                self.snapshot_outputs.append(output)
                self.snapshot_labels.append(label)

    def get_snapshot_data(self):
        """Get collected snapshot data"""
        with self.lock:
            return (self.snapshot_images.copy(), self.snapshot_outputs.copy(), 
                   self.snapshot_labels.copy())


class AutonomousCheckpointer:
    """Autonomous checkpointing system - runs in background thread"""
    
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.manager = UniversalCheckpointManager(
            checkpoint_dir=self.config['checkpoint']['save_dir']
        )
        self.tracker = StateTracker()
        self.is_running = True
        self.models_optimizers = {}
        
        # Store timer reference in tracker
        self.tracker.timers.append(self)

    def register_models(self, models_dict):
        """Register models and optimizers for checkpointing"""
        self.models_optimizers = {
            name: (model, optimizer, self.tracker)
            for name, (model, optimizer) in models_dict.items()
        }

    def start(self):
        """Start the autonomous checkpointing system"""
        if not self.models_optimizers:
            raise ValueError("No models registered. Call register_models() first.")
            
        print("🚀 Starting Autonomous Checkpointing System...")
        print(f"   Checkpoint interval: {self.config['checkpoint']['checkpoint_interval']}s")
        print(f"   Max session time: {self.config['checkpoint']['max_session_time']}s")
        print(f"   Save directory: {self.config['checkpoint']['save_dir']}")
        
        # Start background thread
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        return self.tracker

    def _run(self):
        """Main checkpointing loop (runs in background thread)"""
        session_info = self.manager.load_session_info()
        if session_info:
            total_elapsed = session_info['total_elapsed_time']
            print(f"✓ Resumed session - Total time: {total_elapsed/60:.1f} minutes")
        else:
            total_elapsed = 0
            print("✓ Starting new training session")

        session_start = time.time()
        last_checkpoint_time = time.time()
        
        while self.is_running:
            time.sleep(1)  # Check every second
            
            current_time = time.time()
            session_duration = current_time - session_start
            total_elapsed_so_far = total_elapsed + session_duration
            time_remaining = self.config['checkpoint']['max_session_time'] - session_duration

            # Update tracker with current time
            self.tracker.update_session_time(total_elapsed_so_far)

            # Regular checkpoint interval
            if current_time - last_checkpoint_time >= self.config['checkpoint']['checkpoint_interval']:
                self._save_checkpoints(total_elapsed_so_far, False)
                last_checkpoint_time = current_time

            # Session time limit
            if (self.config['checkpoint']['max_session_time'] > 0 and 
                time_remaining <= 15 and time_remaining > 0):
                print(f"⏰ Time limit approaching: {time_remaining:.1f}s remaining")
                self._save_checkpoints(total_elapsed_so_far, True)
                self.manager.save_session_info(0, session_start, total_elapsed_so_far)
                self.is_running = False
                break

    def _save_checkpoints(self, total_elapsed, is_timeout):
        """Save checkpoints for all models"""
        for model_name, (model, optimizer, _) in self.models_optimizers.items():
            epoch, batch_idx, running_loss, running_accuracy, _, _ = self.tracker.get_state()
            
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
            
            # Save snapshot if enabled
            if self.config['checkpoint']['enable_snapshots']:
                snapshot_images, snapshot_outputs, snapshot_labels = self.tracker.get_snapshot_data()
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
        """Stop the checkpointing system"""
        self.is_running = False

    def get_tracker(self):
        """Get the state tracker for training loop"""
        return self.tracker

    def get_config(self):
        """Get the configuration"""
        return self.config


# Global instance for easy access
_checkpointer = None

def enable_checkpointing(config_path="config.yaml"):
    """
    Enable autonomous checkpointing for the training session
    Usage:
        checkpointer = enable_checkpointing("config.yaml")
        checkpointer.register_models({'model1': (model, optimizer)})
        tracker = checkpointer.start()
    """
    global _checkpointer
    _checkpointer = AutonomousCheckpointer(config_path)
    return _checkpointer

def get_checkpointer():
    """Get the global checkpointer instance"""
    return _checkpointer

def update_training_state(epoch, batch_idx, loss, accuracy):
    """
    Update training state - call this from your training loop
    Usage in training loop:
        update_training_state(epoch, batch_idx, loss.item(), accuracy)
    """
    global _checkpointer
    if _checkpointer and _checkpointer.tracker:
        _checkpointer.tracker.update_state(epoch, batch_idx, loss, accuracy)

def add_snapshot_data(image, output, label):
    """
    Add snapshot data - call this from your training loop
    Usage in training loop:
        add_snapshot_data(inputs[0], outputs[0], labels[0])
    """
    global _checkpointer
    if _checkpointer and _checkpointer.tracker:
        _checkpointer.tracker.add_snapshot_data(image, output, label)

def stop_checkpointing():
    """Stop the checkpointing system"""
    global _checkpointer
    if _checkpointer:
        _checkpointer.stop()
