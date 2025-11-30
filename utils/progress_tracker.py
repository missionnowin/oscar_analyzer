import sys
import threading
import time


class ProgressTracker:
    """Thread-safe progress tracking for parallel runs."""
    
    def __init__(self):
        self.progress = {}
        self.lock = threading.Lock()
    
    def update(self, run_name: str, stage: str, details: str = ""):
        """Update progress for a specific run."""
        with self.lock:
            self.progress[run_name] = {
                'stage': stage,
                'details': details,
                'timestamp': time.time()
            }
    
    def print_all(self):
        """Print all active runs (non-blocking)."""
        with self.lock:
            for run_name in sorted(self.progress.keys()):
                info = self.progress[run_name]
                stage = info['stage']
                details = info['details']
                
                # Build progress line
                if details:
                    line = f"  [{run_name}] {stage:30s} {details}"
                else:
                    line = f"  [{run_name}] {stage:30s}"
                
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
    
    def clear_and_print(self):
        """Clear terminal and print all progress."""
        with self.lock:
            # Move cursor to home
            sys.stdout.write("\033[H\033[J")  # Clear screen
            for run_name in sorted(self.progress.keys()):
                info = self.progress[run_name]
                stage = info['stage']
                details = info['details']
                
                if details:
                    line = f"  [{run_name}] {stage:30s} {details}\n"
                else:
                    line = f"  [{run_name}] {stage:30s}\n"
                
                sys.stdout.write(line)
            sys.stdout.flush()