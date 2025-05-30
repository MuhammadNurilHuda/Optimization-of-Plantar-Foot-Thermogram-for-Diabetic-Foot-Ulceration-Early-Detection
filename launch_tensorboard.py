# launch_tensorboard.py

import subprocess
import os
import webbrowser
import time

def launch_tensorboard(logdir='logs/tfboard', port=6006):
    """Launch TensorBoard in browser"""
    
    print(f"Launching TensorBoard on port {port}...")
    print(f"Log directory: {logdir}")
    
    # Start TensorBoard
    tb_process = subprocess.Popen(
        ['tensorboard', '--logdir', logdir, '--port', str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait a bit for TensorBoard to start
    time.sleep(3)
    
    # Open browser
    url = f'http://localhost:{port}'
    print(f"Opening {url} in browser...")
    webbrowser.open(url)
    
    print("\nTensorBoard is running. Press Ctrl+C to stop.")
    
    try:
        tb_process.wait()
    except KeyboardInterrupt:
        print("\nStopping TensorBoard...")
        tb_process.terminate()

if __name__ == "__main__":
    launch_tensorboard()