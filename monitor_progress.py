# monitor_progress.py

import pandas as pd
import time
import os
from datetime import datetime

def monitor_grid_progress(summary_file='logs/grid_experiment_summary.csv', refresh_interval=5):
    """Monitor progress eksperimen grid search secara real-time"""
    
    print("="*80)
    print("Grid Search Progress Monitor")
    print("="*80)
    print(f"Monitoring: {summary_file}")
    print("Press Ctrl+C to stop\n")
    
    last_modified = None
    
    try:
        while True:
            if os.path.exists(summary_file):
                current_modified = os.path.getmtime(summary_file)
                
                if current_modified != last_modified:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    df = pd.read_csv(summary_file)
                    
                    print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Total Experiments Completed: {len(df)}/60")
                    print("="*80)
                    
                    # Summary statistics
                    if len(df) > 0:
                        print("\nBest Results So Far:")
                        best_acc = df.loc[df['accuracy'].idxmax()]
                        print(f"  Best Accuracy: {best_acc['accuracy']:.4f}")
                        print(f"  - Enhancement: {best_acc['enhancement']}")
                        print(f"  - Parameters: {best_acc['param']}")
                        print(f"  - Model: {best_acc['model']}")
                        
                        print("\nLatest 5 Experiments:")
                        print(df[['enhancement', 'param', 'model', 'accuracy', 'f1']].tail())
                        
                        print("\nEnhancement Performance (Average Accuracy):")
                        enhancement_stats = df.groupby('enhancement')['accuracy'].agg(['mean', 'std', 'count'])
                        print(enhancement_stats)
                    
                    last_modified = current_modified
            else:
                print("Waiting for experiments to start...")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    monitor_grid_progress()