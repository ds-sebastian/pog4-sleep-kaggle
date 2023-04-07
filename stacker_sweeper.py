import multiprocessing
import os

def run_script(script):
    os.system("python " + script)

if __name__ == '__main__':
    scripts = ['knn_sweep.py', 'extratree_sweep.py', 'rf_sweep.py', 'xgb_sweep.py']
    with multiprocessing.Pool() as pool:
        pool.map(run_script, scripts)