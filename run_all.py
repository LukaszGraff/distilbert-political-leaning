import subprocess
import sys

# List of scripts to run in sequence
scripts = [
    "scripts/preprocessing/preprocess.py",
    "scripts/training/train_tfidf.py",
    "scripts/training/train_distilbert.py",
    "scripts/evaluation/running/run_models.py",
    "scripts/evaluation/quantitative/eval.py",
    "scripts/evaluation/qualitative/eval.py",
]

def run_scripts(scripts):
    for script in scripts:
        print(f"RUNNING {script}...")
        try:
            result = subprocess.run([sys.executable, script], check=True)
            print(f"SUCCESSFULLY EXECUTED {script}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR WHILE RUNNING {script}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    run_scripts(scripts)