import os
import subprocess

def download_kaggle_dataset():
    dataset = "paultimothymooney/stock-market-data"
    command = f"kaggle datasets download -d {dataset}"
    
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"Successfully downloaded dataset: {dataset}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    download_kaggle_dataset()
