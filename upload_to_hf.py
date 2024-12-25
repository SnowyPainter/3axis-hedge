from huggingface_hub import upload_file

with open('./hftoken.txt', 'r') as file:
    HF_TOKEN = file.readline().strip()
REPO_ID = "SnowyPainter/AlgoShield"
REPO_TYPE = "model"
LOCAL_FILES = [
    "gap/gap-d1.h5",
    "gap/meta_gap_random_forest.joblib",
]
for local_file in LOCAL_FILES:
    try:
        upload_file(
            path_or_fileobj=local_file,
            path_in_repo=local_file,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=HF_TOKEN,
        )
        print(f"Uploaded {local_file} successfully!")
    except Exception as e:
        print(f"Failed to upload {local_file}: {e}")