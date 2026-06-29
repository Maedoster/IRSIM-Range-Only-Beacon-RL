import os
import random
import shutil

# --- CONFIGURATION ---
SEED = 42  # Keep this the exact same to always get the same split
NUM_EVAL = 500
NUM_TEST = 3000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "IRSimDataset")
EVAL_DIR = os.path.join(BASE_DIR, "EvalDataset")
TEST_DIR = os.path.join(BASE_DIR, "TestDataset")
SEED_FILE = os.path.join(BASE_DIR, "seed_split.txt")

def main():
    #Create target directories if they don't exist
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    #Get all yaml files from the source directory
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".yaml")]

    #Sorting for reproducibility
    files.sort()

    total_files = len(files)
    print(f"Found {total_files} files in '{SOURCE_DIR}'.")

    if total_files < (NUM_EVAL + NUM_TEST):
        print(f"Error: Not enough files to split! Need {NUM_EVAL + NUM_TEST}, but only found {total_files}.")
        return

    #Apply the random seed and shuffle the list
    print(f"Applying random seed: {SEED}")
    random.seed(SEED)
    random.shuffle(files)

    #Slice the list into our sets
    eval_files = files[:NUM_EVAL]
    test_files = files[NUM_EVAL : NUM_EVAL + NUM_TEST]

    #Move the files to EvalDataset
    print(f"\nMoving {NUM_EVAL} files to '{EVAL_DIR}'...")
    for f in eval_files:
        src_path = os.path.join(SOURCE_DIR, f)
        dst_path = os.path.join(EVAL_DIR, f)
        shutil.move(src_path, dst_path)

    #Move the files to TestDataset
    print(f"Moving {NUM_TEST} files to '{TEST_DIR}'...")
    for f in test_files:
        src_path = os.path.join(SOURCE_DIR, f)
        dst_path = os.path.join(TEST_DIR, f)
        shutil.move(src_path, dst_path)

    #Save the seed to a file for future reference
    with open(SEED_FILE, "w") as f:
        f.write(str(SEED))

    print("\n✅ Dataset split complete!")
    print(f"  - EvalDataset: {len(eval_files)} files")
    print(f"  - TestDataset: {len(test_files)} files")
    print(f"  - IRSimDataset (Remaining for Training): {total_files - NUM_EVAL - NUM_TEST} files")
    print(f"  - Seed saved to: {SEED_FILE}")

if __name__ == "__main__":
    main()