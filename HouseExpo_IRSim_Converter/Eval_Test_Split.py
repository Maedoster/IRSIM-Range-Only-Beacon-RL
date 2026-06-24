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

def main():
    # 1. Create target directories if they don't exist
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # 2. Get all yaml files from the source directory
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' does not exist.")
        return

    files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".yaml")]

    # CRITICAL: Sort the files first!
    # os.listdir() is non-deterministic. Sorting guarantees that the list is in the 
    # exact same order before the random seed is applied, ensuring 100% reproducibility.
    files.sort()

    total_files = len(files)
    print(f"Found {total_files} files in '{SOURCE_DIR}'.")

    if total_files < (NUM_EVAL + NUM_TEST):
        print(f"Error: Not enough files to split! Need {NUM_EVAL + NUM_TEST}, but only found {total_files}.")
        return

    # 3. Apply the random seed and shuffle the list
    print(f"Applying random seed: {SEED}")
    random.seed(SEED)
    random.shuffle(files)

    # 4. Slice the list into our sets
    eval_files = files[:NUM_EVAL]
    test_files = files[NUM_EVAL : NUM_EVAL + NUM_TEST]

    # 5. Move the files to EvalDataset
    print(f"\nMoving {NUM_EVAL} files to '{EVAL_DIR}'...")
    for f in eval_files:
        src_path = os.path.join(SOURCE_DIR, f)
        dst_path = os.path.join(EVAL_DIR, f)
        shutil.move(src_path, dst_path)

    # 6. Move the files to TestDataset
    print(f"Moving {NUM_TEST} files to '{TEST_DIR}'...")
    for f in test_files:
        src_path = os.path.join(SOURCE_DIR, f)
        dst_path = os.path.join(TEST_DIR, f)
        shutil.move(src_path, dst_path)

    print("\n✅ Dataset split complete!")
    print(f"  - EvalDataset: {len(eval_files)} files")
    print(f"  - TestDataset: {len(test_files)} files")
    print(f"  - IRSimDataset (Remaining for Training): {total_files - NUM_EVAL - NUM_TEST} files")

if __name__ == "__main__":
    main()