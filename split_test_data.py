"""
Tach 3 anh ngau nhien moi landmark tu data/images/ sang data/tests/
Phan con lai giu nguyen trong data/images/ de embedding bang build_vector_index.py
"""

import random
import shutil
from pathlib import Path

SEED = 42
TEST_PER_LANDMARK = 3
IMAGES_DIR = Path("data/images")
TESTS_DIR = Path("data/tests")


def split_test_data():
    random.seed(SEED)

    if not IMAGES_DIR.exists():
        print(f"Khong tim thay thu muc: {IMAGES_DIR}")
        return

    landmark_dirs = sorted([d for d in IMAGES_DIR.iterdir() if d.is_dir()])
    print(f"Tong landmarks: {len(landmark_dirs)}")

    total_moved = 0
    total_remain = 0
    skipped = []

    for lm_dir in landmark_dirs:
        lm_name = lm_dir.name
        img_files = sorted(
            list(lm_dir.glob("*.jpg")) + list(lm_dir.glob("*.png"))
        )

        if len(img_files) < TEST_PER_LANDMARK + 1:
            print(f"  [SKIP] {lm_name}: chi co {len(img_files)} anh, can >= {TEST_PER_LANDMARK + 1}")
            skipped.append(lm_name)
            total_remain += len(img_files)
            continue

        # Chon ngau nhien 3 anh lam test
        test_files = random.sample(img_files, TEST_PER_LANDMARK)

        # Tao thu muc test
        test_lm_dir = TESTS_DIR / lm_name
        test_lm_dir.mkdir(parents=True, exist_ok=True)

        # Di chuyen anh test
        for f in test_files:
            dest = test_lm_dir / f.name
            shutil.move(str(f), str(dest))

        remain = len(img_files) - TEST_PER_LANDMARK
        total_moved += TEST_PER_LANDMARK
        total_remain += remain
        print(f"  {lm_name}: {TEST_PER_LANDMARK} -> tests/, {remain} con lai")

    print(f"\n{'='*50}")
    print(f"Da di chuyen: {total_moved} anh sang data/tests/")
    print(f"Con lai train: {total_remain} anh trong data/images/")
    if skipped:
        print(f"Landmarks bi skip: {skipped}")
    print(f"\nBuoc tiep: python -m modules.retrieval.build_vector_index")


if __name__ == "__main__":
    split_test_data()
