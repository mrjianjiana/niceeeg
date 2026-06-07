"""Create 200 test class-name files from THINGS-EEG2 image_metadata.npy."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def clean_concept(concept):
    # Example: "00001_aircraft_carrier" -> "aircraft carrier"
    name = re.sub(r"^\d+_", "", concept)
    return name.replace("_", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        default=r"D:\Ascienceproject\data\things-eeg2\image_metadata.npy",
    )
    parser.add_argument("--out_txt", default="class_names_200.txt")
    parser.add_argument("--out_csv", default="class_names_200.csv")
    args = parser.parse_args()

    metadata = np.load(args.metadata, allow_pickle=True).item()
    concepts = metadata["test_img_concepts"]
    files = metadata["test_img_files"]

    if len(concepts) != 200:
        raise ValueError(f"Expected 200 test concepts, got {len(concepts)}")

    out_txt = Path(args.out_txt)
    out_csv = Path(args.out_csv)

    names = [clean_concept(concept) for concept in concepts]
    out_txt.write_text("\n".join(names) + "\n", encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "class_name", "raw_concept", "test_img_file"])
        for label, (name, concept, img_file) in enumerate(zip(names, concepts, files)):
            writer.writerow([label, name, concept, img_file])

    print(f"Wrote {out_txt.resolve()}")
    print(f"Wrote {out_csv.resolve()}")


if __name__ == "__main__":
    main()

