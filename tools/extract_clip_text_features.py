"""Extract CLIP text features for THINGS-EEG2 class/concept names.

Outputs:
- text_clip_feature_maps_training.npy: one text feature per training sample
- text_clip_feature_maps_test.npy: one text feature per 200 test class

These files are used by nice_text_semantic.py for EEG-image-text training.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer


def clean_concept(concept):
    name = re.sub(r"^\d+_", "", concept)
    return name.replace("_", " ")


def encode_texts(model, tokenizer, texts, device, batch_size):
    features = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            features.append(text_features.detach().cpu().numpy())
    return np.concatenate(features, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        default=r"D:\Ascienceproject\data\things-eeg2\image_metadata.npy",
    )
    parser.add_argument("--output_dir", default="./dnn_feature")
    parser.add_argument("--model_name", default="openai/clip-vit-large-patch14")
    parser.add_argument("--prompt_template", default="a photo of a {}")
    parser.add_argument("--batch_size", default=128, type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = np.load(args.metadata, allow_pickle=True).item()
    train_concepts = metadata["train_img_concepts"]
    test_concepts = metadata["test_img_concepts"]

    train_texts = [args.prompt_template.format(clean_concept(c)) for c in train_concepts]
    test_texts = [args.prompt_template.format(clean_concept(c)) for c in test_concepts]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_name} on {device}")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_name)
    model = CLIPModel.from_pretrained(args.model_name).to(device)

    print(f"Encoding {len(train_texts)} training text prompts")
    train_features = encode_texts(model, tokenizer, train_texts, device, args.batch_size)
    print(f"Encoding {len(test_texts)} test text prompts")
    test_features = encode_texts(model, tokenizer, test_texts, device, args.batch_size)

    train_path = output_dir / "text_clip_feature_maps_training.npy"
    test_path = output_dir / "text_clip_feature_maps_test.npy"
    np.save(train_path, train_features)
    np.save(test_path, test_features)

    print(f"Wrote {train_path.resolve()} {train_features.shape}")
    print(f"Wrote {test_path.resolve()} {test_features.shape}")


if __name__ == "__main__":
    main()

