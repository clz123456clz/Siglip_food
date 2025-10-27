from datasets import load_dataset

# Load the dataset from Hugging Face
hf_dataset_name = "jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth"
print(f"Loading dataset: {hf_dataset_name}...")

# Load the desired split (e.g., 'train')
# Use streaming=True for large datasets to avoid downloading everything at once
# dataset = load_dataset(hf_dataset_name, split="train", streaming=True)
dataset = load_dataset(hf_dataset_name, split="train") # Loads the entire split into memory/cache

print("Dataset loaded successfully.")

# Accessing data (example for non-streaming)
if dataset:
    print(f"Number of examples in split: {len(dataset)}")
    # Get the first example
    first_example = dataset[0]
    img = first_example['image']
    caption = first_example['caption']
    print("\nFirst example:")
    print(f" Caption: {caption}")
    print(f" Image type: {type(img)}")
    print(f" Image mode: {img.mode}")
    print(f" Image size: {img.size}")

# You can display the image (requires matplotlib or other library)
# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.title(caption)
# plt.axis('off')
# plt.show()

# Iterate through the dataset (example)
# print("\nIterating through first 5 examples:")
# for i, example in enumerate(dataset):
#     if i &gt;= 5:
#         break
#     print(f" Example {i+1}: Caption - {example['caption'][:50]}...") # Print first 50 chars of caption
    # Process example['image'] and example['caption'] as needed
else:
    print("Failed to load dataset.")

# Example of iterating with streaming=True
# print("\nIterating through first 5 examples (streaming):")
# count = 0
# for example in dataset:
#     if count >= 5:
#         break
#     print(f" Example {count+1}: Caption - {example['caption'][:50]}...")
#     # Process example['image'] and example['caption']
#     count += 1
