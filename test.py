import torch

pdf_data = torch.load("semantic_masks/semantic_mask_pdf.pt")

print(type(pdf_data))
if isinstance(pdf_data, dict):
    print("✅ It's a dict. Keys:", pdf_data.keys())
elif isinstance(pdf_data, (tuple, list)):
    print(f"✅ It's a {type(pdf_data).__name__} of length {len(pdf_data)}")
    for i, item in enumerate(pdf_data):
        print(f"[{i}] Type: {type(item)}, Shape: {getattr(item, 'shape', None)}")
else:
    print("❌ Unknown type:", type(pdf_data))
