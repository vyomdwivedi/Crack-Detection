import torch

# Load the model
model = torch.load('Models/Copy of FPUNet_best_IoU_67.pth', map_location='cpu')

print("Model keys (first 20):")
for i, k in enumerate(list(model.keys())[:20]):
    print(f"{i+1}: {k}")

print(f"\nTotal number of keys: {len(model.keys())}")

# Check if it's a fusion model with multiple architectures
unet_keys = [k for k in model.keys() if k.startswith('unet.')]
fpn_keys = [k for k in model.keys() if k.startswith('fpn.')]
fusion_keys = [k for k in model.keys() if k.startswith('fusion.')]

print(f"\nUNet keys: {len(unet_keys)}")
print(f"FPN keys: {len(fpn_keys)}")
print(f"Fusion keys: {len(fusion_keys)}")

if unet_keys:
    print("\nFirst 5 UNet keys:")
    for k in unet_keys[:5]:
        print(k)

if fpn_keys:
    print("\nFirst 5 FPN keys:")
    for k in fpn_keys[:5]:
        print(k)

if fusion_keys:
    print("\nFusion keys:")
    for k in fusion_keys:
        print(k)
