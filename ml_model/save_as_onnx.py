import torch
from collections import OrderedDict
import timm

# paths (update if you want a different .pth model)
MODEL_PATH = "models/pretrained_model.pth"
ONNX_PATH = "models/pretrained_model.onnx"
NUM_CLASSES = 38  # matches your classes in app.py

device = torch.device("cpu")

# build same architecture
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=NUM_CLASSES)

# load state dict
state = torch.load(MODEL_PATH, map_location=device)
new_state = OrderedDict()
for k, v in state.items():
    name = k[7:] if k.startswith("module.") else k
    new_state[name] = v
model.load_state_dict(new_state)
model.eval()

# dummy input
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

# export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    opset_version=12,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)

print(f"Exported ONNX model saved at {ONNX_PATH}")
