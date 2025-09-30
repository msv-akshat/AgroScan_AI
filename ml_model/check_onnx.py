import onnx
import onnxruntime as ort
import numpy as np

# load & check model
model = onnx.load("models/pretrained_model.onnx")
onnx.checker.check_model(model)
print("ONNX structure is valid ✅")

# run a dummy inference
session = ort.InferenceSession("models/pretrained_model.onnx", providers=["CPUExecutionProvider"])
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {"input": dummy_input})
print("ONNX inference ran successfully, output shape:", outputs[0].shape)
