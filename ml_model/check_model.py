import h5py

file_path = "models/Pretrained_model.h5"
with h5py.File(file_path, "r") as f:
    print("Groups in file:", list(f.keys()))
    if "model_weights" in f:
        print("Layers in model_weights:", list(f["model_weights"].keys()))