from ThermalClassifier.models.resnet import resnet18
import torch
import gcsfs
import time

fs = gcsfs.GCSFileSystem(project="mod-gcp-white-soi-dev-1")

ckpt_path = 'gcs://soi-models/VMD-classifier/debug/checkpoints/epoch=14-step=825.ckpt'
torch_model = resnet18(num_target_classes=3)
torch_dict = torch.load(fs.open(ckpt_path, "rb"), map_location='cpu')
state_dict = {key.replace("model.", "") : value for key, value in torch_dict["state_dict"].items()}     
torch_model.load_state_dict(state_dict)

torch_model.eval()


torch_input = torch.randn(1, 3, 72, 72)
export_output = torch.onnx.export(torch_model, 
                                  torch_input, 
                                  "model.onnx",
                                  dynamic_axes={'input': [0], 'output': [0]},
                                  input_names=['input'],   # the model's input names
                                  output_names=['output'],
                                  do_constant_folding=True
                                )


import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)

# torch_input = torch.randn(10, 3, 72, 72)

import onnxruntime
import numpy as np

ort_session = onnxruntime.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(torch_input)}
start = time.time()
ort_outs = ort_session.run(None, ort_inputs)
print(f"onnx inference took: {time.time() - start}")

start = time.time()
ort_outs = torch_model(torch_input)
print(f"pytorch inference took: {time.time() - start}")