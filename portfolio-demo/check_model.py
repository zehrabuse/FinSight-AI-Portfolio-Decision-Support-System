import onnxruntime as ort

MODEL_PATH = "model/final_model.onnx"

session = ort.InferenceSession(MODEL_PATH)

print("=== MODEL INPUT ===")

for inp in session.get_inputs():
    print("Input name :", inp.name)
    print("Input shape:", inp.shape)
    print("Input type :", inp.type)

print("\n=== MODEL OUTPUT ===")

for out in session.get_outputs():
    print("Output name :", out.name)
    print("Output shape:", out.shape)
    print("Output type :", out.type)
