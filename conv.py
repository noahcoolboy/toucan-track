import onnx_graphsurgeon as gs
import onnx
import numpy as np
model = gs.import_onnx(onnx.load("models/heavy.onnx"))
model.inputs[0].shape = ["N", 3, 256, 256]
model.outputs[0].shape = ["N", 195]
model.outputs[1].shape = ["N", 1]
model.outputs[3].shape = ["N", 64, 64, 39]
del model.outputs[2]
del model.outputs[3]

# write to out.onnx with opset 11
model.cleanup()
model = gs.export_onnx(model, do_type_check=True)
onnx.save(model, "models/pose_landmark_heavy_batched.onnx")

import onnxruntime
sess = onnxruntime.InferenceSession("models/pose_landmark_heavy_batched.onnx")
sess.run(["Identity", "Identity_1", "Identity_3"], { "input_1": np.random.randn(3, 3, 256, 256).astype(np.float32) })
print("Done")