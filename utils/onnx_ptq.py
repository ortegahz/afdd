from onnxruntime.quantization import (
    matmul_4bits_quantizer,
    quant_utils
)

model_fp32_path = "path/to/orignal/model.onnx"
model_int4_path = "path/to/save/quantized/model.onnx"

quant_config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
    block_size=128,  # 2's exponential and >= 16
    is_symmetric=True,  # if true, quantize to Int4. otherwise, quantize to uint4.
    accuracy_level=4,
    # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35
    quant_format=quant_utils.QuantFormat.QOperator,
    op_types_to_quantize=("MatMul", "Gather"),  # specify which op types to quantize
    quant_axes=(("MatMul", 0), ("Gather", 1),)  # specify which axis to quantize for an op type.

model = quant_utils.load_model_with_shape_infer(Path(model_fp32_pfrom onnxruntime.quantization import (
    matmul_4bits_quantizer,
    quant_utils,
    quantize
)
from pathlib import Path

model_fp32_path="path/to/orignal/model.onnx"
model_int4_path="path/to/save/quantized/model.onnx"

quant_config = matmul_4bits_quantizer.DefaultWeightOnlyQuantConfig(
  block_size=128, # 2's exponential and >= 16
  is_symmetric=True, # if true, quantize to Int4. otherwise, quantize to uint4.
  accuracy_level=4, # used by MatMulNbits, see https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#attributes-35
  quant_format=quant_utils.QuantFormat.QOperator,
  op_types_to_quantize=("MatMul","Gather"), # specify which op types to quantize
  quant_axes=(("MatMul", 0), ("Gather", 1),) # specify which axis to quantize for an op type.

model = quant_utils.load_model_with_shape_infer(Path(model_fp32_path))
quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
  model,
  nodes_to_exclude=None, # specify a list of nodes to exclude from quantization
  nodes_to_include=None, # specify a list of nodes to force include from quantization
  algo_config=quant_config,)
quant.process()
quant.model.save_model_to_file(
  model_int4_path,
  True) # save data to external file
ath))
quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
    model,
    nodes_to_exclude=None,  # specify a list of nodes to exclude from quantization
    nodes_to_include=None,  # specify a list of nodes to force include from quantization
    algo_config=quant_config, )
quant.process()
quant.model.save_model_to_file(
    model_int4_path,
    True)  # save data to external file
