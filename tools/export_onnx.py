# FILE: export_onnx.py

import os

import numpy as np
import torch

# 确保您可以从您的项目结构中导入这些模块
# 如果 nets.py 在不同目录下，您可能需要调整 sys.path
from cores.nets import NetAFDAE_UNet

# --- 配置 ---
SAVE_DIR = "/home/manu/tmp"  # 您希望保存ONNX文件的目录
ONNX_FILENAME = "afdd_ae_unet.onnx"
ONNX_PATH = os.path.join(SAVE_DIR, ONNX_FILENAME)


def export_model_to_onnx():
    """
    实例化 NetAFDAE_UNet 模型，并将其导出为 ONNX 格式。
    """
    # 确保保存目录存在
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 1. 初始化模型并设置为评估模式 ---
    print("Step 1: Initializing the NetAFDAE_UNet model...")
    model = NetAFDAE_UNet(latent_dim=128)
    model.eval()  # 非常重要！必须设置为评估模式
    print("Model initialized successfully.")

    # --- 2. 创建一个符合模型输入的示例张量 ---
    # 根据模型定义，输入长度是固定的 448
    # 形状: (batch_size, channels, sequence_length)
    batch_size = 1
    input_channels = 1
    seq_length = 448  # 这是模型内部断言的长度

    dummy_input = torch.randn(batch_size, input_channels, seq_length, requires_grad=False)
    print(f"Step 2: Created a dummy input tensor with shape {dummy_input.shape}.")

    # --- 3. 定义 ONNX 导出参数 ---
    # 为输入和输出节点命名，方便后续调用
    input_names = ["input"]
    # 模型的 forward 方法返回 (reconstructed_x, z)，所以有两个输出
    output_names = ["reconstruction", "latent"]

    print("Step 3: Configuring ONNX export parameters...")
    print(f"  - Input node name: {input_names[0]}")
    print(f"  - Output node names: {output_names}")

    # 设置动态轴，允许模型处理不同大小的批量
    # 'batch_size' 是我们给这个动态轴起的名字
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'reconstruction': {0: 'batch_size'},
        'latent': {0: 'batch_size'}
    }
    print("  - Dynamic batch size has been enabled.")

    # --- 4. 执行导出 ---
    print(f"\nStep 4: Starting ONNX export to '{ONNX_PATH}'...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            ONNX_PATH,
            export_params=True,
            opset_version=12,  # 推荐使用 11 或更高的版本
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        print("ONNX export completed successfully!")
    except Exception as e:
        print(f"An error occurred during export: {e}")
        return

    # --- 5. 验证导出的模型 ---
    print("\nStep 5: Verifying the exported ONNX model...")
    verify_onnx_model(model, dummy_input)


def verify_onnx_model(pytorch_model, dummy_input):
    """
    使用 onnxruntime 加载导出的模型，并验证其输出是否与原始 PyTorch 模型一致。
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nVerification failed: onnxruntime is not installed.")
        print("Please install it using: pip install onnxruntime")
        return

    try:
        # 加载 ONNX 模型
        ort_session = ort.InferenceSession(ONNX_PATH)
        print("ONNX model loaded successfully with onnxruntime.")

        # 准备 onnxruntime 的输入
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

        # 使用 onnxruntime 进行推理
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_reconstruction, ort_latent = ort_outputs

        # 使用 PyTorch 模型进行推理
        with torch.no_grad():
            pt_reconstruction, pt_latent = pytorch_model(dummy_input)

        # 比较 PyTorch 和 ONNX 的输出
        print("Comparing outputs from PyTorch and ONNX Runtime...")

        # 允许一定的浮点误差 (atol=absolute tolerance, rtol=relative tolerance)
        np.testing.assert_allclose(pt_reconstruction.numpy(), ort_reconstruction, rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(pt_latent.numpy(), ort_latent, rtol=1e-03, atol=1e-05)

        print("\nVerification successful! The outputs of the PyTorch and ONNX models match.")

    except Exception as e:
        print(f"\nVerification failed! An error occurred: {e}")


if __name__ == "__main__":
    # 在运行前，请确保您已安装 onnx 和 onnxruntime
    # pip install onnx onnxruntime
    export_model_to_onnx()
