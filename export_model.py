import torch
import torch.nn as nn
from FMDQN_core.FMCQL import CQLAgent
import torch.jit
import struct

def export_model():
    # 创建CQLAgent实例，只指定必要的参数
    agent = CQLAgent(
        observation_dim=(3, 51, 101),  # 实际输入维度
        action_size=6,  # 实际动作空间
    )

    # 加载训练好的权重
    agent.load_model_params()
    agent.evaluate_net.eval()

    # 创建示例输入
    example = torch.randn(1, 3, 51, 101)

    # 导出evaluate_net模型
    with torch.jit.optimized_execution(True):  # 使用优化执行模式
        traced_script_module = torch.jit.trace(agent.evaluate_net, example)

    # 验证导出的模型
    print("Testing exported model...")
    with torch.no_grad():
        original_output = agent.evaluate_net(example)
        traced_output = traced_script_module(example)
        
        print("Original model output shape:", original_output.shape)
        print("Traced model output shape:", traced_output.shape)
        print("Max difference:", torch.max(torch.abs(original_output - traced_output)))

    # 保存模型时使用V2版本
    torch.jit.save(traced_script_module, "data/traced_cql_model.pt", _extra_files={
        "_torch_jit_version": "2"  # 指定使用V2版本
    })
    
    # 读取模型文件并检查版本
    with open("data/traced_cql_model.pt", 'rb') as f:
        # 读取前8个字节
        magic_number = f.read(8)
        # 读取版本号（4字节）
        version = struct.unpack('i', f.read(4))[0]
        print(f"Model file version: {version}")
    
    print("Model exported successfully!")

if __name__ == "__main__":
    export_model()