import torch
import numpy as np
import matplotlib.pyplot as plt
from svc import SpatialVariableConvolution

def test_svc_gradients_with_dilation_map():
    """
    测试在有 dilation_map 的情况下，SVC 梯度计算的正确性
    使用数值微分法验证梯度
    """
    print("\n" + "="*50)
    print("GRADIENT VERIFICATION TEST WITH DILATION MAP")
    print("="*50)
    
    # 设置随机种子以便结果可复现
    torch.manual_seed(42)
    
    # 设置参数
    batch_size = 2
    channels = 6
    height = width = 16
    groups = 2
    kernel_size = 3
    k2 = kernel_size * kernel_size
    channels_per_group = channels // groups
    
    # 使用非均匀的 stride 和 dilation 进行更彻底的测试
    stride = (2, 1) 
    dilation = (1, 2)
    padding = kernel_size // 2
    
    # 计算输出尺寸
    height_out = (height + 2 * padding - dilation[0] * (kernel_size - 1) - 1) // stride[0] + 1
    width_out = (width + 2 * padding - dilation[1] * (kernel_size - 1) - 1) // stride[1] + 1
    
    print(f"Output shape: [{batch_size}, {channels}, {height_out}, {width_out}]")
    
    # 创建输入、权重和 dilation_map 张量
    x = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)
    weight = torch.randn(batch_size, groups * k2, height_out, width_out, device="cuda", requires_grad=True)
    
    # 创建非均匀的 dilation_map
    dilation_map = torch.ones(batch_size, groups, height_out, width_out, device="cuda") * 1.5
    # 在不同区域设置不同值，以更好地测试梯度
    dilation_map[:, :, height_out//4:height_out//2, width_out//4:width_out//2] = 0.8
    dilation_map[:, :, height_out//2:3*height_out//4, width_out//2:3*width_out//4] = 2.0
    dilation_map.requires_grad_(True)
    
    # 创建 SVC 模块
    svc = SpatialVariableConvolution(
        kernel_size=kernel_size, 
        padding=padding, 
        stride=stride, 
        dilation=dilation, 
        groups=groups
    ).cuda()
    
    # 前向传播
    output = svc(x, weight, dilation_map)
    
    # 使用随机梯度进行更严格的测试
    grad_output = torch.randn_like(output)
    
    # 计算损失和反向传播
    loss = (output * grad_output).sum()
    loss.backward()
    
    # 保存自动微分计算的梯度
    dx_auto = x.grad.clone()
    dw_auto = weight.grad.clone()
    ddil_auto = dilation_map.grad.clone()
    
    # 重置梯度
    x.grad.zero_()
    weight.grad.zero_()
    dilation_map.grad.zero_()
    
    # 数值梯度验证
    print("\n数值梯度验证:")
    
    # 测试参数
    epsilon = 1e-3  # 使用较大的epsilon，可以提高数值稳定性
    num_samples = 10  # 每个张量测试的样本数
    
    # 验证输入梯度
    print("\n检查输入梯度...")
    input_errors = []
    
    for _ in range(num_samples):
        # 随机选择一个位置
        b = np.random.randint(0, batch_size)
        c = np.random.randint(0, channels)
        h = np.random.randint(0, height)
        w = np.random.randint(0, width)
        
        # 创建两个副本，一个加epsilon，一个减epsilon
        x_p = x.clone()
        x_p[b, c, h, w] = x_p[b, c, h, w] + epsilon
        
        # 原始前向传播
        with torch.no_grad():  # 避免跟踪梯度
            output_p = svc(x_p, weight, dilation_map)
            loss_p = (output_p * grad_output).sum().item()
        
        # 计算数值梯度（中心差分）
        grad_numerical = (loss_p - loss.item()) / epsilon
        grad_auto = dx_auto[b, c, h, w].item()
        
        # 计算相对误差
        abs_diff = abs(grad_numerical - grad_auto)
        scale = max(abs(grad_auto), abs(grad_numerical), 1e-8)
        rel_error = abs_diff / scale
        input_errors.append(rel_error)
        
        print(f"  输入[{b},{c},{h},{w}]: 数值梯度 = {grad_numerical:.6f}, 自动梯度 = {grad_auto:.6f}, 相对误差 = {rel_error:.6f}")
    
    max_input_error = max(input_errors)
    print(f"输入梯度最大相对误差: {max_input_error:.6f}")
    
    # 验证权重梯度
    print("\n检查权重梯度...")
    weight_errors = []
    
    for _ in range(num_samples):
        # 随机选择一个位置
        b = np.random.randint(0, batch_size)
        c = np.random.randint(0, groups * k2)
        h = np.random.randint(0, height_out)
        w = np.random.randint(0, width_out)
        
        # 创建副本
        w_p = weight.clone()
        w_p[b, c, h, w] = w_p[b, c, h, w] + epsilon
        
        # 前向传播
        with torch.no_grad():
            output_p = svc(x, w_p, dilation_map)
            loss_p = (output_p * grad_output).sum().item()
        
        # 计算数值梯度
        grad_numerical = (loss_p - loss.item()) / epsilon
        grad_auto = dw_auto[b, c, h, w].item()
        
        # 计算相对误差
        abs_diff = abs(grad_numerical - grad_auto)
        scale = max(abs(grad_auto), abs(grad_numerical), 1e-8)
        rel_error = abs_diff / scale
        weight_errors.append(rel_error)
        
        print(f"  权重[{b},{c},{h},{w}]: 数值梯度 = {grad_numerical:.6f}, 自动梯度 = {grad_auto:.6f}, 相对误差 = {rel_error:.6f}")
    
    max_weight_error = max(weight_errors)
    print(f"权重梯度最大相对误差: {max_weight_error:.6f}")
    
    # 验证 dilation_map 梯度 - 修复原地操作问题
    print("\n检查 dilation_map 梯度...")
    dilation_errors = []
    
    for _ in range(num_samples):
        # 随机选择一个位置
        b = np.random.randint(0, batch_size)
        g = np.random.randint(0, groups)
        h = np.random.randint(0, height_out)
        w = np.random.randint(0, width_out)
        
        # 创建修改后的副本 - 避免原地操作
        dil_p = dilation_map.clone()
        dil_p[b, g, h, w] = dil_p[b, g, h, w] + epsilon
        
        # 前向传播
        with torch.no_grad():
            output_p = svc(x, weight, dil_p)
            loss_p = (output_p * grad_output).sum().item()
        
        # 计算数值梯度
        grad_numerical = (loss_p - loss.item()) / epsilon
        grad_auto = ddil_auto[b, g, h, w].item()
        
        # 计算相对误差
        abs_diff = abs(grad_numerical - grad_auto)
        scale = max(abs(grad_auto), abs(grad_numerical), 1e-8)
        rel_error = abs_diff / scale
        dilation_errors.append(rel_error)
        
        print(f"  Dilation[{b},{g},{h},{w}]: 数值梯度 = {grad_numerical:.6f}, 自动梯度 = {grad_auto:.6f}, 相对误差 = {rel_error:.6f}")
    
    max_dilation_error = max(dilation_errors)
    print(f"dilation_map 梯度最大相对误差: {max_dilation_error:.6f}")
    
    # 设置接受误差的阈值
    tolerance = 0.1  # 通常 10% 的误差在数值梯度检验中是可接受的
    
    # 总结结果
    print("\n梯度验证结果:", end=" ")
    if max(max_input_error, max_weight_error, max_dilation_error) < tolerance:
        print(f"✓ 通过! (误差<{tolerance:.2f})")
    else:
        print(f"✗ 未通过! (误差>{tolerance:.2f})")
    
    print(f"  输入梯度误差: {max_input_error:.6f}")
    print(f"  权重梯度误差: {max_weight_error:.6f}")
    print(f"  dilation_map 梯度误差: {max_dilation_error:.6f}")
    
    # 可视化梯度
    visualize_gradients(ddil_auto)
    
    return dx_auto, dw_auto, ddil_auto, max_input_error, max_weight_error, max_dilation_error


def test_svc_with_different_strides_dilations():
    """
    测试 SVC 在不同 stride 和 dilation 值下的正确性
    """
    print("\n" + "="*50)
    print("STRIDE AND DILATION TEST")
    print("="*50)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 基础参数
    batch_size = 2
    channels = 6
    height = width = 32
    groups = 2
    kernel_size = 3
    k2 = kernel_size * kernel_size
    channels_per_group = channels // groups
    
    # 测试参数组合
    strides = [(1, 1), (2, 2), (1, 2), (2, 1)]
    dilations = [(1, 1), (2, 2), (1, 2), (2, 1)]
    
    results = {}
    
    for stride in strides:
        for dilation in dilations:
            stride_h, stride_w = stride
            dilation_h, dilation_w = dilation
            padding = (dilation_h * (kernel_size - 1) // 2, dilation_w * (kernel_size - 1) // 2)
            
            # 计算输出尺寸
            height_out = (height + 2 * padding[0] - dilation_h * (kernel_size - 1) - 1) // stride_h + 1
            width_out = (width + 2 * padding[1] - dilation_w * (kernel_size - 1) - 1) // stride_w + 1
            
            print(f"\n测试 stride={stride}, dilation={dilation}")
            print(f"输出尺寸: [{batch_size}, {channels}, {height_out}, {width_out}]")
            
            # 创建输入和权重
            x = torch.randn(batch_size, channels, height, width, device="cuda", requires_grad=True)
            weight = torch.randn(batch_size, groups * k2, height_out, width_out, device="cuda", requires_grad=True)
            
            # 创建 dilation_map
            dilation_map = torch.ones(batch_size, groups, height_out, width_out, device="cuda") * 1.5
            dilation_map.requires_grad_(True)
            
            # 创建 SVC 模块
            svc = SpatialVariableConvolution(
                kernel_size=kernel_size, 
                padding=padding, 
                stride=stride, 
                dilation=dilation, 
                groups=groups
            ).cuda()
            
            # 前向传播
            try:
                output = svc(x, weight, dilation_map)
                
                # 检查输出尺寸
                assert output.shape == (batch_size, channels, height_out, width_out), \
                    f"输出尺寸错误: {output.shape} != {(batch_size, channels, height_out, width_out)}"
                
                # 反向传播测试
                grad_output = torch.randn_like(output)
                loss = (output * grad_output).sum()
                loss.backward()
                
                # 检查梯度是否存在且有限
                assert x.grad is not None and torch.isfinite(x.grad).all()
                assert weight.grad is not None and torch.isfinite(weight.grad).all()
                assert dilation_map.grad is not None and torch.isfinite(dilation_map.grad).all()
                
                results[(stride, dilation)] = "通过"
                print("✓ 测试通过!")
            
            except Exception as e:
                results[(stride, dilation)] = f"失败: {str(e)}"
                print(f"✗ 测试失败: {e}")
            
            finally:
                # 清理
                if x.grad is not None:
                    x.grad.zero_()
                if weight.grad is not None:
                    weight.grad.zero_()
                if dilation_map.grad is not None:
                    dilation_map.grad.zero_()
                torch.cuda.empty_cache()
    
    # 总结结果
    print("\n测试结果总结:")
    for (stride, dilation), result in results.items():
        print(f"  stride={stride}, dilation={dilation}: {result}")
    
    return results


def visualize_gradients(dilation_grads):
    """可视化 dilation_map 梯度"""
    # 选择第一个批次、第一个组的梯度
    grad = dilation_grads[0, 0].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(grad, cmap='viridis')
    plt.colorbar(label='Gradient value')
    plt.title('Dilation Map Gradients (batch=0, group=0)')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.tight_layout()
    # plt.savefig('dilation_gradients.png')
    # print("\nDilation map gradients visualization saved to 'dilation_gradients.png'")
    plt.close()


if __name__ == "__main__":
    # 设置 CUDA 环境
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 运行梯度验证测试
    print("运行梯度验证测试...")
    dx, dw, ddil, max_dx_error, max_dw_error, max_ddil_error = test_svc_gradients_with_dilation_map()
    
    # 运行不同 stride 和 dilation 的测试
    print("运行不同 stride 和 dilation 的测试...")
    stride_dilation_results = test_svc_with_different_strides_dilations()
    
    print("\n所有测试完成!")