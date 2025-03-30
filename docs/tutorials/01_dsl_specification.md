# AI 编译器 DSL 规范

本文档定义了我们 AI 编译器项目中使用的领域特定语言 (DSL) 规范。这个 DSL 专为 AI 模型计算图的表示和操作而设计。

## 1. 语言概述

我们的 DSL 是一种声明式语言，用于描述 AI 模型的计算图。它允许用户定义张量、操作和计算流程，以便编译器能够优化和生成高效的执行代码。

## 2. 基本语法

### 2.1 注释

```
// 单行注释
/* 多行
   注释 */
```

### 2.2 数据类型

基本数据类型：
- `float`: 32位浮点数
- `int`: 32位整数
- `bool`: 布尔值
- `tensor<type, shape>`: 张量类型，其中 type 是元素类型，shape 是形状描述

示例：
```
tensor<float, [2, 3]>  // 2x3 的浮点张量
tensor<int, [10]>      // 长度为 10 的整数向量
```

### 2.3 变量声明

```
var name: type = value;
```

示例：
```
var x: tensor<float, [2, 3]> = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
var y: int = 5;
```

### 2.4 操作定义

```
op name(param1: type1, param2: type2, ...): return_type {
    // 操作体
}
```

示例：
```
op matmul(a: tensor<float, [M, K]>, b: tensor<float, [K, N]>): tensor<float, [M, N]> {
    // 矩阵乘法实现
}
```

### 2.5 计算图定义

```
graph name(param1: type1, param2: type2, ...): return_type {
    // 图体
}
```

示例：
```
graph mlp(input: tensor<float, [B, I]>, w1: tensor<float, [I, H]>, w2: tensor<float, [H, O]>): tensor<float, [B, O]> {
    var h = matmul(input, w1);
    var h_relu = relu(h);
    var output = matmul(h_relu, w2);
    return output;
}
```

## 3. 内置操作

### 3.1 张量操作

- `add(a, b)`: 张量加法
- `sub(a, b)`: 张量减法
- `mul(a, b)`: 张量乘法（元素级）
- `div(a, b)`: 张量除法（元素级）
- `matmul(a, b)`: 矩阵乘法
- `transpose(a, dims)`: 张量转置
- `reshape(a, shape)`: 张量形状重塑
- `concat(a, b, axis)`: 沿指定轴连接张量

### 3.2 激活函数

- `relu(x)`: ReLU 激活函数
- `sigmoid(x)`: Sigmoid 激活函数
- `tanh(x)`: Tanh 激活函数
- `softmax(x, axis)`: Softmax 函数

### 3.3 规范化操作

- `batch_norm(x, mean, var, scale, bias)`: 批量归一化
- `layer_norm(x, scale, bias)`: 层归一化

### 3.4 卷积操作

- `conv2d(input, filter, stride, padding)`: 2D 卷积
- `max_pool(input, kernel_size, stride, padding)`: 最大池化
- `avg_pool(input, kernel_size, stride, padding)`: 平均池化

## 4. 控制流

### 4.1 条件语句

```
if (condition) {
    // 真分支
} else {
    // 假分支
}
```

### 4.2 循环

```
for (var i = 0; i < n; i = i + 1) {
    // 循环体
}
```

## 5. 函数和模块

### 5.1 函数定义

```
func name(param1: type1, param2: type2, ...): return_type {
    // 函数体
}
```

### 5.2 模块导入

```
import "module_name";
```

## 6. 示例程序

### 6.1 简单的 MLP 网络

```
graph mlp(input: tensor<float, [B, 784]>, 
          w1: tensor<float, [784, 128]>, 
          b1: tensor<float, [128]>,
          w2: tensor<float, [128, 10]>, 
          b2: tensor<float, [10]>): tensor<float, [B, 10]> {
    
    // 第一层：线性 + ReLU
    var z1 = add(matmul(input, w1), b1);
    var a1 = relu(z1);
    
    // 第二层：线性 + Softmax
    var z2 = add(matmul(a1, w2), b2);
    var output = softmax(z2, 1);
    
    return output;
}
```

### 6.2 简单的 CNN 网络

```
graph cnn(input: tensor<float, [B, 1, 28, 28]>,
          conv1_w: tensor<float, [32, 1, 5, 5]>,
          conv1_b: tensor<float, [32]>,
          conv2_w: tensor<float, [64, 32, 5, 5]>,
          conv2_b: tensor<float, [64]>,
          fc1_w: tensor<float, [1024, 128]>,
          fc1_b: tensor<float, [128]>,
          fc2_w: tensor<float, [128, 10]>,
          fc2_b: tensor<float, [10]>): tensor<float, [B, 10]> {
    
    // 第一个卷积层
    var conv1 = conv2d(input, conv1_w, [1, 1], "same");
    var conv1_bias = add(conv1, conv1_b);
    var conv1_relu = relu(conv1_bias);
    var pool1 = max_pool(conv1_relu, [2, 2], [2, 2], "valid");
    
    // 第二个卷积层
    var conv2 = conv2d(pool1, conv2_w, [1, 1], "same");
    var conv2_bias = add(conv2, conv2_b);
    var conv2_relu = relu(conv2_bias);
    var pool2 = max_pool(conv2_relu, [2, 2], [2, 2], "valid");
    
    // 展平
    var flat = reshape(pool2, [B, 1024]);
    
    // 全连接层
    var fc1 = add(matmul(flat, fc1_w), fc1_b);
    var fc1_relu = relu(fc1);
    var fc2 = add(matmul(fc1_relu, fc2_w), fc2_b);
    
    // 输出层
    var output = softmax(fc2, 1);
    
    return output;
}
```

## 7. 语法扩展

随着项目的发展，我们将扩展 DSL 语法以支持更多高级功能，如：

- 自定义操作定义
- 图优化指令
- 内存布局控制
- 并行执行指令
- 设备放置策略
