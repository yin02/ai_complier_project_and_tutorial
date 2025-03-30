// 基础示例：多层感知机 (MLP) 网络
// 这个示例展示了如何使用我们的DSL定义一个简单的MLP网络

// 导入标准库
import "std";

// 定义MLP网络
graph mlp(input: tensor<float, [B, 784]>, 
          w1: tensor<float, [784, 128]>, 
          b1: tensor<float, [128]>,
          w2: tensor<float, [128, 10]>, 
          b2: tensor<float, [10]>): tensor<float, [B, 10]> {
    
    // 第一层：线性 + ReLU
    var z1: tensor<float, [B, 128]> = matmul(input, w1);
    var a1_pre: tensor<float, [B, 128]> = add(z1, b1);
    var a1: tensor<float, [B, 128]> = relu(a1_pre);
    
    // 第二层：线性 + Softmax
    var z2: tensor<float, [B, 10]> = matmul(a1, w2);
    var logits: tensor<float, [B, 10]> = add(z2, b2);
    var output: tensor<float, [B, 10]> = softmax(logits, 1);
    
    return output;
}

// 辅助函数：初始化权重
func init_weights(in_dim: int, out_dim: int): tensor<float, [?, ?]> {
    // 使用Xavier初始化
    var scale: float = sqrt(6.0 / (in_dim + out_dim));
    var weights: tensor<float, [in_dim, out_dim]> = random_uniform(-scale, scale, [in_dim, out_dim]);
    return weights;
}

// 辅助函数：初始化偏置
func init_bias(dim: int): tensor<float, [?]> {
    var bias: tensor<float, [dim]> = zeros([dim]);
    return bias;
}

// 主函数：创建和训练MLP
func main() {
    // 初始化模型参数
    var w1: tensor<float, [784, 128]> = init_weights(784, 128);
    var b1: tensor<float, [128]> = init_bias(128);
    var w2: tensor<float, [128, 10]> = init_weights(128, 10);
    var b2: tensor<float, [10]> = init_bias(10);
    
    // 加载MNIST数据集
    var mnist_data: tensor<float, [60000, 784]> = load_mnist_images("train-images");
    var mnist_labels: tensor<float, [60000, 10]> = load_mnist_labels("train-labels");
    
    // 训练参数
    var batch_size: int = 64;
    var learning_rate: float = 0.01;
    var num_epochs: int = 10;
    
    // 训练循环
    for (var epoch = 0; epoch < num_epochs; epoch = epoch + 1) {
        // 打乱数据
        var shuffled_indices: tensor<int, [60000]> = shuffle_indices(60000);
        
        // 批次训练
        for (var i = 0; i < 60000 / batch_size; i = i + 1) {
            // 获取批次数据
            var start_idx: int = i * batch_size;
            var end_idx: int = start_idx + batch_size;
            var batch_indices: tensor<int, [batch_size]> = shuffled_indices[start_idx:end_idx];
            var batch_data: tensor<float, [batch_size, 784]> = gather(mnist_data, batch_indices);
            var batch_labels: tensor<float, [batch_size, 10]> = gather(mnist_labels, batch_indices);
            
            // 前向传播
            var predictions: tensor<float, [batch_size, 10]> = mlp(batch_data, w1, b1, w2, b2);
            
            // 计算损失
            var loss: float = cross_entropy_loss(predictions, batch_labels);
            
            // 反向传播和参数更新（简化版）
            // 在实际实现中，这里会有梯度计算和参数更新的代码
            // 但为了示例简单，我们省略了这些细节
            
            // 打印进度
            if (i % 100 == 0) {
                print("Epoch: ", epoch, ", Batch: ", i, ", Loss: ", loss);
            }
        }
    }
    
    // 评估模型
    var test_data: tensor<float, [10000, 784]> = load_mnist_images("test-images");
    var test_labels: tensor<float, [10000, 10]> = load_mnist_labels("test-labels");
    var test_predictions: tensor<float, [10000, 10]> = mlp(test_data, w1, b1, w2, b2);
    var accuracy: float = compute_accuracy(test_predictions, test_labels);
    
    print("Test accuracy: ", accuracy);
    
    // 保存模型
    save_model("mlp_model", w1, b1, w2, b2);
}
