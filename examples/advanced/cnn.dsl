// 高级示例：卷积神经网络 (CNN) 模型
// 这个示例展示了如何使用我们的DSL定义一个CNN网络用于图像分类

// 导入标准库
import "std";

// 定义CNN网络
graph cnn(input: tensor<float, [B, 1, 28, 28]>,
          conv1_w: tensor<float, [32, 1, 5, 5]>,
          conv1_b: tensor<float, [32]>,
          conv2_w: tensor<float, [64, 32, 5, 5]>,
          conv2_b: tensor<float, [64]>,
          fc1_w: tensor<float, [1024, 128]>,
          fc1_b: tensor<float, [128]>,
          fc2_w: tensor<float, [128, 10]>,
          fc2_b: tensor<float, [10]>): tensor<float, [B, 10]> {
    
    // 第一个卷积层 + 池化层
    // 输入: [B, 1, 28, 28] -> 输出: [B, 32, 14, 14]
    var conv1: tensor<float, [B, 32, 28, 28]> = conv2d(input, conv1_w, [1, 1], "same");
    var conv1_bias: tensor<float, [B, 32, 28, 28]> = add(conv1, conv1_b);
    var conv1_relu: tensor<float, [B, 32, 28, 28]> = relu(conv1_bias);
    var pool1: tensor<float, [B, 32, 14, 14]> = max_pool(conv1_relu, [2, 2], [2, 2], "valid");
    
    // 第二个卷积层 + 池化层
    // 输入: [B, 32, 14, 14] -> 输出: [B, 64, 7, 7]
    var conv2: tensor<float, [B, 64, 14, 14]> = conv2d(pool1, conv2_w, [1, 1], "same");
    var conv2_bias: tensor<float, [B, 64, 14, 14]> = add(conv2, conv2_b);
    var conv2_relu: tensor<float, [B, 64, 14, 14]> = relu(conv2_bias);
    var pool2: tensor<float, [B, 64, 7, 7]> = max_pool(conv2_relu, [2, 2], [2, 2], "valid");
    
    // 展平操作
    // 输入: [B, 64, 7, 7] -> 输出: [B, 1024]
    var flat: tensor<float, [B, 1024]> = reshape(pool2, [B, 1024]);
    
    // 第一个全连接层
    // 输入: [B, 1024] -> 输出: [B, 128]
    var fc1: tensor<float, [B, 128]> = matmul(flat, fc1_w);
    var fc1_bias: tensor<float, [B, 128]> = add(fc1, fc1_b);
    var fc1_relu: tensor<float, [B, 128]> = relu(fc1_bias);
    
    // 第二个全连接层（输出层）
    // 输入: [B, 128] -> 输出: [B, 10]
    var fc2: tensor<float, [B, 10]> = matmul(fc1_relu, fc2_w);
    var fc2_bias: tensor<float, [B, 10]> = add(fc2, fc2_b);
    
    // Softmax激活
    var output: tensor<float, [B, 10]> = softmax(fc2_bias, 1);
    
    return output;
}

// 辅助函数：初始化卷积权重
func init_conv_weights(out_channels: int, in_channels: int, 
                      kernel_h: int, kernel_w: int): tensor<float, [?, ?, ?, ?]> {
    // 使用He初始化
    var fan_in: int = in_channels * kernel_h * kernel_w;
    var scale: float = sqrt(2.0 / fan_in);
    var weights: tensor<float, [out_channels, in_channels, kernel_h, kernel_w]> = 
        random_normal(0.0, scale, [out_channels, in_channels, kernel_h, kernel_w]);
    return weights;
}

// 辅助函数：初始化全连接层权重
func init_fc_weights(in_dim: int, out_dim: int): tensor<float, [?, ?]> {
    // 使用He初始化
    var scale: float = sqrt(2.0 / in_dim);
    var weights: tensor<float, [in_dim, out_dim]> = random_normal(0.0, scale, [in_dim, out_dim]);
    return weights;
}

// 辅助函数：初始化偏置
func init_bias(dim: int): tensor<float, [?]> {
    var bias: tensor<float, [dim]> = zeros([dim]);
    return bias;
}

// 主函数：创建和训练CNN
func main() {
    // 初始化模型参数
    var conv1_w: tensor<float, [32, 1, 5, 5]> = init_conv_weights(32, 1, 5, 5);
    var conv1_b: tensor<float, [32]> = init_bias(32);
    var conv2_w: tensor<float, [64, 32, 5, 5]> = init_conv_weights(64, 32, 5, 5);
    var conv2_b: tensor<float, [64]> = init_bias(64);
    var fc1_w: tensor<float, [1024, 128]> = init_fc_weights(1024, 128);
    var fc1_b: tensor<float, [128]> = init_bias(128);
    var fc2_w: tensor<float, [128, 10]> = init_fc_weights(128, 10);
    var fc2_b: tensor<float, [10]> = init_bias(10);
    
    // 加载MNIST数据集
    var mnist_data: tensor<float, [60000, 1, 28, 28]> = load_mnist_images("train-images");
    var mnist_labels: tensor<float, [60000, 10]> = load_mnist_labels("train-labels");
    
    // 训练参数
    var batch_size: int = 64;
    var learning_rate: float = 0.001;
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
            var batch_data: tensor<float, [batch_size, 1, 28, 28]> = gather(mnist_data, batch_indices);
            var batch_labels: tensor<float, [batch_size, 10]> = gather(mnist_labels, batch_indices);
            
            // 前向传播
            var predictions: tensor<float, [batch_size, 10]> = cnn(
                batch_data, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b);
            
            // 计算损失
            var loss: float = cross_entropy_loss(predictions, batch_labels);
            
            // 反向传播和参数更新
            // 在实际实现中，这里会有梯度计算和参数更新的代码
            // 但为了示例简单，我们省略了这些细节
            
            // 打印进度
            if (i % 100 == 0) {
                print("Epoch: ", epoch, ", Batch: ", i, ", Loss: ", loss);
            }
        }
        
        // 每个epoch结束后评估模型
        var test_data: tensor<float, [10000, 1, 28, 28]> = load_mnist_images("test-images");
        var test_labels: tensor<float, [10000, 10]> = load_mnist_labels("test-labels");
        var test_predictions: tensor<float, [10000, 10]> = cnn(
            test_data, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b);
        var accuracy: float = compute_accuracy(test_predictions, test_labels);
        
        print("Epoch: ", epoch, ", Test accuracy: ", accuracy);
    }
    
    // 保存模型
    save_model("cnn_model", conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b);
}

// 推理函数：使用训练好的模型进行预测
func predict(image: tensor<float, [1, 28, 28]>) {
    // 加载模型参数
    var params: tensor<float, [8]> = load_model("cnn_model");
    var conv1_w: tensor<float, [32, 1, 5, 5]> = params[0];
    var conv1_b: tensor<float, [32]> = params[1];
    var conv2_w: tensor<float, [64, 32, 5, 5]> = params[2];
    var conv2_b: tensor<float, [64]> = params[3];
    var fc1_w: tensor<float, [1024, 128]> = params[4];
    var fc1_b: tensor<float, [128]> = params[5];
    var fc2_w: tensor<float, [128, 10]> = params[6];
    var fc2_b: tensor<float, [10]> = params[7];
    
    // 添加批次维度
    var input: tensor<float, [1, 1, 28, 28]> = reshape(image, [1, 1, 28, 28]);
    
    // 进行预测
    var prediction: tensor<float, [1, 10]> = cnn(
        input, conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b);
    
    // 获取预测类别
    var predicted_class: int = argmax(prediction, 1);
    
    print("Predicted class: ", predicted_class);
}
