#include <gtest/gtest.h>
#include "../src/optimizer/include/pass.h"
#include "../src/optimizer/include/optimization_passes.h"
#include "../src/ir/include/ir.h"

// 创建测试模块和图
std::shared_ptr<ir::Module> createTestModule() {
    auto module = std::make_shared<ir::Module>("test_module");
    auto graph = std::make_shared<ir::Graph>("test_graph");
    
    // 创建输入
    auto input = std::make_shared<ir::Value>("input", TypeUtils::createTensorType(
        TypeUtils::createFloat32Type(), {1, 3, 224, 224}));
    graph->addInput(input);
    
    // 创建常量
    auto const1 = std::make_shared<ir::ConstantOp>();
    const1->setAttribute("value", "2.0");
    auto const1Value = std::make_shared<ir::Value>("const1", TypeUtils::createFloat32Type());
    const1->addResult(const1Value);
    graph->addOperation(const1);
    
    auto const2 = std::make_shared<ir::ConstantOp>();
    const2->setAttribute("value", "3.0");
    auto const2Value = std::make_shared<ir::Value>("const2", TypeUtils::createFloat32Type());
    const2->addResult(const2Value);
    graph->addOperation(const2);
    
    // 创建乘法操作
    auto mul = std::make_shared<ir::BinaryOp>(ir::BinaryOp::OpKind::MUL);
    mul->addOperand(const1Value);
    mul->addOperand(const2Value);
    auto mulResult = std::make_shared<ir::Value>("mul_result", TypeUtils::createFloat32Type());
    mul->addResult(mulResult);
    graph->addOperation(mul);
    
    // 创建卷积操作
    auto conv = std::make_shared<ir::ConvOp>();
    conv->addOperand(input);
    conv->addOperand(mulResult);
    conv->setAttribute("stride", "1,1");
    conv->setAttribute("padding", "same");
    auto convResult = std::make_shared<ir::Value>("conv_result", TypeUtils::createTensorType(
        TypeUtils::createFloat32Type(), {1, 64, 224, 224}));
    conv->addResult(convResult);
    graph->addOperation(conv);
    
    // 创建ReLU操作
    auto relu = std::make_shared<ir::UnaryOp>(ir::UnaryOp::OpKind::RELU);
    relu->addOperand(convResult);
    auto reluResult = std::make_shared<ir::Value>("relu_result", TypeUtils::createTensorType(
        TypeUtils::createFloat32Type(), {1, 64, 224, 224}));
    relu->addResult(reluResult);
    graph->addOperation(relu);
    
    // 设置输出
    graph->addOutput(reluResult);
    
    // 添加图到模块
    module->addGraph(graph);
    
    return module;
}

// Pass框架测试
TEST(PassFrameworkTest, PassManager) {
    PassManager passManager;
    
    // 创建测试模块
    auto module = createTestModule();
    
    // 添加Pass
    passManager.addPass(std::make_shared<ConstantFoldingPass>());
    
    // 运行Pass
    bool changed = passManager.run(module);
    
    // 验证Pass执行结果
    EXPECT_TRUE(changed);
}

// 常量折叠Pass测试
TEST(OptimizationPassTest, ConstantFolding) {
    // 创建测试模块
    auto module = createTestModule();
    
    // 获取原始操作数量
    int originalOpCount = module->graphs[0]->operations.size();
    
    // 创建并运行常量折叠Pass
    auto pass = std::make_shared<ConstantFoldingPass>();
    bool changed = pass->run(module);
    
    // 验证Pass执行结果
    EXPECT_TRUE(changed);
    
    // 验证操作数量减少（两个常量和一个乘法操作被一个常量替代）
    EXPECT_EQ(module->graphs[0]->operations.size(), originalOpCount - 2);
    
    // 查找常量操作
    bool foundConstant = false;
    for (auto& op : module->graphs[0]->operations) {
        if (op->opType == "constant") {
            auto value = op->getAttribute("value");
            if (value == "6.0") {
                foundConstant = true;
                break;
            }
        }
    }
    
    EXPECT_TRUE(foundConstant);
}

// 死代码消除Pass测试
TEST(OptimizationPassTest, DeadCodeElimination) {
    // 创建测试模块
    auto module = createTestModule();
    auto graph = module->graphs[0];
    
    // 添加一个死代码操作
    auto deadConst = std::make_shared<ir::ConstantOp>();
    deadConst->setAttribute("value", "10.0");
    auto deadConstValue = std::make_shared<ir::Value>("dead_const", TypeUtils::createFloat32Type());
    deadConst->addResult(deadConstValue);
    graph->addOperation(deadConst);
    
    int originalOpCount = graph->operations.size();
    
    // 创建并运行死代码消除Pass
    auto pass = std::make_shared<DeadCodeEliminationPass>();
    bool changed = pass->run(module);
    
    // 验证Pass执行结果
    EXPECT_TRUE(changed);
    
    // 验证操作数量减少
    EXPECT_EQ(graph->operations.size(), originalOpCount - 1);
    
    // 验证死代码操作被移除
    bool foundDeadOp = false;
    for (auto& op : graph->operations) {
        if (op->opType == "constant" && op->getAttribute("value") == "10.0") {
            foundDeadOp = true;
            break;
        }
    }
    
    EXPECT_FALSE(foundDeadOp);
}

// 内核融合Pass测试
TEST(OptimizationPassTest, KernelFusion) {
    // 创建测试模块
    auto module = createTestModule();
    auto graph = module->graphs[0];
    
    // 获取原始操作数量
    int originalOpCount = graph->operations.size();
    
    // 创建并运行内核融合Pass
    auto pass = std::make_shared<KernelFusionPass>();
    bool changed = pass->run(module);
    
    // 验证Pass执行结果
    EXPECT_TRUE(changed);
    
    // 验证操作数量减少
    EXPECT_LT(graph->operations.size(), originalOpCount);
    
    // 查找融合内核
    bool foundFusedKernel = false;
    for (auto& op : graph->operations) {
        if (op->opType == "fused_kernel") {
            foundFusedKernel = true;
            break;
        }
    }
    
    EXPECT_TRUE(foundFusedKernel);
}

// 优化管道测试
TEST(OptimizationPassTest, OptimizationPipeline) {
    // 创建测试模块
    auto module = createTestModule();
    
    // 获取原始操作数量
    int originalOpCount = module->graphs[0]->operations.size();
    
    // 创建并运行优化管道
    OptimizationPipeline pipeline;
    bool changed = pipeline.run(module);
    
    // 验证优化管道执行结果
    EXPECT_TRUE(changed);
    
    // 验证操作数量减少
    EXPECT_LT(module->graphs[0]->operations.size(), originalOpCount);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
