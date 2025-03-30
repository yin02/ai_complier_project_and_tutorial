#include <gtest/gtest.h>
#include "../src/backend/include/llvm_backend.h"
#include "../src/ir/include/ir.h"
#include "../src/ir/include/ast_to_ir.h"
#include "../src/frontend/parser.h"

// 创建测试IR模块
std::shared_ptr<ir::Module> createTestIRModule() {
    auto module = std::make_shared<ir::Module>("test_module");
    auto graph = std::make_shared<ir::Graph>("test_function");
    
    // 创建函数参数
    auto paramA = std::make_shared<ir::Value>("a", TypeUtils::createFloat32Type());
    auto paramB = std::make_shared<ir::Value>("b", TypeUtils::createFloat32Type());
    graph->addInput(paramA);
    graph->addInput(paramB);
    
    // 创建加法操作
    auto addOp = std::make_shared<ir::BinaryOp>(ir::BinaryOp::OpKind::ADD);
    addOp->addOperand(paramA);
    addOp->addOperand(paramB);
    auto addResult = std::make_shared<ir::Value>("add_result", TypeUtils::createFloat32Type());
    addOp->addResult(addResult);
    graph->addOperation(addOp);
    
    // 设置函数返回值
    graph->addOutput(addResult);
    
    // 添加图到模块
    module->addGraph(graph);
    
    return module;
}

// IR到LLVM IR转换测试
TEST(LLVMBackendTest, IRToLLVMConversion) {
    // 创建测试IR模块
    auto irModule = createTestIRModule();
    
    // 创建IR到LLVM IR转换器
    ai_compiler::backend::IRToLLVMConverter converter;
    
    // 转换IR到LLVM IR
    auto llvmModule = converter.convert(irModule);
    
    // 验证转换结果
    ASSERT_NE(llvmModule, nullptr);
    
    // 验证模块名称
    EXPECT_EQ(llvmModule->getName(), "test_module");
    
    // 验证函数存在
    auto func = llvmModule->getFunction("test_function");
    ASSERT_NE(func, nullptr);
    
    // 验证函数参数
    EXPECT_EQ(func->arg_size(), 2);
    
    // 验证函数返回类型
    EXPECT_TRUE(func->getReturnType()->isFloatTy());
    
    // 验证函数不为空
    EXPECT_FALSE(func->empty());
}

// LLVM优化Pass测试
TEST(LLVMBackendTest, LLVMPassManager) {
    // 创建测试IR模块
    auto irModule = createTestIRModule();
    
    // 创建IR到LLVM IR转换器
    ai_compiler::backend::IRToLLVMConverter converter;
    
    // 转换IR到LLVM IR
    auto llvmModule = converter.convert(irModule);
    ASSERT_NE(llvmModule, nullptr);
    
    // 创建LLVM优化Pass管理器
    ai_compiler::backend::LLVMPassManager passManager;
    
    // 设置优化级别
    passManager.setOptimizationLevel(2);
    
    // 运行优化Pass
    bool changed = passManager.runOptimizationPasses(llvmModule.get());
    
    // 验证优化Pass执行结果
    // 注意：优化可能不会改变这个简单的模块，所以不断言changed的值
}

// 代码生成测试
TEST(LLVMBackendTest, CodeGenerator) {
    // 创建测试IR模块
    auto irModule = createTestIRModule();
    
    // 创建IR到LLVM IR转换器
    ai_compiler::backend::IRToLLVMConverter converter;
    
    // 转换IR到LLVM IR
    auto llvmModule = converter.convert(irModule);
    ASSERT_NE(llvmModule, nullptr);
    
    // 创建代码生成器
    ai_compiler::backend::CodeGenerator codeGenerator;
    
    // 生成目标代码
    bool success = codeGenerator.generateCode(llvmModule.get(), "/tmp/test_output.o");
    
    // 验证代码生成结果
    EXPECT_TRUE(success);
    
    // 验证输出文件存在
    std::ifstream file("/tmp/test_output.o");
    EXPECT_TRUE(file.good());
}

// LLVM后端集成测试
TEST(LLVMBackendTest, LLVMBackendIntegration) {
    // 创建测试IR模块
    auto irModule = createTestIRModule();
    
    // 创建LLVM后端
    ai_compiler::backend::LLVMBackend backend;
    
    // 设置优化级别
    backend.setOptimizationLevel(2);
    
    // 编译IR模块
    bool success = backend.compile(irModule, "/tmp/test_output_integrated.o");
    
    // 验证编译结果
    EXPECT_TRUE(success);
    
    // 验证输出文件存在
    std::ifstream file("/tmp/test_output_integrated.o");
    EXPECT_TRUE(file.good());
}

// 端到端测试：从DSL到可执行代码
TEST(LLVMBackendTest, EndToEndCompilation) {
    // 解析DSL代码
    std::string input = "func add(a: float, b: float): float { return a + b; }";
    Parser parser(input);
    auto program = parser.parse();
    ASSERT_NE(program, nullptr);
    
    // 转换AST到IR
    ASTToIRConverter irConverter;
    auto irModule = irConverter.convert(program);
    ASSERT_NE(irModule, nullptr);
    
    // 运行优化
    OptimizationPipeline pipeline;
    pipeline.run(irModule);
    
    // 编译IR到可执行代码
    ai_compiler::backend::LLVMBackend backend;
    bool success = backend.compile(irModule, "/tmp/add_function.o");
    
    // 验证编译结果
    EXPECT_TRUE(success);
    
    // 验证输出文件存在
    std::ifstream file("/tmp/add_function.o");
    EXPECT_TRUE(file.good());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
