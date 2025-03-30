#ifndef LLVM_BACKEND_H
#define LLVM_BACKEND_H

#include "../../ir/include/ir.h"
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <unordered_map>
#include <string>

namespace ai_compiler {
namespace backend {

// IR到LLVM IR的转换器类
class IRToLLVMConverter {
public:
    IRToLLVMConverter();
    ~IRToLLVMConverter();
    
    // 将IR模块转换为LLVM模块
    std::unique_ptr<llvm::Module> convert(const std::shared_ptr<ir::Module>& irModule);
    
    // 获取LLVM上下文
    llvm::LLVMContext& getContext() { return context; }
    
    // 将LLVM模块转储到文件
    void dumpModuleToFile(llvm::Module* module, const std::string& filename);
    
private:
    // LLVM上下文
    llvm::LLVMContext context;
    
    // 当前LLVM模块
    std::unique_ptr<llvm::Module> llvmModule;
    
    // IR Builder
    std::unique_ptr<llvm::IRBuilder<>> builder;
    
    // 符号表：IR值到LLVM值的映射
    std::unordered_map<std::shared_ptr<ir::Value>, llvm::Value*> valueMap;
    
    // 转换IR图到LLVM函数
    llvm::Function* convertGraph(const std::shared_ptr<ir::Graph>& graph);
    
    // 转换IR操作到LLVM指令
    llvm::Value* convertOperation(const std::shared_ptr<ir::Operation>& op);
    
    // 转换特定类型的操作
    llvm::Value* convertBinaryOp(const std::shared_ptr<ir::BinaryOp>& binaryOp);
    llvm::Value* convertUnaryOp(const std::shared_ptr<ir::UnaryOp>& unaryOp);
    llvm::Value* convertConstantOp(const std::shared_ptr<ir::ConstantOp>& constOp);
    llvm::Value* convertConvOp(const std::shared_ptr<ir::ConvOp>& convOp);
    llvm::Value* convertPoolOp(const std::shared_ptr<ir::PoolOp>& poolOp);
    llvm::Value* convertReshapeOp(const std::shared_ptr<ir::ReshapeOp>& reshapeOp);
    
    // 转换IR类型到LLVM类型
    llvm::Type* convertType(const std::shared_ptr<ir::Type>& type);
    
    // 创建运行时函数声明
    void createRuntimeFunctionDeclarations();
    
    // 获取或创建运行时函数
    llvm::Function* getOrCreateRuntimeFunction(const std::string& name, 
                                              llvm::Type* returnType,
                                              const std::vector<llvm::Type*>& paramTypes);
};

// LLVM优化Pass管理器
class LLVMPassManager {
public:
    LLVMPassManager();
    
    // 运行优化Pass
    bool runOptimizationPasses(llvm::Module* module);
    
    // 设置优化级别
    void setOptimizationLevel(int level);
    
private:
    int optimizationLevel;
};

// 代码生成器类
class CodeGenerator {
public:
    CodeGenerator();
    
    // 生成目标代码
    bool generateCode(llvm::Module* module, const std::string& outputFilename);
    
    // 设置目标三元组
    void setTargetTriple(const std::string& triple);
    
    // 设置CPU特性
    void setCPUFeatures(const std::string& features);
    
private:
    std::string targetTriple;
    std::string cpuFeatures;
};

// LLVM后端类
class LLVMBackend {
public:
    LLVMBackend();
    
    // 编译IR模块到目标代码
    bool compile(const std::shared_ptr<ir::Module>& irModule, const std::string& outputFilename);
    
    // 设置优化级别
    void setOptimizationLevel(int level);
    
    // 设置目标三元组
    void setTargetTriple(const std::string& triple);
    
    // 设置CPU特性
    void setCPUFeatures(const std::string& features);
    
private:
    IRToLLVMConverter converter;
    LLVMPassManager passManager;
    CodeGenerator codeGenerator;
};

} // namespace backend
} // namespace ai_compiler

#endif // LLVM_BACKEND_H
