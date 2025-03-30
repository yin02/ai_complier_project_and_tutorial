#include "../include/llvm_backend.h"
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Utils.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Verifier.h>
#include <iostream>
#include <fstream>
#include <sstream>

namespace ai_compiler {
namespace backend {

//===----------------------------------------------------------------------===//
// IRToLLVMConverter Implementation
//===----------------------------------------------------------------------===//

IRToLLVMConverter::IRToLLVMConverter() : builder(std::make_unique<llvm::IRBuilder<>>(context)) {
    // 初始化LLVM目标
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
}

IRToLLVMConverter::~IRToLLVMConverter() {
}

std::unique_ptr<llvm::Module> IRToLLVMConverter::convert(const std::shared_ptr<ir::Module>& irModule) {
    // 创建新的LLVM模块
    llvmModule = std::make_unique<llvm::Module>(irModule->name, context);
    
    // 清空值映射
    valueMap.clear();
    
    // 创建运行时函数声明
    createRuntimeFunctionDeclarations();
    
    // 转换每个图
    for (const auto& graph : irModule->graphs) {
        convertGraph(graph);
    }
    
    // 验证模块
    std::string errorInfo;
    llvm::raw_string_ostream errorStream(errorInfo);
    if (llvm::verifyModule(*llvmModule, &errorStream)) {
        std::cerr << "LLVM module verification failed: " << errorInfo << std::endl;
        return nullptr;
    }
    
    return std::move(llvmModule);
}

void IRToLLVMConverter::dumpModuleToFile(llvm::Module* module, const std::string& filename) {
    std::error_code EC;
    llvm::raw_fd_ostream dest(filename, EC, llvm::sys::fs::OF_None);
    
    if (EC) {
        std::cerr << "Could not open file: " << EC.message() << std::endl;
        return;
    }
    
    module->print(dest, nullptr);
}

llvm::Function* IRToLLVMConverter::convertGraph(const std::shared_ptr<ir::Graph>& graph) {
    // 创建函数类型
    std::vector<llvm::Type*> paramTypes;
    for (const auto& input : graph->inputs) {
        paramTypes.push_back(convertType(input->type));
    }
    
    llvm::Type* returnType;
    if (graph->outputs.size() == 1) {
        returnType = convertType(graph->outputs[0]->type);
    } else if (graph->outputs.empty()) {
        returnType = llvm::Type::getVoidTy(context);
    } else {
        // 多个返回值，使用结构体
        std::vector<llvm::Type*> returnTypes;
        for (const auto& output : graph->outputs) {
            returnTypes.push_back(convertType(output->type));
        }
        returnType = llvm::StructType::create(context, returnTypes, graph->name + "_return_type");
    }
    
    llvm::FunctionType* funcType = llvm::FunctionType::get(returnType, paramTypes, false);
    
    // 创建函数
    llvm::Function* func = llvm::Function::Create(
        funcType, llvm::Function::ExternalLinkage, graph->name, llvmModule.get());
    
    // 设置参数名称
    unsigned idx = 0;
    for (auto& arg : func->args()) {
        arg.setName(graph->inputs[idx]->name);
        // 将参数添加到值映射
        valueMap[graph->inputs[idx]] = &arg;
        idx++;
    }
    
    // 创建入口基本块
    llvm::BasicBlock* entryBB = llvm::BasicBlock::Create(context, "entry", func);
    builder->SetInsertPoint(entryBB);
    
    // 转换图中的操作
    for (const auto& op : graph->operations) {
        convertOperation(op);
    }
    
    // 创建返回指令
    if (graph->outputs.empty()) {
        builder->CreateRetVoid();
    } else if (graph->outputs.size() == 1) {
        llvm::Value* returnValue = valueMap[graph->outputs[0]];
        builder->CreateRet(returnValue);
    } else {
        // 多个返回值，创建结构体
        llvm::Value* returnStruct = llvm::UndefValue::get(returnType);
        for (size_t i = 0; i < graph->outputs.size(); ++i) {
            llvm::Value* outputValue = valueMap[graph->outputs[i]];
            returnStruct = builder->CreateInsertValue(returnStruct, outputValue, i);
        }
        builder->CreateRet(returnStruct);
    }
    
    return func;
}

llvm::Value* IRToLLVMConverter::convertOperation(const std::shared_ptr<ir::Operation>& op) {
    // 根据操作类型调用相应的转换函数
    if (auto binaryOp = std::dynamic_pointer_cast<ir::BinaryOp>(op)) {
        return convertBinaryOp(binaryOp);
    } else if (auto unaryOp = std::dynamic_pointer_cast<ir::UnaryOp>(op)) {
        return convertUnaryOp(unaryOp);
    } else if (auto constOp = std::dynamic_pointer_cast<ir::ConstantOp>(op)) {
        return convertConstantOp(constOp);
    } else if (auto convOp = std::dynamic_pointer_cast<ir::ConvOp>(op)) {
        return convertConvOp(convOp);
    } else if (auto poolOp = std::dynamic_pointer_cast<ir::PoolOp>(op)) {
        return convertPoolOp(poolOp);
    } else if (auto reshapeOp = std::dynamic_pointer_cast<ir::ReshapeOp>(op)) {
        return convertReshapeOp(reshapeOp);
    } else if (op->opType == "return") {
        // 返回操作在convertGraph中处理
        return nullptr;
    } else {
        std::cerr << "Unsupported operation type: " << op->opType << std::endl;
        return nullptr;
    }
}

llvm::Value* IRToLLVMConverter::convertBinaryOp(const std::shared_ptr<ir::BinaryOp>& binaryOp) {
    if (binaryOp->operands.size() != 2 || binaryOp->results.size() != 1) {
        std::cerr << "Binary operation must have 2 operands and 1 result" << std::endl;
        return nullptr;
    }
    
    llvm::Value* lhs = valueMap[binaryOp->operands[0]];
    llvm::Value* rhs = valueMap[binaryOp->operands[1]];
    
    if (!lhs || !rhs) {
        std::cerr << "Operands not found in value map" << std::endl;
        return nullptr;
    }
    
    llvm::Value* result = nullptr;
    
    // 检查操作数类型
    llvm::Type* lhsType = lhs->getType();
    llvm::Type* rhsType = rhs->getType();
    
    // 如果是标量操作
    if (lhsType->isFloatTy() || lhsType->isDoubleTy()) {
        // 浮点数操作
        switch (binaryOp->kind) {
            case ir::BinaryOp::OpKind::ADD:
                result = builder->CreateFAdd(lhs, rhs, "add");
                break;
            case ir::BinaryOp::OpKind::SUB:
                result = builder->CreateFSub(lhs, rhs, "sub");
                break;
            case ir::BinaryOp::OpKind::MUL:
                result = builder->CreateFMul(lhs, rhs, "mul");
                break;
            case ir::BinaryOp::OpKind::DIV:
                result = builder->CreateFDiv(lhs, rhs, "div");
                break;
            case ir::BinaryOp::OpKind::MATMUL:
                // 标量不支持矩阵乘法
                std::cerr << "Matrix multiplication not supported for scalar types" << std::endl;
                return nullptr;
            default:
                std::cerr << "Unsupported binary operation kind" << std::endl;
                return nullptr;
        }
    } else if (lhsType->isIntegerTy()) {
        // 整数操作
        switch (binaryOp->kind) {
            case ir::BinaryOp::OpKind::ADD:
                result = builder->CreateAdd(lhs, rhs, "add");
                break;
            case ir::BinaryOp::OpKind::SUB:
                result = builder->CreateSub(lhs, rhs, "sub");
                break;
            case ir::BinaryOp::OpKind::MUL:
                result = builder->CreateMul(lhs, rhs, "mul");
                break;
            case ir::BinaryOp::OpKind::DIV:
                result = builder->CreateSDiv(lhs, rhs, "div");
                break;
            case ir::BinaryOp::OpKind::MATMUL:
                // 标量不支持矩阵乘法
                std::cerr << "Matrix multiplication not supported for scalar types" << std::endl;
                return nullptr;
            default:
                std::cerr << "Unsupported binary operation kind" << std::endl;
                return nullptr;
        }
    } else if (lhsType->isPointerTy() && lhsType->getPointerElementType()->isArrayTy()) {
        // 张量操作，调用运行时函数
        std::string funcName;
        switch (binaryOp->kind) {
            case ir::BinaryOp::OpKind::ADD:
                funcName = "tensor_add";
                break;
            case ir::BinaryOp::OpKind::SUB:
                funcName = "tensor_sub";
                break;
            case ir::BinaryOp::OpKind::MUL:
                funcName = "tensor_mul";
                break;
            case ir::BinaryOp::OpKind::DIV:
                funcName = "tensor_div";
                break;
            case ir::BinaryOp::OpKind::MATMUL:
                funcName = "tensor_matmul";
                break;
            default:
                std::cerr << "Unsupported binary operation kind for tensors" << std::endl;
                return nullptr;
        }
        
        // 获取运行时函数
        llvm::Function* runtimeFunc = llvmModule->getFunction(funcName);
        if (!runtimeFunc) {
            std::cerr << "Runtime function not found: " << funcName << std::endl;
            return nullptr;
        }
        
        // 调用运行时函数
        std::vector<llvm::Value*> args = {lhs, rhs};
        result = builder->CreateCall(runtimeFunc, args, "tensor_op");
    } else {
        std::cerr << "Unsupported operand types for binary operation" << std::endl;
        return nullptr;
    }
    
    // 将结果添加到值映射
    valueMap[binaryOp->results[0]] = result;
    
    return result;
}

llvm::Value* IRToLLVMConverter::convertUnaryOp(const std::shared_ptr<ir::UnaryOp>& unaryOp) {
    if (unaryOp->operands.size() != 1 || unaryOp->results.size() != 1) {
        std::cerr << "Unary operation must have 1 operand and 1 result" << std::endl;
        return nullptr;
    }
    
    llvm::Value* operand = valueMap[unaryOp->operands[0]];
    
    if (!operand) {
        std::cerr << "Operand not found in value map" << std::endl;
        return nullptr;
    }
    
    llvm::Value* result = nullptr;
    
    // 检查操作数类型
    llvm::Type* operandType = operand->getType();
    
    // 如果是标量操作
    if (operandType->isFloatTy() || operandType->isDoubleTy()) {
        // 浮点数操作
        switch (unaryOp->kind) {
            case ir::UnaryOp::OpKind::NEG:
                result = builder->CreateFNeg(operand, "neg");
                break;
            case ir::UnaryOp::OpKind::RELU:
            case ir::UnaryOp::OpKind::SIGMOID:
            case ir::UnaryOp::OpKind::TANH:
            case ir::UnaryOp::OpKind::SOFTMAX:
                // 这些激活函数需要调用运行时函数
                std::cerr << "Activation functions not implemented for scalar types" << std::endl;
                return nullptr;
            default:
                std::cerr << "Unsupported unary operation kind" << std::endl;
                return nullptr;
        }
    } else if (operandType->isIntegerTy()) {
        // 整数操作
        switch (unaryOp->kind) {
            case ir::UnaryOp::OpKind::NEG:
                result = builder->CreateNeg(operand, "neg");
                break;
            case ir::UnaryOp::OpKind::RELU:
            case ir::UnaryOp::OpKind::SIGMOID:
            case ir::UnaryOp::OpKind::TANH:
            case ir::UnaryOp::OpKind::SOFTMAX:
                // 这些激活函数需要调用运行时函数
                std::cerr << "Activation functions not implemented for scalar types" << std::endl;
                return nullptr;
            default:
                std::cerr << "Unsupported unary operation kind" << std::endl;
                return nullptr;
        }
    } else if (operandType->isPointerTy() && operandType->getPointerElementType()->isArrayTy()) {
        // 张量操作，调用运行时函数
        std::string funcName;
        switch (unaryOp->kind) {
            case ir::UnaryOp::OpKind::NEG:
                funcName = "tensor_neg";
                break;
            case ir::UnaryOp::OpKind::RELU:
                funcName = "tensor_relu";
                break;
            case ir::UnaryOp::OpKind::SIGMOID:
                funcName = "tensor_sigmoid";
                break;
            case ir::UnaryOp::OpKind::TANH:
                funcName = "tensor_tanh";
                break;
            case ir::UnaryOp::OpKind::SOFTMAX:
                funcName = "tensor_softmax";
                break;
            default:
                std::cerr << "Unsupported unary operation kind for tensors" << std::endl;
                return nullptr;
        }
        
        // 获取运行时函数
        llvm::Function* runtimeFunc = llvmModule->getFunction(funcName);
        if (!runtimeFunc) {
            std::cerr << "Runtime function not found: " << funcName << std::endl;
            return nullptr;
        }
        
        // 调用运行时函数
        std::vector<llvm::Value*> args = {operand};
        result = builder->CreateCall(runtimeFunc, args, "tensor_op");
    } else {
        std::cerr << "Unsupported operand type for unary operation" << std::endl;
        return nullptr;
    }
    
    // 将结果添加到值映射
    valueMap[unaryOp->results[0]] = result;
    
    return result;
}

llvm::Value* IRToLLVMConverter::convertConstantOp(const std::shared_ptr<ir::ConstantOp>& constOp) {
    if (constOp->results.size() != 1) {
        std::cerr << "Constant operation must have 1 result" << std::endl;
        return nullptr;
    }
    
    std::shared_ptr<ir::Value> resultValue = constOp->results[0];
    std::string valueStr = constOp->getAttribute("value");
    
    llvm::Value* result = nullptr;
    
    // 根据结果类型创建常量
    if (auto scalarType = std::dynamic_pointer_cast<ir::ScalarType>(resultValue->type)) {
        switch (scalarType->dataType) {
            case ir::DataType::FLOAT32: {
                float value = std::stof(valueStr);
                result = llvm::ConstantFP::get(llvm::Type::getFloatTy(context), value);
                break;
            }
            case ir::DataType::INT32: {
                int value = std::stoi(valueStr);
                result = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), value);
                break;
            }
            case ir::DataType::BOOL: {
                bool value = (valueStr == "true");
                result = llvm::ConstantInt::get(llvm::Type::getInt1Ty(context), value);
                break;
            }
            default:
                std::cerr << "Unsupported scalar type for constant" << std::endl;
                return nullptr;
        }
    } else if (auto tensorType = std::dynamic_pointer_cast<ir::TensorType>(resultValue->type)) {
        // 张量常量需要调用运行时函数
        std::string funcName = "create_tensor_constant";
        
        // 获取运行时函数
        llvm::Function* runtimeFunc = llvmModule->getFunction(funcName);
        if (!runtimeFunc) {
            std::cerr << "Runtime function not found: " << funcName << std::endl;
            return nullptr;
        }
        
        // 创建常量字符串
        llvm::Value* valueStrVal = builder->CreateGlobalStringPtr(valueStr, "const_str");
        
        // 调用运行时函数
        std::vector<llvm::Value*> args = {valueStrVal};
        result = builder->CreateCall(runtimeFunc, args, "tensor_constant");
    } else {
        std::cerr << "Unsupported type for constant" << std::endl;
        return nullptr;
    }
    
    // 将结果添加到值映射
    valueMap[resultValue] = result;
    
    return result;
}

llvm::Value* IRToLLVMConverter::convertConvOp(const std::shared_ptr<ir::ConvOp>& convOp) {
    if (convOp->operands.size() != 2 || convOp->results.size() != 1) {
        std::cerr << "Convolution operation must have 2 operands and 1 result" << std::endl;
        return nullptr;
    }
    
    llvm::Value* input = valueMap[convOp->operands[0]];
    llvm::Value* filter = valueMap[convOp->operands[1]];
    
    if (!input || !filter) {
        std::cerr << "Operands not found in value map" << std::endl;
        return nullptr;
    }
    
    // 获取卷积参数
    std::string strideStr = convOp->getAttribute("stride");
    std::string paddingStr = convOp->getAttribute("padding");
    
    // 创建参数值
    llvm::Value* strideVal = builder->CreateGlobalStringPtr(strideStr, "stride_str");
    llvm::Value* paddingVal = builder->CreateGlobalStringPtr(paddingStr, "padding_str");
    
    // 获取运行时函数
    llvm::Function* runtimeFunc = llvmModule->getFunction("tensor_conv2d");
    if (!runtimeFunc) {
        std::cerr << "Runtime function not found: tensor_conv2d" << std::endl;
        return nullptr;
    }
    
    // 调用运行时函数
    std::vector<llvm::Value*> args = {input, filter, strideVal, paddingVal};
    llvm::Value* result = builder->CreateCall(runtimeFunc, args, "conv2d_result");
    
    // 将结果添加到值映射
    valueMap[convOp->results[0]] = result;
    
    return result;
}

llvm::Value* IRToLLVMConverter::convertPoolOp(const std::shared_ptr<ir::PoolOp>& poolOp) {
    if (poolOp->operands.size() != 1 || poolOp->results.size() != 1) {
        std::cerr << "Pooling operation must have 1 operand and 1 result" << std::endl;
        return nullptr;
    }
    
    llvm::Value* input = valueMap[poolOp->operands[0]];
    
    if (!input) {
        std::cerr << "Operand not found in value map" << std::endl;
        return nullptr;
    }
    
    // 获取池化参数
    std::string kernelSizeStr = poolOp->getAttribute("kernel_size");
    std::string strideStr = poolOp->getAttribute("stride");
    std::string paddingStr = poolOp->getAttribute("padding");
    
    // 创建参数值
    llvm::Value* kernelSizeVal = builder->CreateGlobalStringPtr(kernelSizeStr, "kernel_size_str");
    llvm::Value* strideVal = builder->CreateGlobalStringPtr(strideStr, "stride_str");
    llvm::Value* paddingVal = builder->CreateGlobalStringPtr(paddingStr, "padding_str");
    
    // 获取运行时函数
    std::string funcName;
    switch (poolOp->kind) {
        case ir::PoolOp::PoolKind::MAX:
            funcName = "tensor_max_pool";
            break;
        case ir::PoolOp::PoolKind::AVG:
            funcName = "tensor_avg_pool";
            break;
        default:
            std::cerr << "Unsupported pooling kind" << std::endl;
            return nullptr;
    }
    
    llvm::Function* runtimeFunc = llvmModule->getFunction(funcName);
    if (!runtimeFunc) {
        std::cerr << "Runtime function not found: " << funcName << std::endl;
        return nullptr;
    }
    
    // 调用运行时函数
    std::vector<llvm::Value*> args = {input, kernelSizeVal, strideVal, paddingVal};
    llvm::Value* result = builder->CreateCall(runtimeFunc, args, "pool_result");
    
    // 将结果添加到值映射
    valueMap[poolOp->results[0]] = result;
    
    return result;
}

llvm::Value* IRToLLVMConverter::convertReshapeOp(const std::shared_ptr<ir::ReshapeOp>& reshapeOp) {
    if (reshapeOp->operands.size() != 1 || reshapeOp->results.size() != 1) {
        std::cerr << "Reshape operation must have 1 operand and 1 result" << std::endl;
        return nullptr;
    }
    
    llvm::Value* input = valueMap[reshapeOp->operands[0]];
    
    if (!input) {
        std::cerr << "Operand not found in value map" << std::endl;
        return nullptr;
    }
    
    // 获取新形状
    std::string newShapeStr = reshapeOp->getAttribute("new_shape");
    
    // 创建参数值
    llvm::Value* newShapeVal = builder->CreateGlobalStringPtr(newShapeStr, "new_shape_str");
    
    // 获取运行时函数
    llvm::Function* runtimeFunc = llvmModule->getFunction("tensor_reshape");
    if (!runtimeFunc) {
        std::cerr << "Runtime function not found: tensor_reshape" << std::endl;
        return nullptr;
    }
    
    // 调用运行时函数
    std::vector<llvm::Value*> args = {input, newShapeVal};
    llvm::Value* result = builder->CreateCall(runtimeFunc, args, "reshape_result");
    
    // 将结果添加到值映射
    valueMap[reshapeOp->results[0]] = result;
    
    return result;
}

llvm::Type* IRToLLVMConverter::convertType(const std::shared_ptr<ir::Type>& type) {
    if (auto scalarType = std::dynamic_pointer_cast<ir::ScalarType>(type)) {
        switch (scalarType->dataType) {
            case ir::DataType::FLOAT32:
                return llvm::Type::getFloatTy(context);
            case ir::DataType::INT32:
                return llvm::Type::getInt32Ty(context);
            case ir::DataType::BOOL:
                return llvm::Type::getInt1Ty(context);
            default:
                std::cerr << "Unsupported scalar type" << std::endl;
                return llvm::Type::getVoidTy(context);
        }
    } else if (auto tensorType = std::dynamic_pointer_cast<ir::TensorType>(type)) {
        // 张量类型表示为指向运行时张量对象的指针
        return llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0);
    } else {
        std::cerr << "Unsupported type" << std::endl;
        return llvm::Type::getVoidTy(context);
    }
}

void IRToLLVMConverter::createRuntimeFunctionDeclarations() {
    // 张量操作函数
    
    // 二元操作
    getOrCreateRuntimeFunction(
        "tensor_add",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    getOrCreateRuntimeFunction(
        "tensor_sub",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    getOrCreateRuntimeFunction(
        "tensor_mul",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    getOrCreateRuntimeFunction(
        "tensor_div",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    getOrCreateRuntimeFunction(
        "tensor_matmul",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    // 一元操作
    getOrCreateRuntimeFunction(
        "tensor_neg",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    getOrCreateRuntimeFunction(
        "tensor_relu",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    getOrCreateRuntimeFunction(
        "tensor_sigmoid",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    getOrCreateRuntimeFunction(
        "tensor_tanh",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    getOrCreateRuntimeFunction(
        "tensor_softmax",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    // 卷积和池化
    getOrCreateRuntimeFunction(
        "tensor_conv2d",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    getOrCreateRuntimeFunction(
        "tensor_max_pool",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    getOrCreateRuntimeFunction(
        "tensor_avg_pool",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    // 形状操作
    getOrCreateRuntimeFunction(
        "tensor_reshape",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
         llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
    
    // 常量创建
    getOrCreateRuntimeFunction(
        "create_tensor_constant",
        llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0),
        {llvm::PointerType::get(llvm::Type::getInt8Ty(context), 0)}
    );
}

llvm::Function* IRToLLVMConverter::getOrCreateRuntimeFunction(
    const std::string& name, llvm::Type* returnType, const std::vector<llvm::Type*>& paramTypes) {
    llvm::Function* func = llvmModule->getFunction(name);
    
    if (!func) {
        llvm::FunctionType* funcType = llvm::FunctionType::get(returnType, paramTypes, false);
        func = llvm::Function::Create(funcType, llvm::Function::ExternalLinkage, name, llvmModule.get());
        func->setCallingConv(llvm::CallingConv::C);
    }
    
    return func;
}

//===----------------------------------------------------------------------===//
// LLVMPassManager Implementation
//===----------------------------------------------------------------------===//

LLVMPassManager::LLVMPassManager() : optimizationLevel(2) {
}

bool LLVMPassManager::runOptimizationPasses(llvm::Module* module) {
    // 创建Pass管理器
    llvm::legacy::PassManager passManager;
    
    // 创建Pass管理器构建器
    llvm::PassManagerBuilder passManagerBuilder;
    passManagerBuilder.OptLevel = optimizationLevel;
    
    // 根据优化级别设置内联阈值
    if (optimizationLevel > 1) {
        passManagerBuilder.Inliner = llvm::createFunctionInliningPass();
    }
    
    // 添加标准Pass
    passManagerBuilder.populateModulePassManager(passManager);
    
    // 添加其他Pass
    if (optimizationLevel > 0) {
        // 基本优化
        passManager.add(llvm::createPromoteMemoryToRegisterPass());
        passManager.add(llvm::createInstructionCombiningPass());
        passManager.add(llvm::createReassociatePass());
        passManager.add(llvm::createGVNPass());
        passManager.add(llvm::createCFGSimplificationPass());
    }
    
    if (optimizationLevel > 1) {
        // 高级优化
        passManager.add(llvm::createSROAPass());
        passManager.add(llvm::createEarlyCSEPass());
        passManager.add(llvm::createLICMPass());
        passManager.add(llvm::createAggressiveDCEPass());
        passManager.add(llvm::createCFGSimplificationPass());
    }
    
    if (optimizationLevel > 2) {
        // 更激进的优化
        passManager.add(llvm::createFunctionInliningPass());
        passManager.add(llvm::createArgumentPromotionPass());
        passManager.add(llvm::createTailCallEliminationPass());
        passManager.add(llvm::createJumpThreadingPass());
        passManager.add(llvm::createCFGSimplificationPass());
    }
    
    // 运行Pass
    return passManager.run(*module);
}

void LLVMPassManager::setOptimizationLevel(int level) {
    optimizationLevel = level;
}

//===----------------------------------------------------------------------===//
// CodeGenerator Implementation
//===----------------------------------------------------------------------===//

CodeGenerator::CodeGenerator() : targetTriple(llvm::sys::getDefaultTargetTriple()), cpuFeatures("") {
}

bool CodeGenerator::generateCode(llvm::Module* module, const std::string& outputFilename) {
    // 初始化目标
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
    
    // 设置目标三元组
    module->setTargetTriple(targetTriple);
    
    // 获取目标
    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    
    if (!target) {
        std::cerr << "Error looking up target: " << error << std::endl;
        return false;
    }
    
    // 创建目标机器
    llvm::TargetOptions opt;
    auto RM = llvm::Optional<llvm::Reloc::Model>();
    auto targetMachine = target->createTargetMachine(
        targetTriple, "generic", cpuFeatures, opt, RM);
    
    // 设置数据布局
    module->setDataLayout(targetMachine->createDataLayout());
    
    // 创建输出文件
    std::error_code EC;
    llvm::raw_fd_ostream dest(outputFilename, EC, llvm::sys::fs::OF_None);
    
    if (EC) {
        std::cerr << "Could not open file: " << EC.message() << std::endl;
        return false;
    }
    
    // 创建Pass管理器
    llvm::legacy::PassManager pass;
    
    // 设置输出文件类型
    auto fileType = llvm::CGFT_ObjectFile;
    
    // 添加目标代码生成Pass
    if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
        std::cerr << "Target machine can't emit a file of this type" << std::endl;
        return false;
    }
    
    // 运行Pass
    pass.run(*module);
    dest.flush();
    
    return true;
}

void CodeGenerator::setTargetTriple(const std::string& triple) {
    targetTriple = triple;
}

void CodeGenerator::setCPUFeatures(const std::string& features) {
    cpuFeatures = features;
}

//===----------------------------------------------------------------------===//
// LLVMBackend Implementation
//===----------------------------------------------------------------------===//

LLVMBackend::LLVMBackend() {
}

bool LLVMBackend::compile(const std::shared_ptr<ir::Module>& irModule, const std::string& outputFilename) {
    // 转换IR到LLVM IR
    std::unique_ptr<llvm::Module> llvmModule = converter.convert(irModule);
    if (!llvmModule) {
        std::cerr << "Failed to convert IR to LLVM IR" << std::endl;
        return false;
    }
    
    // 运行优化Pass
    if (!passManager.runOptimizationPasses(llvmModule.get())) {
        std::cerr << "Failed to run optimization passes" << std::endl;
        return false;
    }
    
    // 生成目标代码
    if (!codeGenerator.generateCode(llvmModule.get(), outputFilename)) {
        std::cerr << "Failed to generate code" << std::endl;
        return false;
    }
    
    return true;
}

void LLVMBackend::setOptimizationLevel(int level) {
    passManager.setOptimizationLevel(level);
}

void LLVMBackend::setTargetTriple(const std::string& triple) {
    codeGenerator.setTargetTriple(triple);
}

void LLVMBackend::setCPUFeatures(const std::string& features) {
    codeGenerator.setCPUFeatures(features);
}

} // namespace backend
} // namespace ai_compiler
