#include "include/ast_to_ir.h"
#include <iostream>
#include <sstream>

namespace ai_compiler {

ASTToIRConverter::ASTToIRConverter() {}

std::shared_ptr<ir::Module> ASTToIRConverter::convert(const std::shared_ptr<Program>& program) {
    // 创建新的IR模块
    currentModule = std::make_shared<ir::Module>("main_module");
    
    // 清空符号表
    symbolTable.clear();
    
    // 转换程序中的每个语句
    for (const auto& stmt : program->statements) {
        convertStatement(stmt);
    }
    
    return currentModule;
}

std::shared_ptr<ir::Type> ASTToIRConverter::convertType(const std::shared_ptr<Type>& astType) {
    if (auto basicType = std::dynamic_pointer_cast<BasicType>(astType)) {
        switch (basicType->kind) {
            case Type::TypeKind::INT:
                return std::make_shared<ir::ScalarType>(ir::DataType::INT32);
            case Type::TypeKind::FLOAT:
                return std::make_shared<ir::ScalarType>(ir::DataType::FLOAT32);
            case Type::TypeKind::BOOL:
                return std::make_shared<ir::ScalarType>(ir::DataType::BOOL);
            default:
                return std::make_shared<ir::ScalarType>(ir::DataType::UNKNOWN);
        }
    } else if (auto tensorType = std::dynamic_pointer_cast<TensorType>(astType)) {
        ir::DataType elementDataType;
        
        auto elementAstType = tensorType->elementType;
        if (auto basicElementType = std::dynamic_pointer_cast<BasicType>(elementAstType)) {
            switch (basicElementType->kind) {
                case Type::TypeKind::INT:
                    elementDataType = ir::DataType::INT32;
                    break;
                case Type::TypeKind::FLOAT:
                    elementDataType = ir::DataType::FLOAT32;
                    break;
                case Type::TypeKind::BOOL:
                    elementDataType = ir::DataType::BOOL;
                    break;
                default:
                    elementDataType = ir::DataType::UNKNOWN;
                    break;
            }
        } else {
            elementDataType = ir::DataType::UNKNOWN;
        }
        
        std::vector<int64_t> shape;
        for (int dim : tensorType->shape) {
            shape.push_back(static_cast<int64_t>(dim));
        }
        
        return std::make_shared<ir::TensorType>(elementDataType, shape);
    }
    
    // 默认返回未知类型
    return std::make_shared<ir::ScalarType>(ir::DataType::UNKNOWN);
}

void ASTToIRConverter::convertStatement(const std::shared_ptr<Statement>& stmt) {
    if (auto varDecl = std::dynamic_pointer_cast<VarDeclaration>(stmt)) {
        convertVarDeclaration(varDecl);
    } else if (auto funcDecl = std::dynamic_pointer_cast<FunctionDeclaration>(stmt)) {
        convertFunctionDeclaration(funcDecl);
    } else if (auto opDecl = std::dynamic_pointer_cast<OperationDeclaration>(stmt)) {
        convertOperationDeclaration(opDecl);
    } else if (auto graphDecl = std::dynamic_pointer_cast<GraphDeclaration>(stmt)) {
        convertGraphDeclaration(graphDecl);
    } else if (auto ifStmt = std::dynamic_pointer_cast<IfStatement>(stmt)) {
        convertIfStatement(ifStmt);
    } else if (auto forStmt = std::dynamic_pointer_cast<ForStatement>(stmt)) {
        convertForStatement(forStmt);
    } else if (auto returnStmt = std::dynamic_pointer_cast<ReturnStatement>(stmt)) {
        convertReturnStatement(returnStmt);
    } else if (auto block = std::dynamic_pointer_cast<Block>(stmt)) {
        convertBlock(block);
    } else if (auto exprStmt = std::dynamic_pointer_cast<ExpressionStmt>(stmt)) {
        convertExpressionStatement(exprStmt);
    }
}

void ASTToIRConverter::convertVarDeclaration(const std::shared_ptr<VarDeclaration>& varDecl) {
    // 转换变量类型
    std::shared_ptr<ir::Type> irType = convertType(varDecl->type);
    
    // 创建变量操作
    auto variableOp = std::make_shared<ir::VariableOp>();
    
    // 创建变量值
    std::shared_ptr<ir::Value> value = createValue(varDecl->name, irType);
    
    // 将变量操作添加到当前图
    if (currentGraph) {
        variableOp->addResult(value);
        currentGraph->addOperation(variableOp);
    }
    
    // 如果有初始化表达式，则转换它
    if (varDecl->initializer) {
        std::shared_ptr<ir::Value> initValue = convertExpression(varDecl->initializer);
        
        // 创建赋值操作
        auto assignOp = createOperation("assign");
        assignOp->addOperand(initValue);
        assignOp->addResult(value);
        
        // 将赋值操作添加到当前图
        if (currentGraph) {
            currentGraph->addOperation(assignOp);
        }
    }
    
    // 将变量添加到符号表
    symbolTable[varDecl->name] = value;
}

void ASTToIRConverter::convertFunctionDeclaration(const std::shared_ptr<FunctionDeclaration>& funcDecl) {
    // 创建新的图
    auto graph = std::make_shared<ir::Graph>(funcDecl->name);
    
    // 保存当前图
    auto savedGraph = currentGraph;
    currentGraph = graph;
    
    // 保存当前符号表
    auto savedSymbolTable = symbolTable;
    symbolTable.clear();
    
    // 处理函数参数
    for (const auto& param : funcDecl->parameters) {
        std::shared_ptr<ir::Type> paramType = convertType(param.type);
        std::shared_ptr<ir::Value> paramValue = createValue(param.name, paramType);
        
        // 将参数添加到图的输入
        graph->addInput(paramValue);
        
        // 将参数添加到符号表
        symbolTable[param.name] = paramValue;
    }
    
    // 转换函数体
    convertBlock(funcDecl->body);
    
    // 将图添加到模块
    currentModule->addGraph(graph);
    
    // 恢复当前图和符号表
    currentGraph = savedGraph;
    symbolTable = savedSymbolTable;
}

void ASTToIRConverter::convertOperationDeclaration(const std::shared_ptr<OperationDeclaration>& opDecl) {
    // 操作声明类似于函数声明
    convertFunctionDeclaration(std::make_shared<FunctionDeclaration>(
        opDecl->name, opDecl->parameters, opDecl->returnType, opDecl->body));
}

void ASTToIRConverter::convertGraphDeclaration(const std::shared_ptr<GraphDeclaration>& graphDecl) {
    // 创建新的图
    auto graph = std::make_shared<ir::Graph>(graphDecl->name);
    
    // 保存当前图
    auto savedGraph = currentGraph;
    currentGraph = graph;
    
    // 保存当前符号表
    auto savedSymbolTable = symbolTable;
    symbolTable.clear();
    
    // 处理图参数
    for (const auto& param : graphDecl->parameters) {
        std::shared_ptr<ir::Type> paramType = convertType(param.type);
        std::shared_ptr<ir::Value> paramValue = createValue(param.name, paramType);
        
        // 将参数添加到图的输入
        graph->addInput(paramValue);
        
        // 将参数添加到符号表
        symbolTable[param.name] = paramValue;
    }
    
    // 转换图体
    convertBlock(graphDecl->body);
    
    // 将图添加到模块
    currentModule->addGraph(graph);
    
    // 恢复当前图和符号表
    currentGraph = savedGraph;
    symbolTable = savedSymbolTable;
}

void ASTToIRConverter::convertBlock(const std::shared_ptr<Block>& block) {
    // 转换块中的每个语句
    for (const auto& stmt : block->statements) {
        convertStatement(stmt);
    }
}

void ASTToIRConverter::convertIfStatement(const std::shared_ptr<IfStatement>& ifStmt) {
    // 转换条件表达式
    std::shared_ptr<ir::Value> condition = convertExpression(ifStmt->condition);
    
    // 创建条件操作
    auto condOp = createOperation("if");
    condOp->addOperand(condition);
    
    // 保存当前符号表
    auto savedSymbolTable = symbolTable;
    
    // 转换 then 分支
    convertStatement(ifStmt->thenBranch);
    
    // 如果有 else 分支，则转换它
    if (ifStmt->elseBranch) {
        // 恢复符号表
        symbolTable = savedSymbolTable;
        
        // 转换 else 分支
        convertStatement(ifStmt->elseBranch);
    }
    
    // 恢复符号表
    symbolTable = savedSymbolTable;
}

void ASTToIRConverter::convertForStatement(const std::shared_ptr<ForStatement>& forStmt) {
    // 保存当前符号表
    auto savedSymbolTable = symbolTable;
    
    // 转换初始化语句
    if (forStmt->initializer) {
        convertStatement(forStmt->initializer);
    }
    
    // 转换条件表达式
    std::shared_ptr<ir::Value> condition = nullptr;
    if (forStmt->condition) {
        condition = convertExpression(forStmt->condition);
    }
    
    // 创建循环操作
    auto loopOp = createOperation("for");
    if (condition) {
        loopOp->addOperand(condition);
    }
    
    // 转换循环体
    convertStatement(forStmt->body);
    
    // 转换增量表达式
    if (forStmt->increment) {
        convertExpression(forStmt->increment);
    }
    
    // 恢复符号表
    symbolTable = savedSymbolTable;
}

void ASTToIRConverter::convertReturnStatement(const std::shared_ptr<ReturnStatement>& returnStmt) {
    // 转换返回值表达式
    std::shared_ptr<ir::Value> returnValue = nullptr;
    if (returnStmt->value) {
        returnValue = convertExpression(returnStmt->value);
    }
    
    // 创建返回操作
    auto returnOp = std::make_shared<ir::ReturnOp>();
    if (returnValue) {
        returnOp->addOperand(returnValue);
        
        // 将返回值添加到图的输出
        if (currentGraph) {
            currentGraph->addOutput(returnValue);
        }
    }
    
    // 将返回操作添加到当前图
    if (currentGraph) {
        currentGraph->addOperation(returnOp);
    }
}

void ASTToIRConverter::convertExpressionStatement(const std::shared_ptr<ExpressionStmt>& exprStmt) {
    // 转换表达式
    convertExpression(exprStmt->expression);
}

std::shared_ptr<ir::Value> ASTToIRConverter::convertExpression(const std::shared_ptr<Expression>& expr) {
    if (auto binExpr = std::dynamic_pointer_cast<BinaryExpr>(expr)) {
        return convertBinaryExpr(binExpr);
    } else if (auto unaryExpr = std::dynamic_pointer_cast<UnaryExpr>(expr)) {
        return convertUnaryExpr(unaryExpr);
    } else if (auto callExpr = std::dynamic_pointer_cast<CallExpr>(expr)) {
        return convertCallExpr(callExpr);
    } else if (auto var = std::dynamic_pointer_cast<Variable>(expr)) {
        return convertVariable(var);
    } else if (auto literal = std::dynamic_pointer_cast<Literal>(expr)) {
        return convertLiteral(literal);
    }
    
    // 默认返回空
    return nullptr;
}

std::shared_ptr<ir::Value> ASTToIRConverter::convertBinaryExpr(const std::shared_ptr<BinaryExpr>& binExpr) {
    // 转换左右操作数
    std::shared_ptr<ir::Value> left = convertExpression(binExpr->left);
    std::shared_ptr<ir::Value> right = convertExpression(binExpr->right);
    
    // 确定二元操作类型
    ir::BinaryOp::OpKind opKind;
    switch (binExpr->op) {
        case BinaryExpr::Op::ADD: opKind = ir::BinaryOp::OpKind::ADD; break;
        case BinaryExpr::Op::SUB: opKind = ir::BinaryOp::OpKind::SUB; break;
        case BinaryExpr::Op::MUL: opKind = ir::BinaryOp::OpKind::MUL; break;
        case BinaryExpr::Op::DIV: opKind = ir::BinaryOp::OpKind::DIV; break;
        default: opKind = ir::BinaryOp::OpKind::ADD; break; // 默认为加法
    }
    
    // 创建二元操作
    auto binaryOp = std::make_shared<ir::BinaryOp>(opKind);
    binaryOp->addOperand(left);
    binaryOp->addOperand(right);
    
    // 创建结果值
    std::shared_ptr<ir::Type> resultType = left->type; // 简化：假设结果类型与左操作数相同
    std::shared_ptr<ir::Value> result = createValue(generateUniqueName("binary_result"), resultType);
    binaryOp->addResult(result);
    
    // 将操作添加到当前图
    if (currentGraph) {
        currentGraph->addOperation(binaryOp);
    }
    
    return result;
}

std::shared_ptr<ir::Value> ASTToIRConverter::convertUnaryExpr(const std::shared_ptr<UnaryExpr>& unaryExpr) {
    // 转换操作数
    std::shared_ptr<ir::Value> operand = convertExpression(unaryExpr->expr);
    
    // 确定一元操作类型
    ir::UnaryOp::OpKind opKind;
    switch (unaryExpr->op) {
        case UnaryExpr::Op::NEG: opKind = ir::UnaryOp::OpKind::NEG; break;
        case UnaryExpr::Op::NOT: opKind = ir::UnaryOp::OpKind::NEG; break; // 简化：使用 NEG 代替 NOT
        default: opKind = ir::UnaryOp::OpKind::NEG; break;
    }
    
    // 创建一元操作
    auto unaryOp = std::make_shared<ir::UnaryOp>(opKind);
    unaryOp->addOperand(operand);
    
    // 创建结果值
    std::shared_ptr<ir::Type> resultType = operand->type;
    std::shared_ptr<ir::Value> result = createValue(generateUniqueName("unary_result"), resultType);
    unaryOp->addResult(result);
    
    // 将操作添加到当前图
    if (currentGraph) {
        currentGraph->addOperation(unaryOp);
    }
    
    return result;
}

std::shared_ptr<ir::Value> ASTToIRConverter::convertCallExpr(const std::shared_ptr<CallExpr>& callExpr) {
    // 转换参数
    std::vector<std::shared_ptr<ir::Value>> args;
    for (const auto& arg : callExpr->arguments) {
        args.push_back(convertExpression(arg));
    }
    
    // 创建调用操作
    auto callOp = createOperation(callExpr->callee);
    for (const auto& arg : args) {
        callOp->addOperand(arg);
    }
    
    // 创建结果值
    std::shared_ptr<ir::Type> resultType = convertType(callExpr->getType());
    std::shared_ptr<ir::Value> result = createValue(generateUniqueName("call_result"), resultType);
    callOp->addResult(result);
    
    // 将操作添加到当前图
    if (currentGraph) {
        currentGraph->addOperation(callOp);
    }
    
    return result;
}

std::shared_ptr<ir::Value> ASTToIRConverter::convertVariable(const std::shared_ptr<Variable>& var) {
    // 查找变量在符号表中的值
    auto it = symbolTable.find(var->name);
    if (it != symbolTable.end()) {
        return it->second;
    }
    
    // 如果变量不在符号表中，创建一个新的值
    std::shared_ptr<ir::Type> type = convertType(var->getType());
    std::shared_ptr<ir::Value> value = createValue(var->name, type);
    
    // 将变量添加到符号表
    symbolTable[var->name] = value;
    
    return value;
}

std::shared_ptr<ir::Value> ASTToIRConverter::convertLiteral(const std::shared_ptr<Literal>& literal) {
    if (auto intLiteral = std::dynamic_pointer_cast<IntLiteral>(literal)) {
        // 创建常量操作
        auto constantOp = std::make_shared<ir::ConstantOp>();
        
        // 创建结果值
        std::shared_ptr<ir::Type> type = std::make_shared<ir::ScalarType>(ir::DataType::INT32);
        std::shared_ptr<ir::Value> value = createValue(generateUniqueName("int_literal"), type);
        constantOp->addResult(value);
        
        // 设置常量值
        constantOp->setAttribute("value", std::to_string(intLiteral->value));
        
        // 将操作添加到当前图
        if (currentGraph) {
            currentGraph->addOperation(constantOp);
        }
        
        return value;
    } else if (auto floatLiteral = std::dynamic_pointer_cast<FloatLiteral>(literal)) {
        // 创建常量操作
        auto constantOp = std::make_shared<ir::ConstantOp>();
        
        // 创建结果值
        std::shared_ptr<ir::Type> type = std::make_shared<ir::ScalarType>(ir::DataType::FLOAT32);
        std::shared_ptr<ir::Value> value = createValue(generateUniqueName("float_literal"), type);
        constantOp->addResult(value);
        
        // 设置常量值
        constantOp->setAttribute("value", std::to_string(floatLiteral->value));
        
        // 将操作添加到当前图
        if (currentGraph) {
            currentGraph->addOperation(constantOp);
        }
        
        return value;
    } else if (auto boolLiteral = std::dynamic_pointer_cast<BoolLiteral>(literal)) {
        // 创建常量操作
        auto constantOp = std::make_shared<ir::ConstantOp>();
        
        // 创建结果值
        std::shared_ptr<ir::Type> type = std::make_shared<ir::ScalarType>(ir::DataType::BOOL);
        std::shared_ptr<ir::Value> value = createValue(generateUniqueName("bool_literal"), type);
        constantOp->addResult(value);
        
        // 设置常量值
        constantOp->setAttribute("value", boolLiteral->value ? "true" : "false");
        
        // 将操作添加到当前图
        if (currentGraph) {
            currentGraph->addOperation(constantOp);
        }
        
        return value;
    } else if (auto tensorLiteral = std::dynamic_pointer_cast<TensorLiteral>(literal)) {
        // 创建常量操作
        auto constantOp = std::make_shared<ir::ConstantOp>();
        
        // 创建结果值
        std::shared_ptr<ir::Type> type = convertType(tensorLiteral->getType());
        std::shared_ptr<ir::Value> value = createValue(generateUniqueName("tensor_literal"), type);
        constantOp->addResult(value);
        
        // 设置常量值（简化：不处理实际的张量值）
        constantOp->setAttribute("value", "tensor");
        
        // 将操作添加到当前图
        if (currentGraph) {
            currentGraph->addOperation(constantOp);
        }
        
        return value;
    }
    
    // 默认返回空
    return nullptr;
}

std::shared_ptr<ir::Value> ASTToIRConverter::createValue(const std::string& name, std::shared_ptr<ir::Type> type) {
    if (auto tensorType = std::dynamic_pointer_cast<ir::TensorType>(type)) {
        return std::make_shared<ir::Tensor>(name, tensorType);
    } else {
        return std::make_shared<ir::Value>(name, type);
    }
}

std::shared_ptr<ir::Operation> ASTToIRConverter::createOperation(const std::string& opType) {
    return std::make_shared<ir::Operation>(opType);
}

std::string ASTToIRConverter::generateUniqueName(const std::string& prefix) {
    return prefix + "_" + std::to_string(nameCounter++);
}

} // namespace ai_compiler
