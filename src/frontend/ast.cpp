#include "ast.h"

namespace ai_compiler {

// AST 类的实现文件
// 大部分实现已经在头文件中完成，这里主要提供一些辅助函数和工厂方法

// 创建基本类型的工厂方法
std::shared_ptr<BasicType> createIntType() {
    return std::make_shared<BasicType>(Type::TypeKind::INT);
}

std::shared_ptr<BasicType> createFloatType() {
    return std::make_shared<BasicType>(Type::TypeKind::FLOAT);
}

std::shared_ptr<BasicType> createBoolType() {
    return std::make_shared<BasicType>(Type::TypeKind::BOOL);
}

// 创建张量类型的工厂方法
std::shared_ptr<TensorType> createTensorType(std::shared_ptr<Type> elementType, 
                                            const std::vector<int>& shape) {
    return std::make_shared<TensorType>(elementType, shape);
}

// 创建函数类型的工厂方法
std::shared_ptr<FunctionType> createFunctionType(
    const std::vector<std::shared_ptr<Type>>& paramTypes,
    std::shared_ptr<Type> returnType) {
    return std::make_shared<FunctionType>(paramTypes, returnType);
}

// 创建字面量的工厂方法
std::shared_ptr<IntLiteral> createIntLiteral(int value) {
    return std::make_shared<IntLiteral>(value);
}

std::shared_ptr<FloatLiteral> createFloatLiteral(float value) {
    return std::make_shared<FloatLiteral>(value);
}

std::shared_ptr<BoolLiteral> createBoolLiteral(bool value) {
    return std::make_shared<BoolLiteral>(value);
}

std::shared_ptr<TensorLiteral> createTensorLiteral(
    const std::vector<std::shared_ptr<Expression>>& elements,
    std::shared_ptr<TensorType> type) {
    return std::make_shared<TensorLiteral>(elements, type);
}

// 创建变量引用的工厂方法
std::shared_ptr<Variable> createVariable(const std::string& name, std::shared_ptr<Type> type) {
    return std::make_shared<Variable>(name, type);
}

// 创建二元表达式的工厂方法
std::shared_ptr<BinaryExpr> createBinaryExpr(
    BinaryExpr::Op op,
    std::shared_ptr<Expression> left,
    std::shared_ptr<Expression> right) {
    return std::make_shared<BinaryExpr>(op, left, right);
}

// 创建一元表达式的工厂方法
std::shared_ptr<UnaryExpr> createUnaryExpr(
    UnaryExpr::Op op,
    std::shared_ptr<Expression> expr) {
    return std::make_shared<UnaryExpr>(op, expr);
}

// 创建函数调用表达式的工厂方法
std::shared_ptr<CallExpr> createCallExpr(
    const std::string& callee,
    const std::vector<std::shared_ptr<Expression>>& arguments,
    std::shared_ptr<Type> returnType) {
    return std::make_shared<CallExpr>(callee, arguments, returnType);
}

// 创建表达式语句的工厂方法
std::shared_ptr<ExpressionStmt> createExpressionStmt(std::shared_ptr<Expression> expression) {
    return std::make_shared<ExpressionStmt>(expression);
}

// 创建变量声明的工厂方法
std::shared_ptr<VarDeclaration> createVarDeclaration(
    const std::string& name,
    std::shared_ptr<Type> type,
    std::shared_ptr<Expression> initializer) {
    return std::make_shared<VarDeclaration>(name, type, initializer);
}

// 创建块语句的工厂方法
std::shared_ptr<Block> createBlock(const std::vector<std::shared_ptr<Statement>>& statements) {
    return std::make_shared<Block>(statements);
}

// 创建 If 语句的工厂方法
std::shared_ptr<IfStatement> createIfStatement(
    std::shared_ptr<Expression> condition,
    std::shared_ptr<Statement> thenBranch,
    std::shared_ptr<Statement> elseBranch) {
    return std::make_shared<IfStatement>(condition, thenBranch, elseBranch);
}

// 创建 For 循环语句的工厂方法
std::shared_ptr<ForStatement> createForStatement(
    std::shared_ptr<Statement> initializer,
    std::shared_ptr<Expression> condition,
    std::shared_ptr<Expression> increment,
    std::shared_ptr<Statement> body) {
    return std::make_shared<ForStatement>(initializer, condition, increment, body);
}

// 创建返回语句的工厂方法
std::shared_ptr<ReturnStatement> createReturnStatement(std::shared_ptr<Expression> value) {
    return std::make_shared<ReturnStatement>(value);
}

// 创建函数声明的工厂方法
std::shared_ptr<FunctionDeclaration> createFunctionDeclaration(
    const std::string& name,
    const std::vector<Parameter>& parameters,
    std::shared_ptr<Type> returnType,
    std::shared_ptr<Block> body) {
    return std::make_shared<FunctionDeclaration>(name, parameters, returnType, body);
}

// 创建操作声明的工厂方法
std::shared_ptr<OperationDeclaration> createOperationDeclaration(
    const std::string& name,
    const std::vector<Parameter>& parameters,
    std::shared_ptr<Type> returnType,
    std::shared_ptr<Block> body) {
    return std::make_shared<OperationDeclaration>(name, parameters, returnType, body);
}

// 创建计算图声明的工厂方法
std::shared_ptr<GraphDeclaration> createGraphDeclaration(
    const std::string& name,
    const std::vector<Parameter>& parameters,
    std::shared_ptr<Type> returnType,
    std::shared_ptr<Block> body) {
    return std::make_shared<GraphDeclaration>(name, parameters, returnType, body);
}

// 创建导入语句的工厂方法
std::shared_ptr<ImportStatement> createImportStatement(const std::string& module) {
    return std::make_shared<ImportStatement>(module);
}

// 创建程序的工厂方法
std::shared_ptr<Program> createProgram(const std::vector<std::shared_ptr<Statement>>& statements) {
    return std::make_shared<Program>(statements);
}

} // namespace ai_compiler
