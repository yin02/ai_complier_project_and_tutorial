#ifndef AST_TO_IR_H
#define AST_TO_IR_H

#include "../frontend/ast.h"
#include "include/ir.h"
#include <memory>
#include <unordered_map>
#include <string>

namespace ai_compiler {

// AST到IR的转换器类
class ASTToIRConverter {
public:
    ASTToIRConverter();
    
    // 将AST程序转换为IR模块
    std::shared_ptr<ir::Module> convert(const std::shared_ptr<Program>& program);
    
private:
    // 当前正在构建的模块
    std::shared_ptr<ir::Module> currentModule;
    
    // 当前正在构建的图
    std::shared_ptr<ir::Graph> currentGraph;
    
    // 符号表：变量名到IR值的映射
    std::unordered_map<std::string, std::shared_ptr<ir::Value>> symbolTable;
    
    // 类型转换：AST类型到IR类型
    std::shared_ptr<ir::Type> convertType(const std::shared_ptr<Type>& astType);
    
    // 转换各种AST节点到IR
    void convertStatement(const std::shared_ptr<Statement>& stmt);
    void convertVarDeclaration(const std::shared_ptr<VarDeclaration>& varDecl);
    void convertFunctionDeclaration(const std::shared_ptr<FunctionDeclaration>& funcDecl);
    void convertOperationDeclaration(const std::shared_ptr<OperationDeclaration>& opDecl);
    void convertGraphDeclaration(const std::shared_ptr<GraphDeclaration>& graphDecl);
    void convertBlock(const std::shared_ptr<Block>& block);
    void convertIfStatement(const std::shared_ptr<IfStatement>& ifStmt);
    void convertForStatement(const std::shared_ptr<ForStatement>& forStmt);
    void convertReturnStatement(const std::shared_ptr<ReturnStatement>& returnStmt);
    void convertExpressionStatement(const std::shared_ptr<ExpressionStmt>& exprStmt);
    
    std::shared_ptr<ir::Value> convertExpression(const std::shared_ptr<Expression>& expr);
    std::shared_ptr<ir::Value> convertBinaryExpr(const std::shared_ptr<BinaryExpr>& binExpr);
    std::shared_ptr<ir::Value> convertUnaryExpr(const std::shared_ptr<UnaryExpr>& unaryExpr);
    std::shared_ptr<ir::Value> convertCallExpr(const std::shared_ptr<CallExpr>& callExpr);
    std::shared_ptr<ir::Value> convertVariable(const std::shared_ptr<Variable>& var);
    std::shared_ptr<ir::Value> convertLiteral(const std::shared_ptr<Literal>& literal);
    
    // 创建新的IR值
    std::shared_ptr<ir::Value> createValue(const std::string& name, std::shared_ptr<ir::Type> type);
    
    // 创建新的IR操作
    std::shared_ptr<ir::Operation> createOperation(const std::string& opType);
    
    // 生成唯一的名称
    std::string generateUniqueName(const std::string& prefix);
    
    // 计数器，用于生成唯一名称
    int nameCounter = 0;
};

} // namespace ai_compiler

#endif // AST_TO_IR_H
