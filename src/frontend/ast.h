#ifndef AST_H
#define AST_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace ai_compiler {

// 前向声明
class Type;
class Expression;
class Statement;

// AST 节点基类
class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual std::string toString() const = 0;
};

// 类型系统
class Type : public ASTNode {
public:
    enum class TypeKind {
        INT,
        FLOAT,
        BOOL,
        TENSOR,
        FUNCTION
    };

    TypeKind kind;

    Type(TypeKind kind) : kind(kind) {}
    virtual ~Type() = default;
};

// 基本类型
class BasicType : public Type {
public:
    BasicType(TypeKind kind) : Type(kind) {}
    
    std::string toString() const override {
        switch (kind) {
            case TypeKind::INT: return "int";
            case TypeKind::FLOAT: return "float";
            case TypeKind::BOOL: return "bool";
            default: return "unknown";
        }
    }
};

// 张量类型
class TensorType : public Type {
public:
    std::shared_ptr<Type> elementType;
    std::vector<int> shape;
    
    TensorType(std::shared_ptr<Type> elementType, const std::vector<int>& shape)
        : Type(TypeKind::TENSOR), elementType(elementType), shape(shape) {}
    
    std::string toString() const override {
        std::string result = "tensor<" + elementType->toString() + ", [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) result += ", ";
            result += std::to_string(shape[i]);
        }
        result += "]>";
        return result;
    }
};

// 函数类型
class FunctionType : public Type {
public:
    std::vector<std::shared_ptr<Type>> paramTypes;
    std::shared_ptr<Type> returnType;
    
    FunctionType(const std::vector<std::shared_ptr<Type>>& paramTypes, 
                 std::shared_ptr<Type> returnType)
        : Type(TypeKind::FUNCTION), paramTypes(paramTypes), returnType(returnType) {}
    
    std::string toString() const override {
        std::string result = "(";
        for (size_t i = 0; i < paramTypes.size(); ++i) {
            if (i > 0) result += ", ";
            result += paramTypes[i]->toString();
        }
        result += ") -> " + returnType->toString();
        return result;
    }
};

// 表达式基类
class Expression : public ASTNode {
public:
    virtual ~Expression() = default;
    virtual std::shared_ptr<Type> getType() const = 0;
};

// 字面量表达式基类
class Literal : public Expression {
public:
    virtual ~Literal() = default;
};

// 整数字面量
class IntLiteral : public Literal {
public:
    int value;
    
    IntLiteral(int value) : value(value) {}
    
    std::string toString() const override {
        return std::to_string(value);
    }
    
    std::shared_ptr<Type> getType() const override {
        return std::make_shared<BasicType>(Type::TypeKind::INT);
    }
};

// 浮点数字面量
class FloatLiteral : public Literal {
public:
    float value;
    
    FloatLiteral(float value) : value(value) {}
    
    std::string toString() const override {
        return std::to_string(value);
    }
    
    std::shared_ptr<Type> getType() const override {
        return std::make_shared<BasicType>(Type::TypeKind::FLOAT);
    }
};

// 布尔字面量
class BoolLiteral : public Literal {
public:
    bool value;
    
    BoolLiteral(bool value) : value(value) {}
    
    std::string toString() const override {
        return value ? "true" : "false";
    }
    
    std::shared_ptr<Type> getType() const override {
        return std::make_shared<BasicType>(Type::TypeKind::BOOL);
    }
};

// 张量字面量
class TensorLiteral : public Literal {
public:
    std::vector<std::shared_ptr<Expression>> elements;
    std::shared_ptr<TensorType> type;
    
    TensorLiteral(const std::vector<std::shared_ptr<Expression>>& elements, 
                  std::shared_ptr<TensorType> type)
        : elements(elements), type(type) {}
    
    std::string toString() const override {
        std::string result = "[";
        for (size_t i = 0; i < elements.size(); ++i) {
            if (i > 0) result += ", ";
            result += elements[i]->toString();
        }
        result += "]";
        return result;
    }
    
    std::shared_ptr<Type> getType() const override {
        return type;
    }
};

// 变量引用
class Variable : public Expression {
public:
    std::string name;
    std::shared_ptr<Type> type;
    
    Variable(const std::string& name, std::shared_ptr<Type> type)
        : name(name), type(type) {}
    
    std::string toString() const override {
        return name;
    }
    
    std::shared_ptr<Type> getType() const override {
        return type;
    }
};

// 二元表达式
class BinaryExpr : public Expression {
public:
    enum class Op {
        ADD,
        SUB,
        MUL,
        DIV,
        EQ,
        NEQ,
        LT,
        GT,
        LE,
        GE,
        AND,
        OR
    };
    
    Op op;
    std::shared_ptr<Expression> left;
    std::shared_ptr<Expression> right;
    
    BinaryExpr(Op op, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        : op(op), left(left), right(right) {}
    
    std::string toString() const override {
        std::string opStr;
        switch (op) {
            case Op::ADD: opStr = "+"; break;
            case Op::SUB: opStr = "-"; break;
            case Op::MUL: opStr = "*"; break;
            case Op::DIV: opStr = "/"; break;
            case Op::EQ: opStr = "=="; break;
            case Op::NEQ: opStr = "!="; break;
            case Op::LT: opStr = "<"; break;
            case Op::GT: opStr = ">"; break;
            case Op::LE: opStr = "<="; break;
            case Op::GE: opStr = ">="; break;
            case Op::AND: opStr = "&&"; break;
            case Op::OR: opStr = "||"; break;
        }
        return "(" + left->toString() + " " + opStr + " " + right->toString() + ")";
    }
    
    std::shared_ptr<Type> getType() const override {
        // 简化版类型推断
        if (op == Op::EQ || op == Op::NEQ || op == Op::LT || op == Op::GT || 
            op == Op::LE || op == Op::GE || op == Op::AND || op == Op::OR) {
            return std::make_shared<BasicType>(Type::TypeKind::BOOL);
        }
        return left->getType(); // 假设左右操作数类型相同
    }
};

// 一元表达式
class UnaryExpr : public Expression {
public:
    enum class Op {
        NEG,
        NOT
    };
    
    Op op;
    std::shared_ptr<Expression> expr;
    
    UnaryExpr(Op op, std::shared_ptr<Expression> expr)
        : op(op), expr(expr) {}
    
    std::string toString() const override {
        std::string opStr = (op == Op::NEG) ? "-" : "!";
        return opStr + expr->toString();
    }
    
    std::shared_ptr<Type> getType() const override {
        if (op == Op::NOT) {
            return std::make_shared<BasicType>(Type::TypeKind::BOOL);
        }
        return expr->getType();
    }
};

// 函数调用表达式
class CallExpr : public Expression {
public:
    std::string callee;
    std::vector<std::shared_ptr<Expression>> arguments;
    std::shared_ptr<Type> returnType;
    
    CallExpr(const std::string& callee, 
             const std::vector<std::shared_ptr<Expression>>& arguments,
             std::shared_ptr<Type> returnType)
        : callee(callee), arguments(arguments), returnType(returnType) {}
    
    std::string toString() const override {
        std::string result = callee + "(";
        for (size_t i = 0; i < arguments.size(); ++i) {
            if (i > 0) result += ", ";
            result += arguments[i]->toString();
        }
        result += ")";
        return result;
    }
    
    std::shared_ptr<Type> getType() const override {
        return returnType;
    }
};

// 语句基类
class Statement : public ASTNode {
public:
    virtual ~Statement() = default;
};

// 表达式语句
class ExpressionStmt : public Statement {
public:
    std::shared_ptr<Expression> expression;
    
    ExpressionStmt(std::shared_ptr<Expression> expression)
        : expression(expression) {}
    
    std::string toString() const override {
        return expression->toString() + ";";
    }
};

// 变量声明
class VarDeclaration : public Statement {
public:
    std::string name;
    std::shared_ptr<Type> type;
    std::shared_ptr<Expression> initializer;
    
    VarDeclaration(const std::string& name, std::shared_ptr<Type> type,
                   std::shared_ptr<Expression> initializer)
        : name(name), type(type), initializer(initializer) {}
    
    std::string toString() const override {
        std::string result = "var " + name + ": " + type->toString();
        if (initializer) {
            result += " = " + initializer->toString();
        }
        return result + ";";
    }
};

// 块语句
class Block : public Statement {
public:
    std::vector<std::shared_ptr<Statement>> statements;
    
    Block(const std::vector<std::shared_ptr<Statement>>& statements)
        : statements(statements) {}
    
    std::string toString() const override {
        std::string result = "{\n";
        for (const auto& stmt : statements) {
            result += "  " + stmt->toString() + "\n";
        }
        result += "}";
        return result;
    }
};

// If 语句
class IfStatement : public Statement {
public:
    std::shared_ptr<Expression> condition;
    std::shared_ptr<Statement> thenBranch;
    std::shared_ptr<Statement> elseBranch;
    
    IfStatement(std::shared_ptr<Expression> condition,
                std::shared_ptr<Statement> thenBranch,
                std::shared_ptr<Statement> elseBranch)
        : condition(condition), thenBranch(thenBranch), elseBranch(elseBranch) {}
    
    std::string toString() const override {
        std::string result = "if (" + condition->toString() + ") " + thenBranch->toString();
        if (elseBranch) {
            result += " else " + elseBranch->toString();
        }
        return result;
    }
};

// For 循环语句
class ForStatement : public Statement {
public:
    std::shared_ptr<Statement> initializer;
    std::shared_ptr<Expression> condition;
    std::shared_ptr<Expression> increment;
    std::shared_ptr<Statement> body;
    
    ForStatement(std::shared_ptr<Statement> initializer,
                 std::shared_ptr<Expression> condition,
                 std::shared_ptr<Expression> increment,
                 std::shared_ptr<Statement> body)
        : initializer(initializer), condition(condition), increment(increment), body(body) {}
    
    std::string toString() const override {
        std::string result = "for (";
        if (initializer) {
            result += initializer->toString();
        } else {
            result += ";";
        }
        
        if (condition) {
            result += " " + condition->toString();
        }
        result += ";";
        
        if (increment) {
            result += " " + increment->toString();
        }
        
        result += ") " + body->toString();
        return result;
    }
};

// 返回语句
class ReturnStatement : public Statement {
public:
    std::shared_ptr<Expression> value;
    
    ReturnStatement(std::shared_ptr<Expression> value)
        : value(value) {}
    
    std::string toString() const override {
        if (value) {
            return "return " + value->toString() + ";";
        }
        return "return;";
    }
};

// 函数参数
class Parameter {
public:
    std::string name;
    std::shared_ptr<Type> type;
    
    Parameter(const std::string& name, std::shared_ptr<Type> type)
        : name(name), type(type) {}
    
    std::string toString() const {
        return name + ": " + type->toString();
    }
};

// 函数声明
class FunctionDeclaration : public Statement {
public:
    std::string name;
    std::vector<Parameter> parameters;
    std::shared_ptr<Type> returnType;
    std::shared_ptr<Block> body;
    
    FunctionDeclaration(const std::string& name,
                        const std::vector<Parameter>& parameters,
                        std::shared_ptr<Type> returnType,
                        std::shared_ptr<Block> body)
        : name(name), parameters(parameters), returnType(returnType), body(body) {}
    
    std::string toString() const override {
        std::string result = "func " + name + "(";
        for (size_t i = 0; i < parameters.size(); ++i) {
            if (i > 0) result += ", ";
            result += parameters[i].toString();
        }
        result += "): " + returnType->toString() + " " + body->toString();
        return result;
    }
};

// 操作声明
class OperationDeclaration : public Statement {
public:
    std::string name;
    std::vector<Parameter> parameters;
    std::shared_ptr<Type> returnType;
    std::shared_ptr<Block> body;
    
    OperationDeclaration(const std::string& name,
                         const std::vector<Parameter>& parameters,
                         std::shared_ptr<Type> returnType,
                         std::shared_ptr<Block> body)
        : name(name), parameters(parameters), returnType(returnType), body(body) {}
    
    std::string toString() const override {
        std::string result = "op " + name + "(";
        for (size_t i = 0; i < parameters.size(); ++i) {
            if (i > 0) result += ", ";
            result += parameters[i].toString();
        }
        result += "): " + returnType->toString() + " " + body->toString();
        return result;
    }
};

// 计算图声明
class GraphDeclaration : public Statement {
public:
    std::string name;
    std::vector<Parameter> parameters;
    std::shared_ptr<Type> returnType;
    std::shared_ptr<Block> body;
    
    GraphDeclaration(const std::string& name,
                     const std::vector<Parameter>& parameters,
                     std::shared_ptr<Type> returnType,
                     std::shared_ptr<Block> body)
        : name(name), parameters(parameters), returnType(returnType), body(body) {}
    
    std::string toString() const override {
        std::string result = "graph " + name + "(";
        for (size_t i = 0; i < parameters.size(); ++i) {
            if (i > 0) result += ", ";
            result += parameters[i].toString();
        }
        result += "): " + returnType->toString() + " " + body->toString();
        return result;
    }
};

// 导入语句
class ImportStatement : public Statement {
public:
    std::string module;
    
    ImportStatement(const std::string& module)
        : module(module) {}
    
    std::string toString() const override {
        return "import \"" + module + "\";";
    }
};

// 程序
class Program : public ASTNode {
public:
    std::vector<std::shared_ptr<Statement>> statements;
    
    Program(const std::vector<std::shared_ptr<Statement>>& statements)
        : statements(statements) {}
    
    std::string toString() const override {
        std::string result;
        for (const auto& stmt : statements) {
            result += stmt->toString() + "\n";
        }
        return result;
    }
};

} // namespace ai_compiler

#endif // AST_H
