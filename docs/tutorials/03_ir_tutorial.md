# 中间表示(IR)教程：设计与AST转换

本教程将介绍AI编译器的中间表示(IR)设计以及如何将抽象语法树(AST)转换为IR。

## 1. 中间表示(IR)概述

中间表示(IR)是编译器内部使用的程序表示形式，它介于源代码和目标代码之间。IR的设计对编译器的性能和功能至关重要，因为它影响：

- 编译器能够执行的优化类型
- 代码生成的效率和质量
- 编译器的可扩展性和可维护性

### 1.1 IR的设计目标

我们的AI编译器IR设计有以下目标：

1. **表达能力**：能够表示所有DSL语言特性，包括张量操作、控制流和计算图
2. **优化友好**：便于实现各种优化，如常量折叠、死代码消除、布局转换和内核融合
3. **可分析性**：便于静态分析，如数据流分析和依赖分析
4. **可扩展性**：易于添加新的操作和优化
5. **与LLVM兼容**：便于转换为LLVM IR

### 1.2 IR的主要组件

我们的IR包含以下主要组件：

- **类型系统**：表示数据类型，如标量类型和张量类型
- **值**：表示程序中的数据，如变量、常量和临时值
- **操作**：表示计算，如加法、乘法、卷积等
- **基本块**：包含线性执行的操作序列
- **计算图**：表示数据流和控制流
- **模块**：包含多个计算图和全局数据

## 2. IR类型系统

类型系统是IR的基础，它定义了程序中可以操作的数据类型。

### 2.1 类型层次结构

```cpp
class Type {
public:
    enum class TypeKind {
        SCALAR,
        TENSOR,
        TUPLE
    };
    
    TypeKind kind;
    
    Type(TypeKind kind) : kind(kind) {}
    virtual ~Type() = default;
    
    virtual std::string toString() const = 0;
    virtual bool equals(const Type& other) const = 0;
};

class ScalarType : public Type {
public:
    enum class DataType {
        FLOAT32,
        INT32,
        BOOL
    };
    
    DataType dataType;
    
    ScalarType(DataType dataType) : Type(TypeKind::SCALAR), dataType(dataType) {}
    
    std::string toString() const override {
        switch (dataType) {
            case DataType::FLOAT32: return "float";
            case DataType::INT32: return "int";
            case DataType::BOOL: return "bool";
            default: return "unknown";
        }
    }
    
    bool equals(const Type& other) const override {
        if (other.kind != TypeKind::SCALAR) return false;
        const ScalarType& otherScalar = static_cast<const ScalarType&>(other);
        return dataType == otherScalar.dataType;
    }
};

class TensorType : public Type {
public:
    std::shared_ptr<ScalarType> elementType;
    std::vector<int> shape;
    
    TensorType(std::shared_ptr<ScalarType> elementType, const std::vector<int>& shape)
        : Type(TypeKind::TENSOR), elementType(elementType), shape(shape) {}
    
    std::string toString() const override {
        std::stringstream ss;
        ss << "tensor<" << elementType->toString() << ", [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << shape[i];
        }
        ss << "]>";
        return ss.str();
    }
    
    bool equals(const Type& other) const override {
        if (other.kind != TypeKind::TENSOR) return false;
        const TensorType& otherTensor = static_cast<const TensorType&>(other);
        if (!elementType->equals(*otherTensor.elementType)) return false;
        if (shape.size() != otherTensor.shape.size()) return false;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != otherTensor.shape[i]) return false;
        }
        return true;
    }
};
```

### 2.2 类型实用函数

```cpp
namespace TypeUtils {
    // 创建标量类型
    std::shared_ptr<ScalarType> createFloat32Type() {
        return std::make_shared<ScalarType>(ScalarType::DataType::FLOAT32);
    }
    
    std::shared_ptr<ScalarType> createInt32Type() {
        return std::make_shared<ScalarType>(ScalarType::DataType::INT32);
    }
    
    std::shared_ptr<ScalarType> createBoolType() {
        return std::make_shared<ScalarType>(ScalarType::DataType::BOOL);
    }
    
    // 创建张量类型
    std::shared_ptr<TensorType> createTensorType(std::shared_ptr<ScalarType> elementType, 
                                                const std::vector<int>& shape) {
        return std::make_shared<TensorType>(elementType, shape);
    }
}
```

## 3. IR值和操作

值和操作是IR的核心组件，它们表示程序中的数据和计算。

### 3.1 值

```cpp
class Value {
public:
    std::string name;
    std::shared_ptr<Type> type;
    std::weak_ptr<Operation> definingOp; // 定义此值的操作
    std::vector<std::weak_ptr<Operation>> users; // 使用此值的操作
    
    Value(const std::string& name, std::shared_ptr<Type> type)
        : name(name), type(type) {}
    
    void addUser(std::shared_ptr<Operation> user) {
        users.push_back(user);
    }
    
    void replaceAllUsesWith(std::shared_ptr<Value> newValue) {
        for (auto& userWeak : users) {
            if (auto user = userWeak.lock()) {
                user->replaceOperand(shared_from_this(), newValue);
            }
        }
        users.clear();
    }
};
```

### 3.2 操作

```cpp
class Operation : public std::enable_shared_from_this<Operation> {
public:
    std::string opType;
    std::vector<std::shared_ptr<Value>> operands;
    std::vector<std::shared_ptr<Value>> results;
    std::unordered_map<std::string, std::string> attributes;
    std::weak_ptr<Graph> parentGraph;
    
    Operation(const std::string& opType) : opType(opType) {}
    
    void addOperand(std::shared_ptr<Value> operand) {
        operands.push_back(operand);
        operand->addUser(shared_from_this());
    }
    
    void addResult(std::shared_ptr<Value> result) {
        results.push_back(result);
        result->definingOp = shared_from_this();
    }
    
    void setAttribute(const std::string& name, const std::string& value) {
        attributes[name] = value;
    }
    
    std::string getAttribute(const std::string& name) const {
        auto it = attributes.find(name);
        if (it != attributes.end()) {
            return it->second;
        }
        return "";
    }
    
    void replaceOperand(std::shared_ptr<Value> oldOperand, std::shared_ptr<Value> newOperand) {
        for (size_t i = 0; i < operands.size(); ++i) {
            if (operands[i] == oldOperand) {
                operands[i] = newOperand;
                newOperand->addUser(shared_from_this());
            }
        }
    }
};
```

### 3.3 特定操作类型

```cpp
// 二元操作
class BinaryOp : public Operation {
public:
    enum class OpKind {
        ADD,
        SUB,
        MUL,
        DIV,
        MATMUL
    };
    
    OpKind kind;
    
    BinaryOp(OpKind kind) : Operation(getOpTypeName(kind)), kind(kind) {}
    
    static std::string getOpTypeName(OpKind kind) {
        switch (kind) {
            case OpKind::ADD: return "add";
            case OpKind::SUB: return "sub";
            case OpKind::MUL: return "mul";
            case OpKind::DIV: return "div";
            case OpKind::MATMUL: return "matmul";
            default: return "unknown";
        }
    }
};

// 一元操作
class UnaryOp : public Operation {
public:
    enum class OpKind {
        NEG,
        RELU,
        SIGMOID,
        TANH,
        SOFTMAX
    };
    
    OpKind kind;
    
    UnaryOp(OpKind kind) : Operation(getOpTypeName(kind)), kind(kind) {}
    
    static std::string getOpTypeName(OpKind kind) {
        switch (kind) {
            case OpKind::NEG: return "neg";
            case OpKind::RELU: return "relu";
            case OpKind::SIGMOID: return "sigmoid";
            case OpKind::TANH: return "tanh";
            case OpKind::SOFTMAX: return "softmax";
            default: return "unknown";
        }
    }
};

// 常量操作
class ConstantOp : public Operation {
public:
    ConstantOp() : Operation("constant") {}
};

// 卷积操作
class ConvOp : public Operation {
public:
    ConvOp() : Operation("conv2d") {}
};

// 池化操作
class PoolOp : public Operation {
public:
    enum class PoolKind {
        MAX,
        AVG
    };
    
    PoolKind kind;
    
    PoolOp(PoolKind kind) : Operation(getOpTypeName(kind)), kind(kind) {}
    
    static std::string getOpTypeName(PoolKind kind) {
        switch (kind) {
            case PoolKind::MAX: return "max_pool";
            case PoolKind::AVG: return "avg_pool";
            default: return "unknown";
        }
    }
};

// 重塑操作
class ReshapeOp : public Operation {
public:
    ReshapeOp() : Operation("reshape") {}
};
```

## 4. 计算图和模块

计算图和模块是IR的高级组件，它们组织操作和值。

### 4.1 计算图

```cpp
class Graph : public std::enable_shared_from_this<Graph> {
public:
    std::string name;
    std::vector<std::shared_ptr<Value>> inputs;
    std::vector<std::shared_ptr<Value>> outputs;
    std::vector<std::shared_ptr<Operation>> operations;
    std::weak_ptr<Module> parentModule;
    
    Graph(const std::string& name) : name(name) {}
    
    void addInput(std::shared_ptr<Value> input) {
        inputs.push_back(input);
    }
    
    void addOutput(std::shared_ptr<Value> output) {
        outputs.push_back(output);
    }
    
    void addOperation(std::shared_ptr<Operation> op) {
        operations.push_back(op);
        op->parentGraph = shared_from_this();
    }
};
```

### 4.2 模块

```cpp
class Module {
public:
    std::string name;
    std::vector<std::shared_ptr<Graph>> graphs;
    
    Module(const std::string& name) : name(name) {}
    
    void addGraph(std::shared_ptr<Graph> graph) {
        graphs.push_back(graph);
        graph->parentModule = shared_from_this();
    }
};
```

## 5. AST到IR的转换

AST到IR的转换是编译器前端到中端的桥梁，它将AST转换为IR，为后续的优化和代码生成做准备。

### 5.1 转换器类

```cpp
class ASTToIRConverter {
public:
    ASTToIRConverter() {}
    
    std::shared_ptr<Module> convert(const std::shared_ptr<Program>& program) {
        module = std::make_shared<Module>("main");
        
        // 处理导入语句
        for (const auto& stmt : program->statements) {
            if (auto importStmt = std::dynamic_pointer_cast<ImportStatement>(stmt)) {
                processImport(importStmt);
            }
        }
        
        // 处理函数和图声明
        for (const auto& stmt : program->statements) {
            if (auto funcDecl = std::dynamic_pointer_cast<FunctionDeclaration>(stmt)) {
                processFunction(funcDecl);
            } else if (auto graphDecl = std::dynamic_pointer_cast<GraphDeclaration>(stmt)) {
                processGraph(graphDecl);
            }
        }
        
        return module;
    }
    
private:
    std::shared_ptr<Module> module;
    std::unordered_map<std::string, std::shared_ptr<Value>> valueMap;
    
    void processImport(const std::shared_ptr<ImportStatement>& importStmt) {
        // 处理导入语句
    }
    
    void processFunction(const std::shared_ptr<FunctionDeclaration>& funcDecl) {
        // 创建函数图
        auto graph = std::make_shared<Graph>(funcDecl->name);
        
        // 处理函数参数
        for (const auto& param : funcDecl->params) {
            auto paramType = convertType(param->type);
            auto paramValue = std::make_shared<Value>(param->name, paramType);
            graph->addInput(paramValue);
            valueMap[param->name] = paramValue;
        }
        
        // 处理函数体
        processBlock(funcDecl->body, graph);
        
        // 添加图到模块
        module->addGraph(graph);
    }
    
    void processGraph(const std::shared_ptr<GraphDeclaration>& graphDecl) {
        // 创建计算图
        auto graph = std::make_shared<Graph>(graphDecl->name);
        
        // 处理图参数
        for (const auto& param : graphDecl->params) {
            auto paramType = convertType(param->type);
            auto paramValue = std::make_shared<Value>(param->name, paramType);
            graph->addInput(paramValue);
            valueMap[param->name] = paramValue;
        }
        
        // 处理图体
        processBlock(graphDecl->body, graph);
        
        // 处理返回值
        if (graphDecl->returnType) {
            auto returnType = convertType(graphDecl->returnType);
            auto returnValue = processExpression(graphDecl->returnExpr, graph);
            graph->addOutput(returnValue);
        }
        
        // 添加图到模块
        module->addGraph(graph);
    }
    
    void processBlock(const std::shared_ptr<BlockStatement>& block, std::shared_ptr<Graph> graph) {
        for (const auto& stmt : block->statements) {
            processStatement(stmt, graph);
        }
    }
    
    void processStatement(const std::shared_ptr<Statement>& stmt, std::shared_ptr<Graph> graph) {
        if (auto exprStmt = std::dynamic_pointer_cast<ExpressionStatement>(stmt)) {
            processExpression(exprStmt->expr, graph);
        } else if (auto varDecl = std::dynamic_pointer_cast<VarDeclaration>(stmt)) {
            processVarDeclaration(varDecl, graph);
        } else if (auto returnStmt = std::dynamic_pointer_cast<ReturnStatement>(stmt)) {
            auto returnValue = processExpression(returnStmt->expr, graph);
            graph->addOutput(returnValue);
        } else if (auto ifStmt = std::dynamic_pointer_cast<IfStatement>(stmt)) {
            processIfStatement(ifStmt, graph);
        } else if (auto forStmt = std::dynamic_pointer_cast<ForStatement>(stmt)) {
            processForStatement(forStmt, graph);
        }
    }
    
    std::shared_ptr<Value> processExpression(const std::shared_ptr<Expression>& expr, std::shared_ptr<Graph> graph) {
        if (auto binaryExpr = std::dynamic_pointer_cast<BinaryExpr>(expr)) {
            return processBinaryExpr(binaryExpr, graph);
        } else if (auto unaryExpr = std::dynamic_pointer_cast<UnaryExpr>(expr)) {
            return processUnaryExpr(unaryExpr, graph);
        } else if (auto callExpr = std::dynamic_pointer_cast<CallExpr>(expr)) {
            return processCallExpr(callExpr, graph);
        } else if (auto varExpr = std::dynamic_pointer_cast<VarExpr>(expr)) {
            return processVarExpr(varExpr);
        } else if (auto literalExpr = std::dynamic_pointer_cast<LiteralExpr>(expr)) {
            return processLiteralExpr(literalExpr, graph);
        }
        
        return nullptr;
    }
    
    std::shared_ptr<Value> processBinaryExpr(const std::shared_ptr<BinaryExpr>& binaryExpr, std::shared_ptr<Graph> graph) {
        auto left = processExpression(binaryExpr->left, graph);
        auto right = processExpression(binaryExpr->right, graph);
        
        BinaryOp::OpKind opKind;
        switch (binaryExpr->op) {
            case BinaryExpr::Op::ADD: opKind = BinaryOp::OpKind::ADD; break;
            case BinaryExpr::Op::SUB: opKind = BinaryOp::OpKind::SUB; break;
            case BinaryExpr::Op::MUL: opKind = BinaryOp::OpKind::MUL; break;
            case BinaryExpr::Op::DIV: opKind = BinaryOp::OpKind::DIV; break;
            default: throw std::runtime_error("Unsupported binary operation");
        }
        
        auto binaryOp = std::make_shared<BinaryOp>(opKind);
        binaryOp->addOperand(left);
        binaryOp->addOperand(right);
        
        std::shared_ptr<Type> resultType;
        if (left->type->kind == Type::TypeKind::TENSOR || right->type->kind == Type::TypeKind::TENSOR) {
            // 张量操作
            if (left->type->kind == Type::TypeKind::TENSOR && right->type->kind == Type::TypeKind::TENSOR) {
                auto leftTensor = std::static_pointer_cast<TensorType>(left->type);
                auto rightTensor = std::static_pointer_cast<TensorType>(right->type);
                
                // 简化：使用左操作数的形状
                resultType = std::make_shared<TensorType>(
                    std::static_pointer_cast<ScalarType>(leftTensor->elementType),
                    leftTensor->shape);
            } else if (left->type->kind == Type::TypeKind::TENSOR) {
                resultType = left->type;
            } else {
                resultType = right->type;
            }
        } else {
            // 标量操作
            resultType = left->type;
        }
        
        auto result = std::make_shared<Value>("binary_result", resultType);
        binaryOp->addResult(result);
        
        graph->addOperation(binaryOp);
        
        return result;
    }
    
    // ... 其他处理方法
    
    std::shared_ptr<Type> convertType(const std::shared_ptr<ast::Type>& astType) {
        if (auto scalarType = std::dynamic_pointer_cast<ast::ScalarType>(astType)) {
            switch (scalarType->kind) {
                case ast::ScalarType::Kind::INT:
                    return TypeUtils::createInt32Type();
                case ast::ScalarType::Kind::FLOAT:
                    return TypeUtils::createFloat32Type();
                case ast::ScalarType::Kind::BOOL:
                    return TypeUtils::createBoolType();
                default:
                    throw std::runtime_error("Unsupported scalar type");
            }
        } else if (auto tensorType = std::dynamic_pointer_cast<ast::TensorType>(astType)) {
            auto elementType = std::static_pointer_cast<ScalarType>(
                convertType(tensorType->elementType));
            return TypeUtils::createTensorType(elementType, tensorType->shape);
        }
        
        throw std::runtime_error("Unsupported type");
    }
};
```

### 5.2 使用转换器

```cpp
// 解析源代码
Parser parser(sourceCode);
std::shared_ptr<Program> program = parser.parse();

// 转换为IR
ASTToIRConverter converter;
std::shared_ptr<Module> module = converter.convert(program);

// 打印IR
IRPrinter printer;
printer.print(module);
```

## 6. IR验证器

IR验证器用于检查IR的一致性和正确性，确保IR满足编译器的假设。

```cpp
class IRVerifier {
public:
    bool verify(const std::shared_ptr<Module>& module) {
        bool valid = true;
        
        // 验证模块
        for (const auto& graph : module->graphs) {
            valid &= verifyGraph(graph);
        }
        
        return valid;
    }
    
private:
    bool verifyGraph(const std::shared_ptr<Graph>& graph) {
        bool valid = true;
        
        // 验证图名称
        if (graph->name.empty()) {
            std::cerr << "Graph has empty name" << std::endl;
            valid = false;
        }
        
        // 验证操作
        for (const auto& op : graph->operations) {
            valid &= verifyOperation(op);
        }
        
        // 验证输入和输出
        for (const auto& input : graph->inputs) {
            if (input->definingOp.lock()) {
                std::cerr << "Graph input has defining operation" << std::endl;
                valid = false;
            }
        }
        
        for (const auto& output : graph->outputs) {
            if (!output->definingOp.lock() && std::find(graph->inputs.begin(), graph->inputs.end(), output) == graph->inputs.end()) {
                std::cerr << "Graph output has no defining operation and is not an input" << std::endl;
                valid = false;
            }
        }
        
        return valid;
    }
    
    bool verifyOperation(const std::shared_ptr<Operation>& op) {
        bool valid = true;
        
        // 验证操作类型
        if (op->opType.empty()) {
            std::cerr << "Operation has empty type" << std::endl;
            valid = false;
        }
        
        // 验证操作数和结果
        for (const auto& operand : op->operands) {
            if (!operand) {
                std::cerr << "Operation has null operand" << std::endl;
                valid = false;
            }
        }
        
        for (const auto& result : op->results) {
            if (!result) {
                std::cerr << "Operation has null result" << std::endl;
                valid = false;
            } else if (result->definingOp.lock() != op) {
                std::cerr << "Result's defining operation does not match" << std::endl;
                valid = false;
            }
        }
        
        return valid;
    }
};
```

## 7. 实践：转换一个简单的DSL程序

让我们看一个完整的例子，将以下DSL程序转换为IR：

```
func add(a: float, b: float): float {
    return a + b;
}

func main() {
    var x: float = 5.0;
    var y: float = 10.0;
    var z: float = add(x, y);
    print(z);
}
```

1. 解析源代码，生成AST
2. 使用ASTToIRConverter将AST转换为IR
3. 使用IRVerifier验证IR的正确性
4. 打印IR，查看转换结果

## 8. 练习

1. 扩展IR，支持更多的操作类型
2. 实现一个简单的IR优化，如常量折叠
3. 修改ASTToIRConverter，支持更复杂的控制流

## 9. 总结

中间表示(IR)是编译器的核心组件，它连接前端和后端，为优化提供基础。通过设计良好的IR和实现AST到IR的转换，我们可以将DSL程序转换为更适合优化和代码生成的形式。

在下一个教程中，我们将介绍如何实现图优化Pass，提高生成代码的性能。
