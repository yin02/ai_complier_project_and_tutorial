#ifndef IR_H
#define IR_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <set>

namespace ai_compiler {
namespace ir {

// 前向声明
class Type;
class Value;
class Operation;
class Tensor;
class Graph;
class Module;

// 数据类型枚举
enum class DataType {
    FLOAT32,
    INT32,
    BOOL,
    UNKNOWN
};

// IR 类型系统
class Type {
public:
    virtual ~Type() = default;
    virtual std::string toString() const = 0;
    virtual bool equals(const Type& other) const = 0;
};

// 标量类型
class ScalarType : public Type {
public:
    DataType dataType;
    
    ScalarType(DataType dataType) : dataType(dataType) {}
    
    std::string toString() const override {
        switch (dataType) {
            case DataType::FLOAT32: return "float32";
            case DataType::INT32: return "int32";
            case DataType::BOOL: return "bool";
            default: return "unknown";
        }
    }
    
    bool equals(const Type& other) const override {
        if (auto otherScalar = dynamic_cast<const ScalarType*>(&other)) {
            return dataType == otherScalar->dataType;
        }
        return false;
    }
};

// 张量类型
class TensorType : public Type {
public:
    DataType elementType;
    std::vector<int64_t> shape;
    
    TensorType(DataType elementType, const std::vector<int64_t>& shape)
        : elementType(elementType), shape(shape) {}
    
    std::string toString() const override {
        std::string result = "tensor<";
        switch (elementType) {
            case DataType::FLOAT32: result += "float32"; break;
            case DataType::INT32: result += "int32"; break;
            case DataType::BOOL: result += "bool"; break;
            default: result += "unknown"; break;
        }
        
        result += ", [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) result += ", ";
            if (shape[i] < 0) {
                result += "?";
            } else {
                result += std::to_string(shape[i]);
            }
        }
        result += "]>";
        return result;
    }
    
    bool equals(const Type& other) const override {
        if (auto otherTensor = dynamic_cast<const TensorType*>(&other)) {
            if (elementType != otherTensor->elementType) return false;
            if (shape.size() != otherTensor->shape.size()) return false;
            
            for (size_t i = 0; i < shape.size(); ++i) {
                // -1 表示动态维度，可以匹配任何大小
                if (shape[i] >= 0 && otherTensor->shape[i] >= 0 && shape[i] != otherTensor->shape[i]) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }
};

// IR 值基类
class Value {
public:
    std::string name;
    std::shared_ptr<Type> type;
    std::weak_ptr<Operation> definingOp; // 定义此值的操作
    std::vector<std::weak_ptr<Operation>> users; // 使用此值的操作
    
    Value(const std::string& name, std::shared_ptr<Type> type)
        : name(name), type(type) {}
    
    virtual ~Value() = default;
    virtual std::string toString() const {
        return name + ": " + type->toString();
    }
    
    void addUser(std::shared_ptr<Operation> op) {
        users.push_back(op);
    }
    
    void setDefiningOp(std::shared_ptr<Operation> op) {
        definingOp = op;
    }
};

// 张量值
class Tensor : public Value {
public:
    Tensor(const std::string& name, std::shared_ptr<TensorType> type)
        : Value(name, type) {}
    
    std::shared_ptr<TensorType> getTensorType() const {
        return std::static_pointer_cast<TensorType>(type);
    }
};

// 操作基类
class Operation : public std::enable_shared_from_this<Operation> {
public:
    std::string opType;
    std::vector<std::shared_ptr<Value>> operands;
    std::vector<std::shared_ptr<Value>> results;
    std::weak_ptr<Graph> parentGraph;
    std::unordered_map<std::string, std::string> attributes;
    
    Operation(const std::string& opType) : opType(opType) {}
    virtual ~Operation() = default;
    
    virtual std::string toString() const {
        std::string result = opType + "(";
        for (size_t i = 0; i < operands.size(); ++i) {
            if (i > 0) result += ", ";
            result += operands[i]->name;
        }
        result += ") -> (";
        for (size_t i = 0; i < results.size(); ++i) {
            if (i > 0) result += ", ";
            result += results[i]->name;
        }
        result += ")";
        return result;
    }
    
    void addOperand(std::shared_ptr<Value> value) {
        operands.push_back(value);
        value->addUser(shared_from_this());
    }
    
    void addResult(std::shared_ptr<Value> value) {
        results.push_back(value);
        value->setDefiningOp(shared_from_this());
    }
    
    void setAttribute(const std::string& key, const std::string& value) {
        attributes[key] = value;
    }
    
    std::string getAttribute(const std::string& key) const {
        auto it = attributes.find(key);
        if (it != attributes.end()) {
            return it->second;
        }
        return "";
    }
};

// 计算图
class Graph : public std::enable_shared_from_this<Graph> {
public:
    std::string name;
    std::vector<std::shared_ptr<Value>> inputs;
    std::vector<std::shared_ptr<Value>> outputs;
    std::vector<std::shared_ptr<Operation>> operations;
    
    Graph(const std::string& name) : name(name) {}
    
    void addOperation(std::shared_ptr<Operation> op) {
        operations.push_back(op);
        op->parentGraph = shared_from_this();
    }
    
    void addInput(std::shared_ptr<Value> value) {
        inputs.push_back(value);
    }
    
    void addOutput(std::shared_ptr<Value> value) {
        outputs.push_back(value);
    }
    
    std::string toString() const {
        std::string result = "graph " + name + "(";
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i > 0) result += ", ";
            result += inputs[i]->toString();
        }
        result += ") -> (";
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (i > 0) result += ", ";
            result += outputs[i]->toString();
        }
        result += ") {\n";
        
        for (const auto& op : operations) {
            result += "  " + op->toString() + "\n";
        }
        
        result += "}";
        return result;
    }
};

// 模块（包含多个图）
class Module {
public:
    std::string name;
    std::vector<std::shared_ptr<Graph>> graphs;
    
    Module(const std::string& name) : name(name) {}
    
    void addGraph(std::shared_ptr<Graph> graph) {
        graphs.push_back(graph);
    }
    
    std::string toString() const {
        std::string result = "module " + name + " {\n";
        for (const auto& graph : graphs) {
            result += graph->toString() + "\n";
        }
        result += "}";
        return result;
    }
};

// 常见操作类型

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
    
    BinaryOp(OpKind kind) : Operation(getOpName(kind)), kind(kind) {}
    
    static std::string getOpName(OpKind kind) {
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
    
    UnaryOp(OpKind kind) : Operation(getOpName(kind)), kind(kind) {}
    
    static std::string getOpName(OpKind kind) {
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

// 变量操作
class VariableOp : public Operation {
public:
    VariableOp() : Operation("variable") {}
};

// 返回操作
class ReturnOp : public Operation {
public:
    ReturnOp() : Operation("return") {}
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
    
    PoolOp(PoolKind kind) : Operation(getOpName(kind)), kind(kind) {}
    
    static std::string getOpName(PoolKind kind) {
        switch (kind) {
            case PoolKind::MAX: return "max_pool";
            case PoolKind::AVG: return "avg_pool";
            default: return "unknown";
        }
    }
};

// 形状操作
class ReshapeOp : public Operation {
public:
    ReshapeOp() : Operation("reshape") {}
};

// 转置操作
class TransposeOp : public Operation {
public:
    TransposeOp() : Operation("transpose") {}
};

// 连接操作
class ConcatOp : public Operation {
public:
    ConcatOp() : Operation("concat") {}
};

} // namespace ir
} // namespace ai_compiler

#endif // IR_H
