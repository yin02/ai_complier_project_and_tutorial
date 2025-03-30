# 图优化教程：Pass框架与优化算法

本教程将介绍AI编译器的图优化Pass框架和常见的优化算法，包括常量折叠、死代码消除、布局转换和内核融合等。

## 1. 图优化概述

图优化是AI编译器中至关重要的一环，它可以显著提高生成代码的性能。优化的目标包括：

- 减少计算量
- 提高内存访问效率
- 利用硬件特性
- 减少内存占用
- 降低延迟

### 1.1 优化的分类

我们的优化Pass可以分为以下几类：

1. **通用优化**：适用于大多数程序的优化，如常量折叠、死代码消除等
2. **AI特定优化**：针对AI模型的特定优化，如布局转换、内核融合等
3. **硬件特定优化**：针对特定硬件平台的优化，如利用SIMD指令、GPU加速等

### 1.2 优化的层次

优化可以在不同层次上进行：

1. **模块级优化**：作用于整个模块，如全局变量优化
2. **图级优化**：作用于单个计算图，如图重写、图分区等
3. **操作级优化**：作用于单个操作，如操作融合、操作替换等

## 2. Pass框架设计

Pass框架是优化的基础设施，它提供了注册、调度和执行优化Pass的机制。

### 2.1 Pass基类

```cpp
class Pass {
public:
    enum class PassType {
        MODULE_PASS,    // 作用于整个模块
        GRAPH_PASS,     // 作用于单个计算图
        OPERATION_PASS  // 作用于单个操作
    };
    
    Pass(const std::string& name, PassType type) 
        : name(name), type(type) {}
    
    virtual ~Pass() = default;
    
    // 获取Pass名称
    std::string getName() const { return name; }
    
    // 获取Pass类型
    PassType getType() const { return type; }
    
    // 运行Pass（由子类实现）
    virtual bool run(std::shared_ptr<ir::Module> module) = 0;
    
protected:
    std::string name;
    PassType type;
};
```

### 2.2 特定类型的Pass

```cpp
// 模块级Pass
class ModulePass : public Pass {
public:
    ModulePass(const std::string& name) : Pass(name, PassType::MODULE_PASS) {}
    
    // 运行模块级Pass
    virtual bool run(std::shared_ptr<ir::Module> module) override = 0;
};

// 图级Pass
class GraphPass : public Pass {
public:
    GraphPass(const std::string& name) : Pass(name, PassType::GRAPH_PASS) {}
    
    // 运行图级Pass
    virtual bool run(std::shared_ptr<ir::Module> module) override {
        bool changed = false;
        
        // 对模块中的每个图运行Pass
        for (auto& graph : module->graphs) {
            changed |= runOnGraph(graph);
        }
        
        return changed;
    }
    
    // 在单个图上运行Pass
    virtual bool runOnGraph(std::shared_ptr<ir::Graph> graph) = 0;
};

// 操作级Pass
class OperationPass : public Pass {
public:
    OperationPass(const std::string& name) : Pass(name, PassType::OPERATION_PASS) {}
    
    // 运行操作级Pass
    virtual bool run(std::shared_ptr<ir::Module> module) override {
        bool changed = false;
        
        // 对模块中的每个图运行Pass
        for (auto& graph : module->graphs) {
            changed |= runOnGraph(graph);
        }
        
        return changed;
    }
    
    // 在单个图上运行Pass
    virtual bool runOnGraph(std::shared_ptr<ir::Graph> graph) {
        bool changed = false;
        
        // 对图中的每个操作运行Pass
        for (auto& op : graph->operations) {
            changed |= runOnOperation(op);
        }
        
        return changed;
    }
    
    // 在单个操作上运行Pass
    virtual bool runOnOperation(std::shared_ptr<ir::Operation> operation) = 0;
};
```

### 2.3 Pass管理器

```cpp
class PassManager {
public:
    PassManager() {}
    
    // 添加Pass
    void addPass(std::shared_ptr<Pass> pass) {
        passes.push_back(pass);
    }
    
    // 运行所有Pass
    bool run(std::shared_ptr<ir::Module> module) {
        bool changed = false;
        
        // 按顺序运行所有Pass
        for (auto& pass : passes) {
            changed |= pass->run(module);
        }
        
        return changed;
    }
    
    // 清空所有Pass
    void clear() {
        passes.clear();
    }
    
private:
    std::vector<std::shared_ptr<Pass>> passes;
};
```

### 2.4 Pass注册表

```cpp
class PassRegistry {
public:
    using PassCreator = std::function<std::shared_ptr<Pass>()>;
    
    // 获取单例实例
    static PassRegistry& getInstance() {
        static PassRegistry instance;
        return instance;
    }
    
    // 注册Pass
    void registerPass(const std::string& name, PassCreator creator) {
        registry[name] = creator;
    }
    
    // 创建Pass
    std::shared_ptr<Pass> createPass(const std::string& name) {
        auto it = registry.find(name);
        if (it != registry.end()) {
            return it->second();
        }
        return nullptr;
    }
    
    // 获取所有注册的Pass名称
    std::vector<std::string> getRegisteredPassNames() const {
        std::vector<std::string> names;
        for (const auto& entry : registry) {
            names.push_back(entry.first);
        }
        return names;
    }
    
private:
    PassRegistry() = default;
    ~PassRegistry() = default;
    
    // 禁止拷贝和赋值
    PassRegistry(const PassRegistry&) = delete;
    PassRegistry& operator=(const PassRegistry&) = delete;
    
    std::unordered_map<std::string, PassCreator> registry;
};

// Pass注册辅助宏
#define REGISTER_PASS(PassClass) \
    static bool PassClass##_registered = []() { \
        PassRegistry::getInstance().registerPass( \
            #PassClass, []() { return std::make_shared<PassClass>(); }); \
        return true; \
    }();
```

## 3. 常量折叠优化

常量折叠是一种基本的优化技术，它在编译时计算常量表达式，减少运行时计算。

### 3.1 常量折叠Pass实现

```cpp
class ConstantFoldingPass : public OperationPass {
public:
    ConstantFoldingPass() : OperationPass("ConstantFoldingPass") {}
    
    bool runOnOperation(std::shared_ptr<ir::Operation> operation) override {
        // 如果操作不能被折叠，直接返回
        if (!canFold(operation)) {
            return false;
        }
        
        // 计算常量操作的结果
        std::shared_ptr<ir::Value> result = evaluateConstantOp(operation);
        if (!result) {
            return false;
        }
        
        // 创建新的常量操作
        std::shared_ptr<ir::Operation> constOp = createConstantOp(
            operation->getAttribute("value"), result->type);
        
        // 将原操作的结果替换为新常量操作的结果
        for (auto& user : operation->results[0]->users) {
            if (auto userOp = user.lock()) {
                // 替换操作数
                for (size_t i = 0; i < userOp->operands.size(); ++i) {
                    if (userOp->operands[i] == operation->results[0]) {
                        userOp->operands[i] = constOp->results[0];
                    }
                }
            }
        }
        
        // 将新的常量操作添加到图中
        if (auto graph = operation->parentGraph.lock()) {
            graph->addOperation(constOp);
            
            // 从图中移除原操作
            auto& ops = graph->operations;
            ops.erase(std::remove(ops.begin(), ops.end(), operation), ops.end());
        }
        
        return true;
    }
    
private:
    // 检查操作是否可以被折叠
    bool canFold(std::shared_ptr<ir::Operation> operation) {
        // 检查操作是否是二元操作
        if (auto binaryOp = std::dynamic_pointer_cast<ir::BinaryOp>(operation)) {
            // 检查操作数是否都是常量
            for (auto& operand : operation->operands) {
                // 查找定义此操作数的操作
                if (auto defOp = operand->definingOp.lock()) {
                    if (defOp->opType != "constant") {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            return true;
        }
        
        // 检查操作是否是一元操作
        if (auto unaryOp = std::dynamic_pointer_cast<ir::UnaryOp>(operation)) {
            // 检查操作数是否是常量
            if (operation->operands.empty()) {
                return false;
            }
            
            auto& operand = operation->operands[0];
            if (auto defOp = operand->definingOp.lock()) {
                return defOp->opType == "constant";
            }
        }
        
        return false;
    }
    
    // 计算常量操作的结果
    std::shared_ptr<ir::Value> evaluateConstantOp(std::shared_ptr<ir::Operation> operation) {
        // 简化实现：只处理简单的二元操作
        if (auto binaryOp = std::dynamic_pointer_cast<ir::BinaryOp>(operation)) {
            if (operation->operands.size() != 2 || operation->results.size() != 1) {
                return nullptr;
            }
            
            // 获取操作数的值
            auto op1DefOp = operation->operands[0]->definingOp.lock();
            auto op2DefOp = operation->operands[1]->definingOp.lock();
            
            if (!op1DefOp || !op2DefOp || 
                op1DefOp->opType != "constant" || op2DefOp->opType != "constant") {
                return nullptr;
            }
            
            std::string val1 = op1DefOp->getAttribute("value");
            std::string val2 = op2DefOp->getAttribute("value");
            
            // 尝试将值转换为数字
            try {
                float num1 = std::stof(val1);
                float num2 = std::stof(val2);
                float result = 0.0f;
                
                // 根据操作类型计算结果
                switch (binaryOp->kind) {
                    case ir::BinaryOp::OpKind::ADD:
                        result = num1 + num2;
                        break;
                    case ir::BinaryOp::OpKind::SUB:
                        result = num1 - num2;
                        break;
                    case ir::BinaryOp::OpKind::MUL:
                        result = num1 * num2;
                        break;
                    case ir::BinaryOp::OpKind::DIV:
                        if (num2 == 0.0f) {
                            return nullptr; // 除以零错误
                        }
                        result = num1 / num2;
                        break;
                    default:
                        return nullptr; // 不支持的操作类型
                }
                
                // 创建结果值
                std::shared_ptr<ir::Value> resultValue = std::make_shared<ir::Value>(
                    "const_" + std::to_string(result), operation->results[0]->type);
                
                return resultValue;
            } catch (const std::exception& e) {
                return nullptr; // 转换失败
            }
        }
        
        return nullptr;
    }
    
    // 创建常量操作
    std::shared_ptr<ir::Operation> createConstantOp(const std::string& value, std::shared_ptr<ir::Type> type) {
        auto constOp = std::make_shared<ir::ConstantOp>();
        constOp->setAttribute("value", value);
        
        auto resultValue = std::make_shared<ir::Value>("const_" + value, type);
        constOp->addResult(resultValue);
        
        return constOp;
    }
};

// 注册Pass
REGISTER_PASS(ConstantFoldingPass);
```

## 4. 死代码消除优化

死代码消除是一种优化技术，它移除程序中不会影响结果的代码。

### 4.1 死代码消除Pass实现

```cpp
class DeadCodeEliminationPass : public GraphPass {
public:
    DeadCodeEliminationPass() : GraphPass("DeadCodeEliminationPass") {}
    
    bool runOnGraph(std::shared_ptr<ir::Graph> graph) override {
        // 标记活跃值
        std::unordered_set<std::shared_ptr<ir::Value>> liveValues;
        markLiveValues(graph, liveValues);
        
        // 移除死代码
        return removeDeadOperations(graph, liveValues);
    }
    
private:
    // 标记活跃值
    void markLiveValues(std::shared_ptr<ir::Graph> graph, 
                        std::unordered_set<std::shared_ptr<ir::Value>>& liveValues) {
        // 从图的输出开始标记
        std::queue<std::shared_ptr<ir::Value>> workList;
        
        // 将图的输出添加到工作列表
        for (auto& output : graph->outputs) {
            workList.push(output);
            liveValues.insert(output);
        }
        
        // 广度优先遍历，标记所有活跃值
        while (!workList.empty()) {
            auto value = workList.front();
            workList.pop();
            
            // 获取定义此值的操作
            if (auto defOp = value->definingOp.lock()) {
                // 将操作的所有操作数添加到工作列表
                for (auto& operand : defOp->operands) {
                    if (liveValues.find(operand) == liveValues.end()) {
                        liveValues.insert(operand);
                        workList.push(operand);
                    }
                }
            }
        }
    }
    
    // 移除死操作
    bool removeDeadOperations(std::shared_ptr<ir::Graph> graph, 
                             const std::unordered_set<std::shared_ptr<ir::Value>>& liveValues) {
        bool changed = false;
        
        // 收集死操作
        std::vector<std::shared_ptr<ir::Operation>> deadOps;
        
        for (auto& op : graph->operations) {
            bool isLive = false;
            
            // 检查操作的结果是否有活跃的
            for (auto& result : op->results) {
                if (liveValues.find(result) != liveValues.end()) {
                    isLive = true;
                    break;
                }
            }
            
            if (!isLive) {
                deadOps.push_back(op);
            }
        }
        
        // 从图中移除死操作
        if (!deadOps.empty()) {
            changed = true;
            
            for (auto& deadOp : deadOps) {
                auto& ops = graph->operations;
                ops.erase(std::remove(ops.begin(), ops.end(), deadOp), ops.end());
            }
        }
        
        return changed;
    }
};

// 注册Pass
REGISTER_PASS(DeadCodeEliminationPass);
```

## 5. 布局转换优化

布局转换是一种AI特定的优化，它改变张量的内存布局，以提高内存访问效率和计算性能。

### 5.1 布局转换Pass实现

```cpp
class LayoutTransformationPass : public GraphPass {
public:
    LayoutTransformationPass() : GraphPass("LayoutTransformationPass") {}
    
    bool runOnGraph(std::shared_ptr<ir::Graph> graph) override {
        bool changed = false;
        
        // 遍历图中的所有操作
        for (auto& op : graph->operations) {
            // 检查操作是否支持布局转换
            if (supportsLayoutTransformation(op)) {
                // 应用布局转换
                if (applyLayoutTransformation(op)) {
                    changed = true;
                }
            }
        }
        
        return changed;
    }
    
private:
    // 检查操作是否支持布局转换
    bool supportsLayoutTransformation(std::shared_ptr<ir::Operation> operation) {
        // 简化实现：只考虑特定类型的操作
        return operation->opType == "conv2d" || 
               operation->opType == "max_pool" || 
               operation->opType == "avg_pool";
    }
    
    // 应用布局转换
    bool applyLayoutTransformation(std::shared_ptr<ir::Operation> operation) {
        // 简化实现：假设我们要将NCHW布局转换为NHWC布局
        
        // 检查当前布局
        std::string currentLayout = operation->getAttribute("data_format");
        if (currentLayout.empty()) {
            // 默认为NCHW
            currentLayout = "NCHW";
        }
        
        // 如果已经是目标布局，则不需要转换
        if (currentLayout == "NHWC") {
            return false;
        }
        
        // 为每个操作数添加布局转换
        for (size_t i = 0; i < operation->operands.size(); ++i) {
            auto& operand = operation->operands[i];
            
            // 创建布局转换操作
            auto transformOp = createLayoutTransformOp(operand, "NHWC");
            
            // 替换操作数
            operation->operands[i] = transformOp->results[0];
            
            // 将布局转换操作添加到图中
            if (auto graph = operation->parentGraph.lock()) {
                graph->addOperation(transformOp);
            }
        }
        
        // 更新操作的布局属性
        operation->setAttribute("data_format", "NHWC");
        
        return true;
    }
    
    // 创建布局转换操作
    std::shared_ptr<ir::Operation> createLayoutTransformOp(
        std::shared_ptr<ir::Value> input, const std::string& targetLayout) {
        // 创建布局转换操作
        auto transformOp = std::make_shared<ir::Operation>("layout_transform");
        
        // 设置操作数
        transformOp->addOperand(input);
        
        // 设置结果
        auto result = std::make_shared<ir::Value>(
            "transformed_" + input->name, input->type);
        transformOp->addResult(result);
        
        // 设置属性
        transformOp->setAttribute("target_layout", targetLayout);
        
        return transformOp;
    }
};

// 注册Pass
REGISTER_PASS(LayoutTransformationPass);
```

## 6. 内核融合优化

内核融合是一种AI特定的优化，它将多个操作融合为一个操作，减少内存访问和内核启动开销。

### 6.1 内核融合Pass实现

```cpp
class KernelFusionPass : public GraphPass {
public:
    KernelFusionPass() : GraphPass("KernelFusionPass") {}
    
    bool runOnGraph(std::shared_ptr<ir::Graph> graph) override {
        bool changed = false;
        
        // 查找可融合的内核组
        auto fusibleGroups = findFusibleKernelGroups(graph);
        
        // 融合内核组
        for (const auto& group : fusibleGroups) {
            if (group.size() <= 1) {
                continue;
            }
            
            // 融合内核组
            auto fusedKernel = fuseKernelGroup(group);
            if (fusedKernel) {
                // 将融合后的内核添加到图中
                graph->addOperation(fusedKernel);
                
                // 从图中移除原内核
                auto& ops = graph->operations;
                for (auto& op : group) {
                    ops.erase(std::remove(ops.begin(), ops.end(), op), ops.end());
                }
                
                changed = true;
            }
        }
        
        return changed;
    }
    
private:
    // 检查操作是否可以作为内核融合的候选
    bool isKernelFusionCandidate(std::shared_ptr<ir::Operation> operation) {
        // 简化实现：只考虑特定类型的操作
        return operation->opType == "conv2d" || 
               operation->opType == "matmul" || 
               operation->opType == "relu" || 
               operation->opType == "add";
    }
    
    // 查找可融合的内核组
    std::vector<std::vector<std::shared_ptr<ir::Operation>>> 
    findFusibleKernelGroups(std::shared_ptr<ir::Graph> graph) {
        std::vector<std::vector<std::shared_ptr<ir::Operation>>> groups;
        
        // 简化实现：使用贪心算法查找可融合的内核组
        std::unordered_set<std::shared_ptr<ir::Operation>> visited;
        
        for (auto& op : graph->operations) {
            if (visited.find(op) != visited.end() || !isKernelFusionCandidate(op)) {
                continue;
            }
            
            // 开始一个新的组
            std::vector<std::shared_ptr<ir::Operation>> group;
            group.push_back(op);
            visited.insert(op);
            
            // 查找可以添加到此组的操作
            bool expanded;
            do {
                expanded = false;
                
                for (auto& candidate : graph->operations) {
                    if (visited.find(candidate) != visited.end() || !isKernelFusionCandidate(candidate)) {
                        continue;
                    }
                    
                    // 检查是否可以添加到组中
                    bool canAdd = false;
                    
                    // 检查是否有数据依赖
                    for (auto& groupOp : group) {
                        for (auto& result : groupOp->results) {
                            for (auto& operand : candidate->operands) {
                                if (result == operand) {
                                    canAdd = true;
                                    break;
                                }
                            }
                            if (canAdd) break;
                        }
                        if (canAdd) break;
                    }
                    
                    if (canAdd) {
                        group.push_back(candidate);
                        visited.insert(candidate);
                        expanded = true;
                    }
                }
            } while (expanded);
            
            // 添加组
            if (group.size() > 1) {
                groups.push_back(group);
            }
        }
        
        return groups;
    }
    
    // 融合内核组
    std::shared_ptr<ir::Operation> fuseKernelGroup(
        const std::vector<std::shared_ptr<ir::Operation>>& kernelGroup) {
        if (kernelGroup.empty()) {
            return nullptr;
        }
        
        // 创建融合内核
        std::string fusedOpType = "fused_kernel";
        auto fusedOp = std::make_shared<ir::Operation>(fusedOpType);
        
        // 收集所有输入和输出
        std::unordered_set<std::shared_ptr<ir::Value>> inputs;
        std::unordered_set<std::shared_ptr<ir::Value>> intermediates;
        std::unordered_set<std::shared_ptr<ir::Value>> outputs;
        
        // 收集中间结果
        for (auto& op : kernelGroup) {
            for (auto& result : op->results) {
                intermediates.insert(result);
            }
        }
        
        // 收集输入和输出
        for (auto& op : kernelGroup) {
            for (auto& operand : op->operands) {
                // 如果操作数不是组内任何操作的结果，则它是输入
                if (intermediates.find(operand) == intermediates.end()) {
                    inputs.insert(operand);
                }
            }
            
            for (auto& result : op->results) {
                // 检查结果是否被组外的操作使用
                bool usedOutside = false;
                for (auto& userWeak : result->users) {
                    if (auto user = userWeak.lock()) {
                        // 检查用户是否在组外
                        bool inGroup = false;
                        for (auto& groupOp : kernelGroup) {
                            if (user == groupOp) {
                                inGroup = true;
                                break;
                            }
                        }
                        
                        if (!inGroup) {
                            usedOutside = true;
                            break;
                        }
                    }
                }
                
                if (usedOutside) {
                    outputs.insert(result);
                }
            }
        }
        
        // 设置融合操作的操作数和结果
        for (auto& input : inputs) {
            fusedOp->addOperand(input);
        }
        
        for (auto& output : outputs) {
            fusedOp->addResult(output);
        }
        
        // 设置融合操作的属性
        fusedOp->setAttribute("fused_ops", std::to_string(kernelGroup.size()));
        
        // 设置子操作的类型
        std::string subOps;
        for (size_t i = 0; i < kernelGroup.size(); ++i) {
            if (i > 0) subOps += ",";
            subOps += kernelGroup[i]->opType;
        }
        fusedOp->setAttribute("sub_ops", subOps);
        
        return fusedOp;
    }
};

// 注册Pass
REGISTER_PASS(KernelFusionPass);
```

## 7. 优化管道

优化管道是一系列按特定顺序执行的优化Pass，用于提高程序性能。

### 7.1 优化管道实现

```cpp
class OptimizationPipeline {
public:
    OptimizationPipeline() {
        initializeDefaultPasses();
    }
    
    // 运行优化管道
    bool run(std::shared_ptr<ir::Module> module) {
        return passManager.run(module);
    }
    
    // 添加自定义Pass
    void addPass(std::shared_ptr<Pass> pass) {
        passManager.addPass(pass);
    }
    
private:
    PassManager passManager;
    
    // 初始化默认Pass
    void initializeDefaultPasses() {
        // 添加默认的优化Pass
        passManager.addPass(std::make_shared<ConstantFoldingPass>());
        passManager.addPass(std::make_shared<DeadCodeEliminationPass>());
        passManager.addPass(std::make_shared<CommonSubexpressionEliminationPass>());
        passManager.addPass(std::make_shared<OperationFusionPass>());
        passManager.addPass(std::make_shared<LayoutTransformationPass>());
        passManager.addPass(std::make_shared<KernelFusionPass>());
        passManager.addPass(std::make_shared<MemoryOptimizationPass>());
    }
};
```

## 8. 实践：优化一个简单的计算图

让我们看一个完整的例子，优化以下计算图：

```
// 创建一个简单的计算图
auto graph = std::make_shared<ir::Graph>("example");

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

// 创建计算
auto mul1 = std::make_shared<ir::BinaryOp>(ir::BinaryOp::OpKind::MUL);
mul1->addOperand(const1Value);
mul1->addOperand(const2Value);
auto mul1Result = std::make_shared<ir::Value>("mul1_result", TypeUtils::createFloat32Type());
mul1->addResult(mul1Result);
graph->addOperation(mul1);

auto conv = std::make_shared<ir::ConvOp>();
conv->addOperand(input);
conv->addOperand(mul1Result);
conv->setAttribute("stride", "1,1");
conv->setAttribute("padding", "same");
auto convResult = std::make_shared<ir::Value>("conv_result", TypeUtils::createTensorType(
    TypeUtils::createFloat32Type(), {1, 64, 224, 224}));
conv->addResult(convResult);
graph->addOperation(conv);

auto relu = std::make_shared<ir::UnaryOp>(ir::UnaryOp::OpKind::RELU);
relu->addOperand(convResult);
auto reluResult = std::make_shared<ir::Value>("relu_result", TypeUtils::createTensorType(
    TypeUtils::createFloat32Type(), {1, 64, 224, 224}));
relu->addResult(reluResult);
graph->addOperation(relu);

// 设置输出
graph->addOutput(reluResult);

// 创建模块
auto module = std::make_shared<ir::Module>("example_module");
module->addGraph(graph);

// 创建优化管道
OptimizationPipeline pipeline;

// 运行优化
bool changed = pipeline.run(module);

// 打印优化后的IR
IRPrinter printer;
printer.print(module);
```

优化后的计算图将包含以下变化：

1. 常量折叠：`const1 * const2` 将被折叠为常量 `6.0`
2. 内核融合：`conv + relu` 将被融合为一个操作 `fused_kernel`

## 9. 练习

1. 实现一个新的优化Pass，如循环展开或向量化
2. 扩展内核融合Pass，支持更多的融合模式
3. 实现一个自定义的优化管道，针对特定的模型或硬件平台

## 10. 总结

图优化是AI编译器的核心组件，它可以显著提高生成代码的性能。通过设计良好的Pass框架和实现各种优化算法，我们可以自动化地优化AI模型，提高其执行效率。

在下一个教程中，我们将介绍如何将优化后的IR转换为LLVM IR，为代码生成做准备。
