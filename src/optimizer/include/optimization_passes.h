#ifndef OPTIMIZATION_PASSES_H
#define OPTIMIZATION_PASSES_H

#include "pass.h"
#include "../../ir/include/ir.h"
#include <unordered_set>
#include <unordered_map>
#include <queue>

namespace ai_compiler {
namespace optimizer {

// 常量折叠Pass
class ConstantFoldingPass : public OperationPass {
public:
    ConstantFoldingPass() : OperationPass("ConstantFoldingPass") {}
    
    bool runOnOperation(std::shared_ptr<ir::Operation> operation) override;
    
private:
    // 计算常量操作的结果
    std::shared_ptr<ir::Value> evaluateConstantOp(std::shared_ptr<ir::Operation> operation);
    
    // 检查操作是否可以被折叠
    bool canFold(std::shared_ptr<ir::Operation> operation);
    
    // 创建新的常量操作
    std::shared_ptr<ir::Operation> createConstantOp(const std::string& value, std::shared_ptr<ir::Type> type);
};

// 死代码消除Pass
class DeadCodeEliminationPass : public GraphPass {
public:
    DeadCodeEliminationPass() : GraphPass("DeadCodeEliminationPass") {}
    
    bool runOnGraph(std::shared_ptr<ir::Graph> graph) override;
    
private:
    // 标记活跃值
    void markLiveValues(std::shared_ptr<ir::Graph> graph, std::unordered_set<std::shared_ptr<ir::Value>>& liveValues);
    
    // 移除死代码
    bool removeDeadOperations(std::shared_ptr<ir::Graph> graph, const std::unordered_set<std::shared_ptr<ir::Value>>& liveValues);
};

// 公共子表达式消除Pass
class CommonSubexpressionEliminationPass : public GraphPass {
public:
    CommonSubexpressionEliminationPass() : GraphPass("CommonSubexpressionEliminationPass") {}
    
    bool runOnGraph(std::shared_ptr<ir::Graph> graph) override;
    
private:
    // 计算操作的哈希值
    std::string computeOperationHash(std::shared_ptr<ir::Operation> operation);
    
    // 检查两个操作是否等价
    bool areOperationsEquivalent(std::shared_ptr<ir::Operation> op1, std::shared_ptr<ir::Operation> op2);
};

// 操作融合Pass
class OperationFusionPass : public GraphPass {
public:
    OperationFusionPass() : GraphPass("OperationFusionPass") {}
    
    bool runOnGraph(std::shared_ptr<ir::Graph> graph) override;
    
private:
    // 检查两个操作是否可以融合
    bool canFuse(std::shared_ptr<ir::Operation> producer, std::shared_ptr<ir::Operation> consumer);
    
    // 融合两个操作
    std::shared_ptr<ir::Operation> fuseOperations(std::shared_ptr<ir::Operation> producer, std::shared_ptr<ir::Operation> consumer);
    
    // 查找可融合的操作对
    std::vector<std::pair<std::shared_ptr<ir::Operation>, std::shared_ptr<ir::Operation>>> 
    findFusionCandidates(std::shared_ptr<ir::Graph> graph);
};

// 布局转换Pass
class LayoutTransformationPass : public GraphPass {
public:
    LayoutTransformationPass() : GraphPass("LayoutTransformationPass") {}
    
    bool runOnGraph(std::shared_ptr<ir::Graph> graph) override;
    
private:
    // 检查操作是否支持布局转换
    bool supportsLayoutTransformation(std::shared_ptr<ir::Operation> operation);
    
    // 应用布局转换
    bool applyLayoutTransformation(std::shared_ptr<ir::Operation> operation);
    
    // 创建布局转换操作
    std::shared_ptr<ir::Operation> createLayoutTransformOp(std::shared_ptr<ir::Value> input, const std::string& targetLayout);
};

// 内核融合Pass
class KernelFusionPass : public GraphPass {
public:
    KernelFusionPass() : GraphPass("KernelFusionPass") {}
    
    bool runOnGraph(std::shared_ptr<ir::Graph> graph) override;
    
private:
    // 检查操作是否可以作为内核融合的候选
    bool isKernelFusionCandidate(std::shared_ptr<ir::Operation> operation);
    
    // 查找可融合的内核组
    std::vector<std::vector<std::shared_ptr<ir::Operation>>> 
    findFusibleKernelGroups(std::shared_ptr<ir::Graph> graph);
    
    // 融合内核组
    std::shared_ptr<ir::Operation> fuseKernelGroup(const std::vector<std::shared_ptr<ir::Operation>>& kernelGroup);
};

// 内存优化Pass
class MemoryOptimizationPass : public GraphPass {
public:
    MemoryOptimizationPass() : GraphPass("MemoryOptimizationPass") {}
    
    bool runOnGraph(std::shared_ptr<ir::Graph> graph) override;
    
private:
    // 分析值的生命周期
    void analyzeValueLifetimes(std::shared_ptr<ir::Graph> graph);
    
    // 分配内存
    void allocateMemory(std::shared_ptr<ir::Graph> graph);
    
    // 值的生命周期信息
    struct ValueLifetime {
        int firstUse;
        int lastUse;
    };
    
    std::unordered_map<std::shared_ptr<ir::Value>, ValueLifetime> valueLifetimes;
};

// 图优化管道
class OptimizationPipeline {
public:
    OptimizationPipeline();
    
    // 运行优化管道
    bool run(std::shared_ptr<ir::Module> module);
    
    // 添加自定义Pass
    void addPass(std::shared_ptr<Pass> pass);
    
private:
    PassManager passManager;
    
    // 初始化默认Pass
    void initializeDefaultPasses();
};

} // namespace optimizer
} // namespace ai_compiler

#endif // OPTIMIZATION_PASSES_H
