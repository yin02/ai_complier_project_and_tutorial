#include "../include/optimization_passes.h"
#include <algorithm>
#include <cassert>
#include <sstream>

namespace ai_compiler {
namespace optimizer {

//===----------------------------------------------------------------------===//
// ConstantFoldingPass Implementation
//===----------------------------------------------------------------------===//

bool ConstantFoldingPass::runOnOperation(std::shared_ptr<ir::Operation> operation) {
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
        
        // 从图中移除原操作（简化实现，实际应该更新图的操作列表）
        auto& ops = graph->operations;
        ops.erase(std::remove(ops.begin(), ops.end(), operation), ops.end());
    }
    
    return true;
}

bool ConstantFoldingPass::canFold(std::shared_ptr<ir::Operation> operation) {
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

std::shared_ptr<ir::Value> ConstantFoldingPass::evaluateConstantOp(std::shared_ptr<ir::Operation> operation) {
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

std::shared_ptr<ir::Operation> ConstantFoldingPass::createConstantOp(const std::string& value, std::shared_ptr<ir::Type> type) {
    auto constOp = std::make_shared<ir::ConstantOp>();
    constOp->setAttribute("value", value);
    
    auto resultValue = std::make_shared<ir::Value>("const_" + value, type);
    constOp->addResult(resultValue);
    
    return constOp;
}

//===----------------------------------------------------------------------===//
// DeadCodeEliminationPass Implementation
//===----------------------------------------------------------------------===//

bool DeadCodeEliminationPass::runOnGraph(std::shared_ptr<ir::Graph> graph) {
    // 标记活跃值
    std::unordered_set<std::shared_ptr<ir::Value>> liveValues;
    markLiveValues(graph, liveValues);
    
    // 移除死代码
    return removeDeadOperations(graph, liveValues);
}

void DeadCodeEliminationPass::markLiveValues(std::shared_ptr<ir::Graph> graph, 
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

bool DeadCodeEliminationPass::removeDeadOperations(std::shared_ptr<ir::Graph> graph, 
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

//===----------------------------------------------------------------------===//
// CommonSubexpressionEliminationPass Implementation
//===----------------------------------------------------------------------===//

bool CommonSubexpressionEliminationPass::runOnGraph(std::shared_ptr<ir::Graph> graph) {
    bool changed = false;
    
    // 操作哈希到操作的映射
    std::unordered_map<std::string, std::shared_ptr<ir::Operation>> opHashMap;
    
    // 收集需要删除的操作
    std::vector<std::shared_ptr<ir::Operation>> opsToRemove;
    
    // 遍历图中的所有操作
    for (auto& op : graph->operations) {
        // 计算操作的哈希值
        std::string opHash = computeOperationHash(op);
        
        // 检查是否已经有等价的操作
        auto it = opHashMap.find(opHash);
        if (it != opHashMap.end()) {
            // 找到等价操作，检查是否真的等价
            if (areOperationsEquivalent(op, it->second)) {
                // 替换操作的结果
                for (size_t i = 0; i < op->results.size(); ++i) {
                    auto& result = op->results[i];
                    auto& equivResult = it->second->results[i];
                    
                    // 替换所有使用此结果的操作
                    for (auto& userWeak : result->users) {
                        if (auto user = userWeak.lock()) {
                            // 替换操作数
                            for (size_t j = 0; j < user->operands.size(); ++j) {
                                if (user->operands[j] == result) {
                                    user->operands[j] = equivResult;
                                }
                            }
                        }
                    }
                }
                
                // 标记操作为待删除
                opsToRemove.push_back(op);
                changed = true;
            }
        } else {
            // 添加到哈希映射
            opHashMap[opHash] = op;
        }
    }
    
    // 从图中移除冗余操作
    for (auto& op : opsToRemove) {
        auto& ops = graph->operations;
        ops.erase(std::remove(ops.begin(), ops.end(), op), ops.end());
    }
    
    return changed;
}

std::string CommonSubexpressionEliminationPass::computeOperationHash(std::shared_ptr<ir::Operation> operation) {
    std::stringstream ss;
    
    // 添加操作类型
    ss << operation->opType << ":";
    
    // 添加操作数类型和定义操作
    for (auto& operand : operation->operands) {
        ss << operand->type->toString() << "@";
        if (auto defOp = operand->definingOp.lock()) {
            ss << defOp->opType;
        } else {
            ss << "input";
        }
        ss << ",";
    }
    
    // 添加属性
    for (const auto& attr : operation->attributes) {
        ss << attr.first << "=" << attr.second << ",";
    }
    
    return ss.str();
}

bool CommonSubexpressionEliminationPass::areOperationsEquivalent(std::shared_ptr<ir::Operation> op1, std::shared_ptr<ir::Operation> op2) {
    // 检查操作类型
    if (op1->opType != op2->opType) {
        return false;
    }
    
    // 检查操作数数量
    if (op1->operands.size() != op2->operands.size()) {
        return false;
    }
    
    // 检查结果数量
    if (op1->results.size() != op2->results.size()) {
        return false;
    }
    
    // 检查操作数
    for (size_t i = 0; i < op1->operands.size(); ++i) {
        auto& operand1 = op1->operands[i];
        auto& operand2 = op2->operands[i];
        
        // 检查类型
        if (!operand1->type->equals(*operand2->type)) {
            return false;
        }
        
        // 检查定义操作
        auto defOp1 = operand1->definingOp.lock();
        auto defOp2 = operand2->definingOp.lock();
        
        if ((defOp1 && !defOp2) || (!defOp1 && defOp2)) {
            return false;
        }
        
        if (defOp1 && defOp2 && defOp1 != defOp2) {
            return false;
        }
    }
    
    // 检查属性
    if (op1->attributes.size() != op2->attributes.size()) {
        return false;
    }
    
    for (const auto& attr : op1->attributes) {
        auto it = op2->attributes.find(attr.first);
        if (it == op2->attributes.end() || it->second != attr.second) {
            return false;
        }
    }
    
    return true;
}

//===----------------------------------------------------------------------===//
// OperationFusionPass Implementation
//===----------------------------------------------------------------------===//

bool OperationFusionPass::runOnGraph(std::shared_ptr<ir::Graph> graph) {
    bool changed = false;
    
    // 查找可融合的操作对
    auto fusionCandidates = findFusionCandidates(graph);
    
    // 融合操作
    for (const auto& candidatePair : fusionCandidates) {
        auto producer = candidatePair.first;
        auto consumer = candidatePair.second;
        
        // 融合操作
        auto fusedOp = fuseOperations(producer, consumer);
        if (fusedOp) {
            // 将融合后的操作添加到图中
            graph->addOperation(fusedOp);
            
            // 从图中移除原操作
            auto& ops = graph->operations;
            ops.erase(std::remove(ops.begin(), ops.end(), producer), ops.end());
            ops.erase(std::remove(ops.begin(), ops.end(), consumer), ops.end());
            
            changed = true;
        }
    }
    
    return changed;
}

std::vector<std::pair<std::shared_ptr<ir::Operation>, std::shared_ptr<ir::Operation>>> 
OperationFusionPass::findFusionCandidates(std::shared_ptr<ir::Graph> graph) {
    std::vector<std::pair<std::shared_ptr<ir::Operation>, std::shared_ptr<ir::Operation>>> candidates;
    
    // 构建操作之间的生产者-消费者关系
    for (auto& consumer : graph->operations) {
        for (auto& operand : consumer->operands) {
            if (auto defOp = operand->definingOp.lock()) {
                // 检查是否可以融合
                if (canFuse(defOp, consumer)) {
                    candidates.emplace_back(defOp, consumer);
                }
            }
        }
    }
    
    return candidates;
}

bool OperationFusionPass::canFuse(std::shared_ptr<ir::Operation> producer, std::shared_ptr<ir::Operation> consumer) {
    // 简化实现：只考虑特定类型的操作融合
    
    // 例如：融合 relu + add
    if (producer->opType == "relu" && consumer->opType == "add") {
        return true;
    }
    
    // 例如：融合 matmul + add
    if (producer->opType == "matmul" && consumer->opType == "add") {
        return true;
    }
    
    // 例如：融合 conv2d + relu
    if (producer->opType == "conv2d" && consumer->opType == "relu") {
        return true;
    }
    
    return false;
}

std::shared_ptr<ir::Operation> OperationFusionPass::fuseOperations(
    std::shared_ptr<ir::Operation> producer, std::shared_ptr<ir::Operation> consumer) {
    // 创建融合操作
    std::string fusedOpType = producer->opType + "_" + consumer->opType;
    auto fusedOp = std::make_shared<ir::Operation>(fusedOpType);
    
    // 设置融合操作的操作数（除了producer的结果）
    for (auto& operand : producer->operands) {
        fusedOp->addOperand(operand);
    }
    
    for (auto& operand : consumer->operands) {
        // 跳过producer的结果
        bool isProducerResult = false;
        for (auto& result : producer->results) {
            if (operand == result) {
                isProducerResult = true;
                break;
            }
        }
        
        if (!isProducerResult) {
            fusedOp->addOperand(operand);
        }
    }
    
    // 设置融合操作的结果
    for (auto& result : consumer->results) {
        fusedOp->addResult(result);
    }
    
    // 设置融合操作的属性
    for (const auto& attr : producer->attributes) {
        fusedOp->setAttribute(attr.first, attr.second);
    }
    
    for (const auto& attr : consumer->attributes) {
        fusedOp->setAttribute(attr.first, attr.second);
    }
    
    // 设置融合标志
    fusedOp->setAttribute("fused", "true");
    
    return fusedOp;
}

//===----------------------------------------------------------------------===//
// LayoutTransformationPass Implementation
//===----------------------------------------------------------------------===//

bool LayoutTransformationPass::runOnGraph(std::shared_ptr<ir::Graph> graph) {
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

bool LayoutTransformationPass::supportsLayoutTransformation(std::shared_ptr<ir::Operation> operation) {
    // 简化实现：只考虑特定类型的操作
    return operation->opType == "conv2d" || 
           operation->opType == "max_pool" || 
           operation->opType == "avg_pool";
}

bool LayoutTransformationPass::applyLayoutTransformation(std::shared_ptr<ir::Operation> operation) {
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

std::shared_ptr<ir::Operation> LayoutTransformationPass::createLayoutTransformOp(
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

//===----------------------------------------------------------------------===//
// KernelFusionPass Implementation
//===----------------------------------------------------------------------===//

bool KernelFusionPass::runOnGraph(std::shared_ptr<ir::Graph> graph) {
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

bool KernelFusionPass::isKernelFusionCandidate(std::shared_ptr<ir::Operation> operation) {
    // 简化实现：只考虑特定类型的操作
    return operation->opType == "conv2d" || 
           operation->opType == "matmul" || 
           operation->opType == "relu" || 
           operation->opType == "add";
}

std::vector<std::vector<std::shared_ptr<ir::Operation>>> 
KernelFusionPass::findFusibleKernelGroups(std::shared_ptr<ir::Graph> graph) {
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

std::shared_ptr<ir::Operation> KernelFusionPass::fuseKernelGroup(
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

//===----------------------------------------------------------------------===//
// MemoryOptimizationPass Implementation
//===----------------------------------------------------------------------===//

bool MemoryOptimizationPass::runOnGraph(std::shared_ptr<ir::Graph> graph) {
    // 分析值的生命周期
    analyzeValueLifetimes(graph);
    
    // 分配内存
    allocateMemory(graph);
    
    return true;
}

void MemoryOptimizationPass::analyzeValueLifetimes(std::shared_ptr<ir::Graph> graph) {
    valueLifetimes.clear();
    
    // 为每个操作分配一个时间戳
    int timestamp = 0;
    
    // 初始化所有值的生命周期
    for (auto& op : graph->operations) {
        for (auto& result : op->results) {
            valueLifetimes[result] = {timestamp, timestamp};
        }
        
        timestamp++;
    }
    
    // 更新值的最后使用时间
    timestamp = 0;
    for (auto& op : graph->operations) {
        for (auto& operand : op->operands) {
            auto it = valueLifetimes.find(operand);
            if (it != valueLifetimes.end()) {
                it->second.lastUse = std::max(it->second.lastUse, timestamp);
            }
        }
        
        timestamp++;
    }
    
    // 图输出的生命周期延长到最后
    for (auto& output : graph->outputs) {
        auto it = valueLifetimes.find(output);
        if (it != valueLifetimes.end()) {
            it->second.lastUse = timestamp;
        }
    }
}

void MemoryOptimizationPass::allocateMemory(std::shared_ptr<ir::Graph> graph) {
    // 简化实现：使用贪心算法分配内存
    
    // 按照第一次使用时间排序
    std::vector<std::shared_ptr<ir::Value>> values;
    for (const auto& entry : valueLifetimes) {
        values.push_back(entry.first);
    }
    
    std::sort(values.begin(), values.end(), [this](const std::shared_ptr<ir::Value>& a, const std::shared_ptr<ir::Value>& b) {
        return valueLifetimes[a].firstUse < valueLifetimes[b].firstUse;
    });
    
    // 分配内存
    std::vector<std::pair<std::shared_ptr<ir::Value>, int>> allocations; // 值和内存偏移
    int totalMemory = 0;
    
    for (auto& value : values) {
        // 查找可以重用的内存
        bool found = false;
        for (auto& alloc : allocations) {
            auto& allocValue = alloc.first;
            auto& offset = alloc.second;
            
            // 检查生命周期是否重叠
            if (valueLifetimes[allocValue].lastUse < valueLifetimes[value].firstUse) {
                // 可以重用
                alloc.first = value;
                found = true;
                break;
            }
        }
        
        if (!found) {
            // 分配新内存
            allocations.emplace_back(value, totalMemory);
            totalMemory += 1; // 简化：每个值占用1个内存单元
        }
    }
    
    // 设置内存分配属性
    for (const auto& alloc : allocations) {
        auto& value = alloc.first;
        auto& offset = alloc.second;
        
        // 获取定义此值的操作
        if (auto defOp = value->definingOp.lock()) {
            defOp->setAttribute("mem_offset", std::to_string(offset));
        }
    }
    
    // 设置总内存大小
    for (auto& op : graph->operations) {
        op->setAttribute("total_memory", std::to_string(totalMemory));
    }
}

//===----------------------------------------------------------------------===//
// OptimizationPipeline Implementation
//===----------------------------------------------------------------------===//

OptimizationPipeline::OptimizationPipeline() {
    initializeDefaultPasses();
}

bool OptimizationPipeline::run(std::shared_ptr<ir::Module> module) {
    return passManager.run(module);
}

void OptimizationPipeline::addPass(std::shared_ptr<Pass> pass) {
    passManager.addPass(pass);
}

void OptimizationPipeline::initializeDefaultPasses() {
    // 添加默认的优化Pass
    passManager.addPass(std::make_shared<ConstantFoldingPass>());
    passManager.addPass(std::make_shared<DeadCodeEliminationPass>());
    passManager.addPass(std::make_shared<CommonSubexpressionEliminationPass>());
    passManager.addPass(std::make_shared<OperationFusionPass>());
    passManager.addPass(std::make_shared<LayoutTransformationPass>());
    passManager.addPass(std::make_shared<KernelFusionPass>());
    passManager.addPass(std::make_shared<MemoryOptimizationPass>());
}

} // namespace optimizer
} // namespace ai_compiler
