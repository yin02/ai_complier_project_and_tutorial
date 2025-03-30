#include "../include/pass.h"

namespace ai_compiler {
namespace optimizer {

// GraphPass 实现
bool GraphPass::run(std::shared_ptr<ir::Module> module) {
    bool changed = false;
    
    // 对模块中的每个图运行Pass
    for (auto& graph : module->graphs) {
        changed |= runOnGraph(graph);
    }
    
    return changed;
}

// OperationPass 实现
bool OperationPass::run(std::shared_ptr<ir::Module> module) {
    bool changed = false;
    
    // 对模块中的每个图运行Pass
    for (auto& graph : module->graphs) {
        changed |= runOnGraph(graph);
    }
    
    return changed;
}

bool OperationPass::runOnGraph(std::shared_ptr<ir::Graph> graph) {
    bool changed = false;
    
    // 对图中的每个操作运行Pass
    for (auto& op : graph->operations) {
        changed |= runOnOperation(op);
    }
    
    return changed;
}

// PassManager 实现
PassManager::PassManager() {}

void PassManager::addPass(std::shared_ptr<Pass> pass) {
    passes.push_back(pass);
}

bool PassManager::run(std::shared_ptr<ir::Module> module) {
    bool changed = false;
    
    // 按顺序运行所有Pass
    for (auto& pass : passes) {
        changed |= pass->run(module);
    }
    
    return changed;
}

void PassManager::clear() {
    passes.clear();
}

// PassRegistry 实现
PassRegistry& PassRegistry::getInstance() {
    static PassRegistry instance;
    return instance;
}

void PassRegistry::registerPass(const std::string& name, PassCreator creator) {
    registry[name] = creator;
}

std::shared_ptr<Pass> PassRegistry::createPass(const std::string& name) {
    auto it = registry.find(name);
    if (it != registry.end()) {
        return it->second();
    }
    return nullptr;
}

std::vector<std::string> PassRegistry::getRegisteredPassNames() const {
    std::vector<std::string> names;
    for (const auto& entry : registry) {
        names.push_back(entry.first);
    }
    return names;
}

} // namespace optimizer
} // namespace ai_compiler
