#ifndef PASS_H
#define PASS_H

#include "../../ir/include/ir.h"
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>

namespace ai_compiler {
namespace optimizer {

// 优化Pass基类
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
    virtual bool run(std::shared_ptr<ir::Module> module) override;
    
    // 在单个图上运行Pass
    virtual bool runOnGraph(std::shared_ptr<ir::Graph> graph) = 0;
};

// 操作级Pass
class OperationPass : public Pass {
public:
    OperationPass(const std::string& name) : Pass(name, PassType::OPERATION_PASS) {}
    
    // 运行操作级Pass
    virtual bool run(std::shared_ptr<ir::Module> module) override;
    
    // 在单个图上运行Pass
    virtual bool runOnGraph(std::shared_ptr<ir::Graph> graph);
    
    // 在单个操作上运行Pass
    virtual bool runOnOperation(std::shared_ptr<ir::Operation> operation) = 0;
};

// Pass管理器
class PassManager {
public:
    PassManager();
    
    // 添加Pass
    void addPass(std::shared_ptr<Pass> pass);
    
    // 运行所有Pass
    bool run(std::shared_ptr<ir::Module> module);
    
    // 清空所有Pass
    void clear();
    
private:
    std::vector<std::shared_ptr<Pass>> passes;
};

// Pass注册表
class PassRegistry {
public:
    using PassCreator = std::function<std::shared_ptr<Pass>()>;
    
    // 获取单例实例
    static PassRegistry& getInstance();
    
    // 注册Pass
    void registerPass(const std::string& name, PassCreator creator);
    
    // 创建Pass
    std::shared_ptr<Pass> createPass(const std::string& name);
    
    // 获取所有注册的Pass名称
    std::vector<std::string> getRegisteredPassNames() const;
    
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
        ai_compiler::optimizer::PassRegistry::getInstance().registerPass( \
            #PassClass, []() { return std::make_shared<PassClass>(); }); \
        return true; \
    }();

} // namespace optimizer
} // namespace ai_compiler

#endif // PASS_H
