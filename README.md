# AI 编译器交互式课程

这是一个带有 LLVM 前端和后端的 AI 编译器项目，支持从简单 DSL 到 LLVM IR，再到可执行代码的转换。本项目旨在提供一个完整的交互式学习环境，帮助您理解和实践 AI 编译器的核心概念和技术。

## 项目结构

```
ai_compiler_course/
├── src/                    # 源代码
│   ├── frontend/           # 前端：词法分析器、语法分析器、AST 生成
│   ├── ir/                 # 中间表示：DSL IR 定义和操作
│   ├── optimizer/          # 优化器：图优化、布局转换、内核融合等
│   ├── backend/            # 后端：LLVM IR 生成、代码生成
│   └── runtime/            # 运行时：模型执行环境
├── docs/                   # 文档
│   ├── tutorials/          # 教程
│   ├── api/                # API 文档
│   └── examples/           # 示例文档
├── examples/               # 示例代码
│   ├── basic/              # 基础示例
│   └── advanced/           # 高级示例
└── tests/                  # 测试
    ├── unit/               # 单元测试
    └── integration/        # 集成测试
```

## 课程内容

1. **前端设计与实现**
   - 词法分析器设计
   - 语法分析器实现
   - AST 构建

2. **中间表示 (IR)**
   - DSL 到 IR 的转换
   - IR 数据结构设计
   - IR 操作和转换

3. **图优化技术**
   - 计算图表示
   - 图优化 Pass
   - 布局转换
   - 内核融合

4. **LLVM 后端集成**
   - IR 到 LLVM IR 的转换
   - LLVM Pass 开发
   - 代码生成

5. **运行时系统**
   - 模型执行环境
   - 内存管理
   - 性能分析

## 先决条件

- C++ 编程基础
- 编译原理基础知识
- LLVM 基本概念
- 深度学习基础

## 开始使用

请按照 `docs/tutorials/` 目录中的指南开始学习和实践。
# ai_complier_project_and_tutorial
