# AI编译器项目安装和使用指南

本文档提供了AI编译器项目的安装、构建和使用说明。

## 1. 系统要求

- 操作系统：Linux (Ubuntu 20.04+)、macOS 10.15+或Windows 10+
- 编译器：支持C++17的编译器（GCC 7+、Clang 6+或MSVC 19.14+）
- 构建工具：CMake 3.10+、Ninja或Make
- 依赖项：
  - LLVM 12.0+
  - Python 3.6+
  - GoogleTest (用于测试)

## 2. 安装依赖项

### Ubuntu

```bash
# 安装基本工具
sudo apt update
sudo apt install -y build-essential cmake ninja-build git

# 安装LLVM
sudo apt install -y llvm-12-dev clang-12 lld-12

# 安装Python
sudo apt install -y python3 python3-pip

# 安装GoogleTest
sudo apt install -y libgtest-dev
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib
```

### macOS

```bash
# 安装Homebrew（如果尚未安装）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装基本工具
brew install cmake ninja git

# 安装LLVM
brew install llvm@12

# 安装Python
brew install python3

# 安装GoogleTest
brew install googletest
```

### Windows

建议使用Windows Subsystem for Linux (WSL)或使用Visual Studio 2019+和vcpkg：

```bash
# 使用vcpkg安装依赖项
vcpkg install llvm:x64-windows
vcpkg install gtest:x64-windows
```

## 3. 获取源代码

```bash
git clone https://github.com/your-username/ai_compiler_course.git
cd ai_compiler_course
```

## 4. 构建项目

```bash
# 创建构建目录
mkdir build
cd build

# 配置项目
cmake -G Ninja ..

# 构建项目
ninja

# 运行测试
ninja test
```

## 5. 安装项目

```bash
# 安装到系统（可能需要管理员权限）
sudo ninja install
```

或者安装到自定义目录：

```bash
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install/dir -G Ninja ..
ninja install
```

## 6. 使用编译器

### 6.1 编译DSL程序

```bash
# 编译DSL程序
ai_compiler path/to/your/program.dsl -o output_file

# 使用不同优化级别
ai_compiler path/to/your/program.dsl -o output_file -O2

# 生成LLVM IR
ai_compiler path/to/your/program.dsl -emit-llvm -o output_file.ll

# 生成汇编代码
ai_compiler path/to/your/program.dsl -S -o output_file.s
```

### 6.2 运行编译后的程序

```bash
# 运行编译后的程序
./output_file

# 传递参数
./output_file arg1 arg2
```

## 7. 示例程序

项目包含多个示例程序，位于`examples`目录：

### 7.1 基础示例

```bash
# 编译并运行MLP示例
ai_compiler examples/basic/mlp.dsl -o mlp
./mlp
```

### 7.2 高级示例

```bash
# 编译并运行CNN示例
ai_compiler examples/advanced/cnn.dsl -o cnn
./cnn
```

## 8. 交互式课程

项目包含交互式课程材料，位于`docs/tutorials`目录：

1. DSL规范：`docs/tutorials/01_dsl_specification.md`
2. 前端教程：`docs/tutorials/02_frontend_tutorial.md`
3. IR教程：`docs/tutorials/03_ir_tutorial.md`
4. 图优化教程：`docs/tutorials/04_graph_optimization_tutorial.md`
5. LLVM后端教程：`docs/tutorials/05_llvm_backend_tutorial.md`

## 9. 故障排除

### 9.1 构建问题

- **找不到LLVM**：确保LLVM已正确安装，并设置`LLVM_DIR`环境变量：
  ```bash
  export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
  ```

- **找不到GoogleTest**：确保GoogleTest已正确安装，并设置`GTest_DIR`环境变量：
  ```bash
  export GTest_DIR=/path/to/googletest/lib/cmake/GTest
  ```

### 9.2 运行问题

- **找不到共享库**：确保库路径已正确设置：
  ```bash
  export LD_LIBRARY_PATH=/path/to/install/dir/lib:$LD_LIBRARY_PATH
  ```

## 10. 贡献

欢迎贡献代码、报告问题或提出改进建议。请参阅`CONTRIBUTING.md`文件了解详情。

## 11. 许可证

本项目采用MIT许可证。请参阅`LICENSE`文件了解详情。
