cmake_minimum_required(VERSION 3.10)
project(ai_compiler_course VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# 查找LLVM包
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# 包含LLVM头文件
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# 查找GoogleTest包
find_package(GTest REQUIRED)

# 设置包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/frontend
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ir/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optimizer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime
)

# 前端源文件
set(FRONTEND_SOURCES
    src/frontend/lexer.cpp
    src/frontend/parser.cpp
    src/frontend/ast.cpp
)

# IR源文件
set(IR_SOURCES
    src/ir/src/ast_to_ir.cpp
)

# 优化器源文件
set(OPTIMIZER_SOURCES
    src/optimizer/src/pass.cpp
    src/optimizer/src/optimization_passes.cpp
)

# 后端源文件
set(BACKEND_SOURCES
    src/backend/src/llvm_backend.cpp
)

# 运行时源文件
set(RUNTIME_SOURCES
    src/runtime/runtime.cpp
)

# 创建库
add_library(frontend STATIC ${FRONTEND_SOURCES})
add_library(ir STATIC ${IR_SOURCES})
add_library(optimizer STATIC ${OPTIMIZER_SOURCES})
add_library(backend STATIC ${BACKEND_SOURCES})
add_library(runtime STATIC ${RUNTIME_SOURCES})

# 设置库依赖
target_link_libraries(ir PRIVATE frontend)
target_link_libraries(optimizer PRIVATE ir)
target_link_libraries(backend PRIVATE optimizer)
target_link_libraries(backend PRIVATE ${LLVM_AVAILABLE_LIBS})

# 创建编译器可执行文件
add_executable(ai_compiler src/main.cpp)
target_link_libraries(ai_compiler PRIVATE frontend ir optimizer backend runtime)

# 单元测试
enable_testing()

# 前端测试
add_executable(frontend_tests tests/unit/frontend_tests.cpp ${FRONTEND_SOURCES})
target_link_libraries(frontend_tests PRIVATE GTest::GTest GTest::Main)
add_test(NAME FrontendTests COMMAND frontend_tests)

# IR测试
add_executable(ir_tests tests/unit/ir_tests.cpp ${FRONTEND_SOURCES} ${IR_SOURCES})
target_link_libraries(ir_tests PRIVATE GTest::GTest GTest::Main)
add_test(NAME IRTests COMMAND ir_tests)

# 优化器测试
add_executable(optimizer_tests tests/unit/optimizer_tests.cpp ${FRONTEND_SOURCES} ${IR_SOURCES} ${OPTIMIZER_SOURCES})
target_link_libraries(optimizer_tests PRIVATE GTest::GTest GTest::Main)
add_test(NAME OptimizerTests COMMAND optimizer_tests)

# 后端测试
add_executable(backend_tests tests/unit/backend_tests.cpp ${FRONTEND_SOURCES} ${IR_SOURCES} ${OPTIMIZER_SOURCES} ${BACKEND_SOURCES})
target_link_libraries(backend_tests PRIVATE GTest::GTest GTest::Main ${LLVM_AVAILABLE_LIBS})
add_test(NAME BackendTests COMMAND backend_tests)

# 集成测试
add_executable(integration_tests tests/integration/integration_tests.cpp)
target_link_libraries(integration_tests PRIVATE frontend ir optimizer backend runtime GTest::GTest GTest::Main)
add_test(NAME IntegrationTests COMMAND integration_tests)

# 安装规则
install(TARGETS ai_compiler DESTINATION bin)
install(TARGETS frontend ir optimizer backend runtime DESTINATION lib)
install(DIRECTORY src/ DESTINATION include/ai_compiler
        FILES_MATCHING PATTERN "*.h")
install(DIRECTORY docs/ DESTINATION share/ai_compiler/docs)
install(DIRECTORY examples/ DESTINATION share/ai_compiler/examples)

# 打包规则
set(CPACK_PACKAGE_NAME "ai_compiler_course")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "AI Compiler Course")
set(CPACK_PACKAGE_VENDOR "AI Compiler Team")
set(CPACK_GENERATOR "TGZ")
include(CPack)
