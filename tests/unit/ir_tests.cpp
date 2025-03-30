#include <gtest/gtest.h>
#include "../src/ir/include/ir.h"
#include "../src/ir/include/ast_to_ir.h"
#include "../src/frontend/parser.h"

// IR类型系统测试
TEST(IRTypeTest, ScalarTypes) {
    auto floatType = TypeUtils::createFloat32Type();
    auto intType = TypeUtils::createInt32Type();
    auto boolType = TypeUtils::createBoolType();
    
    EXPECT_EQ(floatType->toString(), "float");
    EXPECT_EQ(intType->toString(), "int");
    EXPECT_EQ(boolType->toString(), "bool");
    
    EXPECT_TRUE(floatType->equals(*floatType));
    EXPECT_FALSE(floatType->equals(*intType));
}

TEST(IRTypeTest, TensorTypes) {
    auto elementType = TypeUtils::createFloat32Type();
    auto tensorType1 = TypeUtils::createTensorType(elementType, {1, 3, 224, 224});
    auto tensorType2 = TypeUtils::createTensorType(elementType, {1, 3, 224, 224});
    auto tensorType3 = TypeUtils::createTensorType(elementType, {1, 3, 224, 225});
    
    EXPECT_TRUE(tensorType1->equals(*tensorType2));
    EXPECT_FALSE(tensorType1->equals(*tensorType3));
    
    EXPECT_EQ(tensorType1->toString(), "tensor<float, [1, 3, 224, 224]>");
}

// IR值和操作测试
TEST(IRValueTest, ValueCreation) {
    auto floatType = TypeUtils::createFloat32Type();
    auto value = std::make_shared<ir::Value>("test_value", floatType);
    
    EXPECT_EQ(value->name, "test_value");
    EXPECT_EQ(value->type, floatType);
    EXPECT_TRUE(value->users.empty());
    EXPECT_TRUE(value->definingOp.expired());
}

TEST(IROperationTest, OperationCreation) {
    auto floatType = TypeUtils::createFloat32Type();
    auto input1 = std::make_shared<ir::Value>("input1", floatType);
    auto input2 = std::make_shared<ir::Value>("input2", floatType);
    auto result = std::make_shared<ir::Value>("result", floatType);
    
    auto addOp = std::make_shared<ir::BinaryOp>(ir::BinaryOp::OpKind::ADD);
    addOp->addOperand(input1);
    addOp->addOperand(input2);
    addOp->addResult(result);
    
    EXPECT_EQ(addOp->opType, "add");
    ASSERT_EQ(addOp->operands.size(), 2);
    EXPECT_EQ(addOp->operands[0], input1);
    EXPECT_EQ(addOp->operands[1], input2);
    ASSERT_EQ(addOp->results.size(), 1);
    EXPECT_EQ(addOp->results[0], result);
    
    EXPECT_EQ(result->definingOp.lock(), addOp);
    ASSERT_EQ(input1->users.size(), 1);
    EXPECT_EQ(input1->users[0].lock(), addOp);
    ASSERT_EQ(input2->users.size(), 1);
    EXPECT_EQ(input2->users[0].lock(), addOp);
}

// AST到IR转换测试
TEST(ASTToIRTest, SimpleExpression) {
    // 创建一个简单的AST：1 + 2
    auto one = std::make_shared<LiteralExpr>("1", LiteralExpr::Kind::INT);
    auto two = std::make_shared<LiteralExpr>("2", LiteralExpr::Kind::INT);
    auto add = std::make_shared<BinaryExpr>(BinaryExpr::Op::ADD, one, two);
    
    // 创建一个包含表达式语句的程序
    auto exprStmt = std::make_shared<ExpressionStatement>(add);
    std::vector<std::shared_ptr<Statement>> statements = {exprStmt};
    auto program = std::make_shared<Program>(statements);
    
    // 转换为IR
    ASTToIRConverter converter;
    auto module = converter.convert(program);
    
    // 验证IR
    ASSERT_NE(module, nullptr);
    ASSERT_GE(module->graphs.size(), 1);
    
    auto graph = module->graphs[0];
    ASSERT_GE(graph->operations.size(), 3); // 至少有两个常量操作和一个加法操作
    
    // 查找加法操作
    std::shared_ptr<ir::Operation> addOp = nullptr;
    for (auto& op : graph->operations) {
        if (op->opType == "add") {
            addOp = op;
            break;
        }
    }
    
    ASSERT_NE(addOp, nullptr);
    ASSERT_EQ(addOp->operands.size(), 2);
    ASSERT_EQ(addOp->results.size(), 1);
    
    // 验证操作数是常量
    for (auto& operand : addOp->operands) {
        auto defOp = operand->definingOp.lock();
        ASSERT_NE(defOp, nullptr);
        EXPECT_EQ(defOp->opType, "constant");
    }
}

TEST(ASTToIRTest, FunctionConversion) {
    // 解析一个简单的函数
    std::string input = "func add(a: float, b: float): float { return a + b; }";
    Parser parser(input);
    auto program = parser.parse();
    
    // 转换为IR
    ASTToIRConverter converter;
    auto module = converter.convert(program);
    
    // 验证IR
    ASSERT_NE(module, nullptr);
    ASSERT_EQ(module->graphs.size(), 1);
    
    auto graph = module->graphs[0];
    EXPECT_EQ(graph->name, "add");
    ASSERT_EQ(graph->inputs.size(), 2);
    EXPECT_EQ(graph->inputs[0]->name, "a");
    EXPECT_EQ(graph->inputs[1]->name, "b");
    ASSERT_EQ(graph->outputs.size(), 1);
    
    // 查找加法操作
    std::shared_ptr<ir::Operation> addOp = nullptr;
    for (auto& op : graph->operations) {
        if (op->opType == "add") {
            addOp = op;
            break;
        }
    }
    
    ASSERT_NE(addOp, nullptr);
    ASSERT_EQ(addOp->operands.size(), 2);
    EXPECT_EQ(addOp->operands[0], graph->inputs[0]);
    EXPECT_EQ(addOp->operands[1], graph->inputs[1]);
    ASSERT_EQ(addOp->results.size(), 1);
    EXPECT_EQ(addOp->results[0], graph->outputs[0]);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
