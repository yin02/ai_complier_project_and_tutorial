#include <gtest/gtest.h>
#include "../src/frontend/lexer.h"
#include "../src/frontend/parser.h"
#include "../src/frontend/ast.h"

// 词法分析器测试
TEST(LexerTest, BasicTokenization) {
    std::string input = "var x: int = 5;";
    Lexer lexer(input);
    std::vector<Token> tokens = lexer.scanTokens();
    
    ASSERT_EQ(tokens.size(), 8); // 7个token + EOF
    EXPECT_EQ(tokens[0].type, TokenType::VAR);
    EXPECT_EQ(tokens[1].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[1].lexeme, "x");
    EXPECT_EQ(tokens[2].type, TokenType::COLON);
    EXPECT_EQ(tokens[3].type, TokenType::TYPE_INT);
    EXPECT_EQ(tokens[4].type, TokenType::ASSIGN);
    EXPECT_EQ(tokens[5].type, TokenType::INT_LITERAL);
    EXPECT_EQ(tokens[5].lexeme, "5");
    EXPECT_EQ(tokens[6].type, TokenType::SEMICOLON);
    EXPECT_EQ(tokens[7].type, TokenType::EOF_TOKEN);
}

TEST(LexerTest, OperatorsAndKeywords) {
    std::string input = "if (a + b * c == 10) { return true; }";
    Lexer lexer(input);
    std::vector<Token> tokens = lexer.scanTokens();
    
    ASSERT_GE(tokens.size(), 14); // 至少14个token
    EXPECT_EQ(tokens[0].type, TokenType::IF);
    EXPECT_EQ(tokens[1].type, TokenType::LPAREN);
    EXPECT_EQ(tokens[2].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[3].type, TokenType::PLUS);
    EXPECT_EQ(tokens[4].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[5].type, TokenType::STAR);
    EXPECT_EQ(tokens[6].type, TokenType::IDENTIFIER);
    EXPECT_EQ(tokens[7].type, TokenType::EQ);
    EXPECT_EQ(tokens[8].type, TokenType::INT_LITERAL);
    EXPECT_EQ(tokens[9].type, TokenType::RPAREN);
    EXPECT_EQ(tokens[10].type, TokenType::LBRACE);
    EXPECT_EQ(tokens[11].type, TokenType::RETURN);
    EXPECT_EQ(tokens[12].type, TokenType::BOOL_LITERAL);
    EXPECT_EQ(tokens[13].type, TokenType::SEMICOLON);
}

// 语法分析器测试
TEST(ParserTest, VariableDeclaration) {
    std::string input = "var x: int = 5;";
    Parser parser(input);
    std::shared_ptr<Program> program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->statements.size(), 1);
    
    auto varDecl = std::dynamic_pointer_cast<VarDeclaration>(program->statements[0]);
    ASSERT_NE(varDecl, nullptr);
    EXPECT_EQ(varDecl->name, "x");
    
    auto scalarType = std::dynamic_pointer_cast<ScalarType>(varDecl->type);
    ASSERT_NE(scalarType, nullptr);
    EXPECT_EQ(scalarType->kind, ScalarType::Kind::INT);
    
    auto literal = std::dynamic_pointer_cast<LiteralExpr>(varDecl->initializer);
    ASSERT_NE(literal, nullptr);
    EXPECT_EQ(literal->value, "5");
}

TEST(ParserTest, FunctionDeclaration) {
    std::string input = "func add(a: int, b: int): int { return a + b; }";
    Parser parser(input);
    std::shared_ptr<Program> program = parser.parse();
    
    ASSERT_NE(program, nullptr);
    ASSERT_EQ(program->statements.size(), 1);
    
    auto funcDecl = std::dynamic_pointer_cast<FunctionDeclaration>(program->statements[0]);
    ASSERT_NE(funcDecl, nullptr);
    EXPECT_EQ(funcDecl->name, "add");
    ASSERT_EQ(funcDecl->params.size(), 2);
    EXPECT_EQ(funcDecl->params[0]->name, "a");
    EXPECT_EQ(funcDecl->params[1]->name, "b");
    
    auto returnType = std::dynamic_pointer_cast<ScalarType>(funcDecl->returnType);
    ASSERT_NE(returnType, nullptr);
    EXPECT_EQ(returnType->kind, ScalarType::Kind::INT);
    
    ASSERT_EQ(funcDecl->body->statements.size(), 1);
    auto returnStmt = std::dynamic_pointer_cast<ReturnStatement>(funcDecl->body->statements[0]);
    ASSERT_NE(returnStmt, nullptr);
    
    auto binaryExpr = std::dynamic_pointer_cast<BinaryExpr>(returnStmt->expr);
    ASSERT_NE(binaryExpr, nullptr);
    EXPECT_EQ(binaryExpr->op, BinaryExpr::Op::ADD);
    
    auto leftVar = std::dynamic_pointer_cast<VarExpr>(binaryExpr->left);
    ASSERT_NE(leftVar, nullptr);
    EXPECT_EQ(leftVar->name, "a");
    
    auto rightVar = std::dynamic_pointer_cast<VarExpr>(binaryExpr->right);
    ASSERT_NE(rightVar, nullptr);
    EXPECT_EQ(rightVar->name, "b");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
