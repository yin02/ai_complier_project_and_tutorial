#ifndef PARSER_H
#define PARSER_H

#include "lexer.h"
#include "ast.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <stdexcept>

namespace ai_compiler {

// 解析错误异常
class ParseError : public std::runtime_error {
public:
    ParseError(const std::string& message) : std::runtime_error(message) {}
};

// 语法分析器类
class Parser {
public:
    Parser(const std::string& source);
    
    // 解析整个程序
    std::shared_ptr<Program> parse();
    
private:
    Lexer lexer;
    Token currentToken;
    std::unordered_map<std::string, std::shared_ptr<Type>> typeEnvironment;
    
    // 辅助方法
    void advance();
    bool match(TokenType type);
    bool check(TokenType type) const;
    Token consume(TokenType type, const std::string& message);
    ParseError error(const std::string& message);
    void synchronize();
    
    // 递归下降解析方法
    std::shared_ptr<Program> parseProgram();
    std::shared_ptr<Statement> parseDeclaration();
    std::shared_ptr<ImportStatement> parseImportStatement();
    std::shared_ptr<FunctionDeclaration> parseFunctionDeclaration();
    std::shared_ptr<OperationDeclaration> parseOperationDeclaration();
    std::shared_ptr<GraphDeclaration> parseGraphDeclaration();
    std::vector<Parameter> parseParameters();
    Parameter parseParameter();
    std::shared_ptr<Statement> parseStatement();
    std::shared_ptr<VarDeclaration> parseVarDeclaration();
    std::shared_ptr<Statement> parseIfStatement();
    std::shared_ptr<Statement> parseForStatement();
    std::shared_ptr<Statement> parseReturnStatement();
    std::shared_ptr<Block> parseBlock();
    std::shared_ptr<Statement> parseExpressionStatement();
    
    std::shared_ptr<Expression> parseExpression();
    std::shared_ptr<Expression> parseAssignment();
    std::shared_ptr<Expression> parseLogicalOr();
    std::shared_ptr<Expression> parseLogicalAnd();
    std::shared_ptr<Expression> parseEquality();
    std::shared_ptr<Expression> parseComparison();
    std::shared_ptr<Expression> parseTerm();
    std::shared_ptr<Expression> parseFactor();
    std::shared_ptr<Expression> parseUnary();
    std::shared_ptr<Expression> parseCall();
    std::shared_ptr<Expression> parsePrimary();
    
    std::shared_ptr<Type> parseType();
    std::shared_ptr<TensorType> parseTensorType();
};

} // namespace ai_compiler

#endif // PARSER_H
