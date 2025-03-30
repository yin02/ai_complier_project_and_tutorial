#include "parser.h"
#include <iostream>
#include <sstream>

namespace ai_compiler {

// 构造函数
Parser::Parser(const std::string& source) : lexer(source) {
    advance(); // 初始化当前词法单元
}

// 前进到下一个词法单元
void Parser::advance() {
    currentToken = lexer.scanToken();
}

// 检查当前词法单元是否匹配指定类型，如果匹配则消费它
bool Parser::match(TokenType type) {
    if (!check(type)) return false;
    advance();
    return true;
}

// 检查当前词法单元是否为指定类型
bool Parser::check(TokenType type) const {
    return currentToken.type == type;
}

// 消费当前词法单元，如果类型不匹配则抛出错误
Token Parser::consume(TokenType type, const std::string& message) {
    if (check(type)) {
        Token token = currentToken;
        advance();
        return token;
    }
    
    throw error(message);
}

// 创建解析错误
ParseError Parser::error(const std::string& message) {
    std::stringstream ss;
    ss << "Error at line " << currentToken.line << ": " << message;
    ss << " (got '" << currentToken.lexeme << "')";
    return ParseError(ss.str());
}

// 错误恢复：跳过词法单元直到找到语句边界
void Parser::synchronize() {
    advance();
    
    while (currentToken.type != TokenType::EOF_TOKEN) {
        if (currentToken.type == TokenType::SEMICOLON) {
            advance();
            return;
        }
        
        switch (currentToken.type) {
            case TokenType::VAR:
            case TokenType::FUNC:
            case TokenType::OP:
            case TokenType::GRAPH:
            case TokenType::IF:
            case TokenType::FOR:
            case TokenType::RETURN:
            case TokenType::IMPORT:
                return;
            default:
                break;
        }
        
        advance();
    }
}

// 解析整个程序
std::shared_ptr<Program> Parser::parse() {
    try {
        return parseProgram();
    } catch (const ParseError& error) {
        std::cerr << error.what() << std::endl;
        return nullptr;
    }
}

// 解析程序
std::shared_ptr<Program> Parser::parseProgram() {
    std::vector<std::shared_ptr<Statement>> statements;
    
    while (currentToken.type != TokenType::EOF_TOKEN) {
        try {
            statements.push_back(parseDeclaration());
        } catch (const ParseError& error) {
            std::cerr << error.what() << std::endl;
            synchronize();
        }
    }
    
    return std::make_shared<Program>(statements);
}

// 解析声明
std::shared_ptr<Statement> Parser::parseDeclaration() {
    if (match(TokenType::IMPORT)) {
        return parseImportStatement();
    }
    if (match(TokenType::FUNC)) {
        return parseFunctionDeclaration();
    }
    if (match(TokenType::OP)) {
        return parseOperationDeclaration();
    }
    if (match(TokenType::GRAPH)) {
        return parseGraphDeclaration();
    }
    if (match(TokenType::VAR)) {
        return parseVarDeclaration();
    }
    
    return parseStatement();
}

// 解析导入语句
std::shared_ptr<ImportStatement> Parser::parseImportStatement() {
    Token module = consume(TokenType::STRING_LITERAL, "Expected module name string after 'import'.");
    consume(TokenType::SEMICOLON, "Expected ';' after import statement.");
    return std::make_shared<ImportStatement>(module.lexeme);
}

// 解析函数声明
std::shared_ptr<FunctionDeclaration> Parser::parseFunctionDeclaration() {
    Token name = consume(TokenType::IDENTIFIER, "Expected function name.");
    
    consume(TokenType::LPAREN, "Expected '(' after function name.");
    std::vector<Parameter> parameters = parseParameters();
    consume(TokenType::RPAREN, "Expected ')' after parameters.");
    
    consume(TokenType::COLON, "Expected ':' after function parameters.");
    std::shared_ptr<Type> returnType = parseType();
    
    std::shared_ptr<Block> body = parseBlock();
    
    return std::make_shared<FunctionDeclaration>(name.lexeme, parameters, returnType, body);
}

// 解析操作声明
std::shared_ptr<OperationDeclaration> Parser::parseOperationDeclaration() {
    Token name = consume(TokenType::IDENTIFIER, "Expected operation name.");
    
    consume(TokenType::LPAREN, "Expected '(' after operation name.");
    std::vector<Parameter> parameters = parseParameters();
    consume(TokenType::RPAREN, "Expected ')' after parameters.");
    
    consume(TokenType::COLON, "Expected ':' after operation parameters.");
    std::shared_ptr<Type> returnType = parseType();
    
    std::shared_ptr<Block> body = parseBlock();
    
    return std::make_shared<OperationDeclaration>(name.lexeme, parameters, returnType, body);
}

// 解析计算图声明
std::shared_ptr<GraphDeclaration> Parser::parseGraphDeclaration() {
    Token name = consume(TokenType::IDENTIFIER, "Expected graph name.");
    
    consume(TokenType::LPAREN, "Expected '(' after graph name.");
    std::vector<Parameter> parameters = parseParameters();
    consume(TokenType::RPAREN, "Expected ')' after parameters.");
    
    consume(TokenType::COLON, "Expected ':' after graph parameters.");
    std::shared_ptr<Type> returnType = parseType();
    
    std::shared_ptr<Block> body = parseBlock();
    
    return std::make_shared<GraphDeclaration>(name.lexeme, parameters, returnType, body);
}

// 解析参数列表
std::vector<Parameter> Parser::parseParameters() {
    std::vector<Parameter> parameters;
    
    if (!check(TokenType::RPAREN)) {
        do {
            parameters.push_back(parseParameter());
        } while (match(TokenType::COMMA));
    }
    
    return parameters;
}

// 解析单个参数
Parameter Parser::parseParameter() {
    Token name = consume(TokenType::IDENTIFIER, "Expected parameter name.");
    consume(TokenType::COLON, "Expected ':' after parameter name.");
    std::shared_ptr<Type> type = parseType();
    
    return Parameter(name.lexeme, type);
}

// 解析语句
std::shared_ptr<Statement> Parser::parseStatement() {
    if (match(TokenType::IF)) {
        return parseIfStatement();
    }
    if (match(TokenType::FOR)) {
        return parseForStatement();
    }
    if (match(TokenType::RETURN)) {
        return parseReturnStatement();
    }
    if (match(TokenType::LBRACE)) {
        return parseBlock();
    }
    
    return parseExpressionStatement();
}

// 解析变量声明
std::shared_ptr<VarDeclaration> Parser::parseVarDeclaration() {
    Token name = consume(TokenType::IDENTIFIER, "Expected variable name.");
    
    consume(TokenType::COLON, "Expected ':' after variable name.");
    std::shared_ptr<Type> type = parseType();
    
    std::shared_ptr<Expression> initializer = nullptr;
    if (match(TokenType::ASSIGN)) {
        initializer = parseExpression();
    }
    
    consume(TokenType::SEMICOLON, "Expected ';' after variable declaration.");
    
    return std::make_shared<VarDeclaration>(name.lexeme, type, initializer);
}

// 解析 if 语句
std::shared_ptr<Statement> Parser::parseIfStatement() {
    consume(TokenType::LPAREN, "Expected '(' after 'if'.");
    std::shared_ptr<Expression> condition = parseExpression();
    consume(TokenType::RPAREN, "Expected ')' after if condition.");
    
    std::shared_ptr<Statement> thenBranch = parseStatement();
    std::shared_ptr<Statement> elseBranch = nullptr;
    
    if (match(TokenType::ELSE)) {
        elseBranch = parseStatement();
    }
    
    return std::make_shared<IfStatement>(condition, thenBranch, elseBranch);
}

// 解析 for 循环
std::shared_ptr<Statement> Parser::parseForStatement() {
    consume(TokenType::LPAREN, "Expected '(' after 'for'.");
    
    std::shared_ptr<Statement> initializer;
    if (match(TokenType::SEMICOLON)) {
        initializer = nullptr;
    } else if (match(TokenType::VAR)) {
        initializer = parseVarDeclaration();
    } else {
        initializer = parseExpressionStatement();
    }
    
    std::shared_ptr<Expression> condition = nullptr;
    if (!check(TokenType::SEMICOLON)) {
        condition = parseExpression();
    }
    consume(TokenType::SEMICOLON, "Expected ';' after loop condition.");
    
    std::shared_ptr<Expression> increment = nullptr;
    if (!check(TokenType::RPAREN)) {
        increment = parseExpression();
    }
    consume(TokenType::RPAREN, "Expected ')' after for clauses.");
    
    std::shared_ptr<Statement> body = parseStatement();
    
    return std::make_shared<ForStatement>(initializer, condition, increment, body);
}

// 解析 return 语句
std::shared_ptr<Statement> Parser::parseReturnStatement() {
    std::shared_ptr<Expression> value = nullptr;
    if (!check(TokenType::SEMICOLON)) {
        value = parseExpression();
    }
    
    consume(TokenType::SEMICOLON, "Expected ';' after return value.");
    
    return std::make_shared<ReturnStatement>(value);
}

// 解析代码块
std::shared_ptr<Block> Parser::parseBlock() {
    if (!check(TokenType::LBRACE)) {
        consume(TokenType::LBRACE, "Expected '{' before block.");
    } else {
        advance(); // 消费 '{'
    }
    
    std::vector<std::shared_ptr<Statement>> statements;
    
    while (!check(TokenType::RBRACE) && !check(TokenType::EOF_TOKEN)) {
        statements.push_back(parseDeclaration());
    }
    
    consume(TokenType::RBRACE, "Expected '}' after block.");
    
    return std::make_shared<Block>(statements);
}

// 解析表达式语句
std::shared_ptr<Statement> Parser::parseExpressionStatement() {
    std::shared_ptr<Expression> expr = parseExpression();
    consume(TokenType::SEMICOLON, "Expected ';' after expression.");
    return std::make_shared<ExpressionStmt>(expr);
}

// 解析表达式
std::shared_ptr<Expression> Parser::parseExpression() {
    return parseAssignment();
}

// 解析赋值表达式
std::shared_ptr<Expression> Parser::parseAssignment() {
    std::shared_ptr<Expression> expr = parseLogicalOr();
    
    if (match(TokenType::ASSIGN)) {
        std::shared_ptr<Expression> value = parseAssignment();
        
        if (auto var = std::dynamic_pointer_cast<Variable>(expr)) {
            return std::make_shared<BinaryExpr>(BinaryExpr::Op::ASSIGN, expr, value);
        }
        
        throw error("Invalid assignment target.");
    }
    
    return expr;
}

// 解析逻辑或表达式
std::shared_ptr<Expression> Parser::parseLogicalOr() {
    std::shared_ptr<Expression> expr = parseLogicalAnd();
    
    while (match(TokenType::OR)) {
        std::shared_ptr<Expression> right = parseLogicalAnd();
        expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::OR, expr, right);
    }
    
    return expr;
}

// 解析逻辑与表达式
std::shared_ptr<Expression> Parser::parseLogicalAnd() {
    std::shared_ptr<Expression> expr = parseEquality();
    
    while (match(TokenType::AND)) {
        std::shared_ptr<Expression> right = parseEquality();
        expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::AND, expr, right);
    }
    
    return expr;
}

// 解析相等性表达式
std::shared_ptr<Expression> Parser::parseEquality() {
    std::shared_ptr<Expression> expr = parseComparison();
    
    while (match(TokenType::EQ) || match(TokenType::NEQ)) {
        TokenType op = currentToken.type;
        advance();
        std::shared_ptr<Expression> right = parseComparison();
        
        if (op == TokenType::EQ) {
            expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::EQ, expr, right);
        } else {
            expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::NEQ, expr, right);
        }
    }
    
    return expr;
}

// 解析比较表达式
std::shared_ptr<Expression> Parser::parseComparison() {
    std::shared_ptr<Expression> expr = parseTerm();
    
    while (match(TokenType::LT) || match(TokenType::GT) || 
           match(TokenType::LE) || match(TokenType::GE)) {
        TokenType op = currentToken.type;
        advance();
        std::shared_ptr<Expression> right = parseTerm();
        
        switch (op) {
            case TokenType::LT:
                expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::LT, expr, right);
                break;
            case TokenType::GT:
                expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::GT, expr, right);
                break;
            case TokenType::LE:
                expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::LE, expr, right);
                break;
            case TokenType::GE:
                expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::GE, expr, right);
                break;
            default:
                break;
        }
    }
    
    return expr;
}

// 解析项表达式
std::shared_ptr<Expression> Parser::parseTerm() {
    std::shared_ptr<Expression> expr = parseFactor();
    
    while (match(TokenType::PLUS) || match(TokenType::MINUS)) {
        TokenType op = currentToken.type;
        advance();
        std::shared_ptr<Expression> right = parseFactor();
        
        if (op == TokenType::PLUS) {
            expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::ADD, expr, right);
        } else {
            expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::SUB, expr, right);
        }
    }
    
    return expr;
}

// 解析因子表达式
std::shared_ptr<Expression> Parser::parseFactor() {
    std::shared_ptr<Expression> expr = parseUnary();
    
    while (match(TokenType::STAR) || match(TokenType::SLASH)) {
        TokenType op = currentToken.type;
        advance();
        std::shared_ptr<Expression> right = parseUnary();
        
        if (op == TokenType::STAR) {
            expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::MUL, expr, right);
        } else {
            expr = std::make_shared<BinaryExpr>(BinaryExpr::Op::DIV, expr, right);
        }
    }
    
    return expr;
}

// 解析一元表达式
std::shared_ptr<Expression> Parser::parseUnary() {
    if (match(TokenType::NOT) || match(TokenType::MINUS)) {
        TokenType op = currentToken.type;
        advance();
        std::shared_ptr<Expression> right = parseUnary();
        
        if (op == TokenType::NOT) {
            return std::make_shared<UnaryExpr>(UnaryExpr::Op::NOT, right);
        } else {
            return std::make_shared<UnaryExpr>(UnaryExpr::Op::NEG, right);
        }
    }
    
    return parseCall();
}

// 解析函数调用表达式
std::shared_ptr<Expression> Parser::parseCall() {
    std::shared_ptr<Expression> expr = parsePrimary();
    
    while (true) {
        if (match(TokenType::LPAREN)) {
            expr = finishCall(expr);
        } else {
            break;
        }
    }
    
    return expr;
}

// 完成函数调用解析
std::shared_ptr<Expression> Parser::finishCall(std::shared_ptr<Expression> callee) {
    std::vector<std::shared_ptr<Expression>> arguments;
    
    if (!check(TokenType::RPAREN)) {
        do {
            arguments.push_back(parseExpression());
        } while (match(TokenType::COMMA));
    }
    
    consume(TokenType::RPAREN, "Expected ')' after arguments.");
    
    // 简化版：假设返回类型为 float
    std::shared_ptr<Type> returnType = std::make_shared<BasicType>(Type::TypeKind::FLOAT);
    
    // 获取函数名
    std::string calleeName;
    if (auto var = std::dynamic_pointer_cast<Variable>(callee)) {
        calleeName = var->name;
    } else {
        throw error("Expected function name.");
    }
    
    return std::make_shared<CallExpr>(calleeName, arguments, returnType);
}

// 解析基本表达式
std::shared_ptr<Expression> Parser::parsePrimary() {
    if (match(TokenType::INT_LITERAL)) {
        int value = std::stoi(currentToken.lexeme);
        return std::make_shared<IntLiteral>(value);
    }
    
    if (match(TokenType::FLOAT_LITERAL)) {
        float value = std::stof(currentToken.lexeme);
        return std::make_shared<FloatLiteral>(value);
    }
    
    if (match(TokenType::BOOL_LITERAL)) {
        bool value = (currentToken.lexeme == "true");
        return std::make_shared<BoolLiteral>(value);
    }
    
    if (match(TokenType::STRING_LITERAL)) {
        // 字符串字面量处理
        return nullptr; // 简化版：不处理字符串字面量
    }
    
    if (match(TokenType::IDENTIFIER)) {
        std::string name = currentToken.lexeme;
        // 简化版：假设变量类型为 float
        std::shared_ptr<Type> type = std::make_shared<BasicType>(Type::TypeKind::FLOAT);
        return std::make_shared<Variable>(name, type);
    }
    
    if (match(TokenType::LPAREN)) {
        std::shared_ptr<Expression> expr = parseExpression();
        consume(TokenType::RPAREN, "Expected ')' after expression.");
        return expr;
    }
    
    if (match(TokenType::LBRACKET)) {
        // 解析张量字面量
        std::vector<std::shared_ptr<Expression>> elements;
        
        if (!check(TokenType::RBRACKET)) {
            do {
                elements.push_back(parseExpression());
            } while (match(TokenType::COMMA));
        }
        
        consume(TokenType::RBRACKET, "Expected ']' after tensor elements.");
        
        // 简化版：假设张量元素类型为 float，形状为 [elements.size()]
        std::shared_ptr<Type> elementType = std::make_shared<BasicType>(Type::TypeKind::FLOAT);
        std::vector<int> shape = {static_cast<int>(elements.size())};
        std::shared_ptr<TensorType> tensorType = std::make_shared<TensorType>(elementType, shape);
        
        return std::make_shared<TensorLiteral>(elements, tensorType);
    }
    
    throw error("Expected expression.");
}

// 解析类型
std::shared_ptr<Type> Parser::parseType() {
    if (match(TokenType::TYPE_INT)) {
        return std::make_shared<BasicType>(Type::TypeKind::INT);
    }
    
    if (match(TokenType::TYPE_FLOAT)) {
        return std::make_shared<BasicType>(Type::TypeKind::FLOAT);
    }
    
    if (match(TokenType::TYPE_BOOL)) {
        return std::make_shared<BasicType>(Type::TypeKind::BOOL);
    }
    
    if (match(TokenType::TYPE_TENSOR)) {
        return parseTensorType();
    }
    
    throw error("Expected type.");
}

// 解析张量类型
std::shared_ptr<TensorType> Parser::parseTensorType() {
    consume(TokenType::LT, "Expected '<' after 'tensor'.");
    
    std::shared_ptr<Type> elementType = parseType();
    
    consume(TokenType::COMMA, "Expected ',' after element type.");
    
    consume(TokenType::LBRACKET, "Expected '[' for tensor shape.");
    
    std::vector<int> shape;
    if (!check(TokenType::RBRACKET)) {
        do {
            if (check(TokenType::INT_LITERAL)) {
                int dim = std::stoi(currentToken.lexeme);
                shape.push_back(dim);
                advance();
            } else if (check(TokenType::IDENTIFIER)) {
                // 符号维度，简化版：使用 -1 表示
                shape.push_back(-1);
                advance();
            } else {
                throw error("Expected dimension size or identifier.");
            }
        } while (match(TokenType::COMMA));
    }
    
    consume(TokenType::RBRACKET, "Expected ']' after tensor shape.");
    consume(TokenType::GT, "Expected '>' after tensor type.");
    
    return std::make_shared<TensorType>(elementType, shape);
}

} // namespace ai_compiler
