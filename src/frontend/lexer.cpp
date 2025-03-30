#include "lexer.h"
#include <cctype>
#include <iostream>

namespace ai_compiler {

// 初始化关键字映射表
std::unordered_map<std::string, TokenType> Lexer::keywords = {
    {"var", TokenType::VAR},
    {"func", TokenType::FUNC},
    {"op", TokenType::OP},
    {"graph", TokenType::GRAPH},
    {"if", TokenType::IF},
    {"else", TokenType::ELSE},
    {"for", TokenType::FOR},
    {"return", TokenType::RETURN},
    {"import", TokenType::IMPORT},
    {"int", TokenType::TYPE_INT},
    {"float", TokenType::TYPE_FLOAT},
    {"bool", TokenType::TYPE_BOOL},
    {"tensor", TokenType::TYPE_TENSOR},
    {"true", TokenType::BOOL_LITERAL},
    {"false", TokenType::BOOL_LITERAL}
};

// Token 类的 toString 方法实现
std::string Token::toString() const {
    std::string typeStr;
    
    // 将 TokenType 转换为字符串
    switch (type) {
        case TokenType::VAR: typeStr = "VAR"; break;
        case TokenType::FUNC: typeStr = "FUNC"; break;
        case TokenType::OP: typeStr = "OP"; break;
        case TokenType::GRAPH: typeStr = "GRAPH"; break;
        case TokenType::IF: typeStr = "IF"; break;
        case TokenType::ELSE: typeStr = "ELSE"; break;
        case TokenType::FOR: typeStr = "FOR"; break;
        case TokenType::RETURN: typeStr = "RETURN"; break;
        case TokenType::IMPORT: typeStr = "IMPORT"; break;
        case TokenType::TYPE_INT: typeStr = "TYPE_INT"; break;
        case TokenType::TYPE_FLOAT: typeStr = "TYPE_FLOAT"; break;
        case TokenType::TYPE_BOOL: typeStr = "TYPE_BOOL"; break;
        case TokenType::TYPE_TENSOR: typeStr = "TYPE_TENSOR"; break;
        case TokenType::IDENTIFIER: typeStr = "IDENTIFIER"; break;
        case TokenType::INT_LITERAL: typeStr = "INT_LITERAL"; break;
        case TokenType::FLOAT_LITERAL: typeStr = "FLOAT_LITERAL"; break;
        case TokenType::STRING_LITERAL: typeStr = "STRING_LITERAL"; break;
        case TokenType::BOOL_LITERAL: typeStr = "BOOL_LITERAL"; break;
        case TokenType::PLUS: typeStr = "PLUS"; break;
        case TokenType::MINUS: typeStr = "MINUS"; break;
        case TokenType::STAR: typeStr = "STAR"; break;
        case TokenType::SLASH: typeStr = "SLASH"; break;
        case TokenType::ASSIGN: typeStr = "ASSIGN"; break;
        case TokenType::EQ: typeStr = "EQ"; break;
        case TokenType::NEQ: typeStr = "NEQ"; break;
        case TokenType::LT: typeStr = "LT"; break;
        case TokenType::GT: typeStr = "GT"; break;
        case TokenType::LE: typeStr = "LE"; break;
        case TokenType::GE: typeStr = "GE"; break;
        case TokenType::AND: typeStr = "AND"; break;
        case TokenType::OR: typeStr = "OR"; break;
        case TokenType::NOT: typeStr = "NOT"; break;
        case TokenType::LPAREN: typeStr = "LPAREN"; break;
        case TokenType::RPAREN: typeStr = "RPAREN"; break;
        case TokenType::LBRACE: typeStr = "LBRACE"; break;
        case TokenType::RBRACE: typeStr = "RBRACE"; break;
        case TokenType::LBRACKET: typeStr = "LBRACKET"; break;
        case TokenType::RBRACKET: typeStr = "RBRACKET"; break;
        case TokenType::COMMA: typeStr = "COMMA"; break;
        case TokenType::SEMICOLON: typeStr = "SEMICOLON"; break;
        case TokenType::COLON: typeStr = "COLON"; break;
        case TokenType::DOT: typeStr = "DOT"; break;
        case TokenType::EOF_TOKEN: typeStr = "EOF"; break;
        case TokenType::ERROR: typeStr = "ERROR"; break;
        default: typeStr = "UNKNOWN"; break;
    }
    
    return typeStr + " '" + lexeme + "' at line " + std::to_string(line);
}

// Lexer 构造函数
Lexer::Lexer(const std::string& source) : source(source) {}

// 判断是否到达源代码末尾
bool Lexer::isAtEnd() const {
    return current >= source.length();
}

// 获取当前字符并前进
char Lexer::advance() {
    return source[current++];
}

// 条件前进
bool Lexer::match(char expected) {
    if (isAtEnd()) return false;
    if (source[current] != expected) return false;
    
    current++;
    return true;
}

// 查看当前字符但不前进
char Lexer::peek() const {
    if (isAtEnd()) return '\0';
    return source[current];
}

// 查看下一个字符但不前进
char Lexer::peekNext() const {
    if (current + 1 >= source.length()) return '\0';
    return source[current + 1];
}

// 添加词法单元
void Lexer::addToken(TokenType type) {
    std::string text = source.substr(start, current - start);
    tokens.push_back(Token(type, text, line));
}

// 添加带有特定词素的词法单元
void Lexer::addToken(TokenType type, const std::string& lexeme) {
    tokens.push_back(Token(type, lexeme, line));
}

// 处理字符串字面量
void Lexer::handleString() {
    while (peek() != '"' && !isAtEnd()) {
        if (peek() == '\n') line++;
        advance();
    }
    
    if (isAtEnd()) {
        // 未闭合的字符串
        addToken(TokenType::ERROR, "Unterminated string.");
        return;
    }
    
    // 消费闭合的引号
    advance();
    
    // 提取字符串值（不包括引号）
    std::string value = source.substr(start + 1, current - start - 2);
    addToken(TokenType::STRING_LITERAL, value);
}

// 处理数字字面量
void Lexer::handleNumber() {
    while (isDigit(peek())) advance();
    
    // 查找小数部分
    if (peek() == '.' && isDigit(peekNext())) {
        // 消费小数点
        advance();
        
        while (isDigit(peek())) advance();
        
        addToken(TokenType::FLOAT_LITERAL);
    } else {
        addToken(TokenType::INT_LITERAL);
    }
}

// 处理标识符
void Lexer::handleIdentifier() {
    while (isAlphaNumeric(peek())) advance();
    
    // 检查是否是关键字
    std::string text = source.substr(start, current - start);
    
    auto it = keywords.find(text);
    if (it != keywords.end()) {
        addToken(it->second);
    } else {
        addToken(TokenType::IDENTIFIER);
    }
}

// 扫描单个词法单元
void Lexer::scanToken() {
    char c = advance();
    
    switch (c) {
        // 单字符词法单元
        case '(': addToken(TokenType::LPAREN); break;
        case ')': addToken(TokenType::RPAREN); break;
        case '{': addToken(TokenType::LBRACE); break;
        case '}': addToken(TokenType::RBRACE); break;
        case '[': addToken(TokenType::LBRACKET); break;
        case ']': addToken(TokenType::RBRACKET); break;
        case ',': addToken(TokenType::COMMA); break;
        case '.': addToken(TokenType::DOT); break;
        case '-': addToken(TokenType::MINUS); break;
        case '+': addToken(TokenType::PLUS); break;
        case ';': addToken(TokenType::SEMICOLON); break;
        case '*': addToken(TokenType::STAR); break;
        case ':': addToken(TokenType::COLON); break;
        
        // 可能是单字符或双字符的词法单元
        case '!':
            addToken(match('=') ? TokenType::NEQ : TokenType::NOT);
            break;
        case '=':
            addToken(match('=') ? TokenType::EQ : TokenType::ASSIGN);
            break;
        case '<':
            addToken(match('=') ? TokenType::LE : TokenType::LT);
            break;
        case '>':
            addToken(match('=') ? TokenType::GE : TokenType::GT);
            break;
        case '&':
            if (match('&')) {
                addToken(TokenType::AND);
            } else {
                addToken(TokenType::ERROR, "Expected '&' after '&'.");
            }
            break;
        case '|':
            if (match('|')) {
                addToken(TokenType::OR);
            } else {
                addToken(TokenType::ERROR, "Expected '|' after '|'.");
            }
            break;
        case '/':
            if (match('/')) {
                // 单行注释，一直读到行尾
                while (peek() != '\n' && !isAtEnd()) advance();
            } else if (match('*')) {
                // 多行注释，一直读到 */
                while (!isAtEnd() && !(peek() == '*' && peekNext() == '/')) {
                    if (peek() == '\n') line++;
                    advance();
                }
                
                if (isAtEnd()) {
                    addToken(TokenType::ERROR, "Unterminated comment.");
                    return;
                }
                
                // 消费闭合的 */
                advance(); // *
                advance(); // /
            } else {
                addToken(TokenType::SLASH);
            }
            break;
            
        // 忽略空白字符
        case ' ':
        case '\r':
        case '\t':
            break;
            
        // 换行
        case '\n':
            line++;
            break;
            
        // 字符串字面量
        case '"':
            handleString();
            break;
            
        default:
            if (isDigit(c)) {
                handleNumber();
            } else if (isAlpha(c)) {
                handleIdentifier();
            } else {
                addToken(TokenType::ERROR, "Unexpected character.");
            }
            break;
    }
}

// 扫描所有词法单元
std::vector<Token> Lexer::scanTokens() {
    while (!isAtEnd()) {
        start = current;
        scanToken();
    }
    
    tokens.push_back(Token(TokenType::EOF_TOKEN, "", line));
    return tokens;
}

// 获取下一个词法单元
Token Lexer::scanToken() {
    if (tokens.empty()) {
        scanTokens();
    }
    
    if (tokens.empty()) {
        return Token(TokenType::EOF_TOKEN, "", 0);
    }
    
    Token token = tokens[0];
    tokens.erase(tokens.begin());
    return token;
}

// 辅助判断字符类型的方法
bool Lexer::isDigit(char c) {
    return c >= '0' && c <= '9';
}

bool Lexer::isAlpha(char c) {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           c == '_';
}

bool Lexer::isAlphaNumeric(char c) {
    return isAlpha(c) || isDigit(c);
}

} // namespace ai_compiler
