#ifndef LEXER_H
#define LEXER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace ai_compiler {

// 词法单元类型枚举
enum class TokenType {
    // 关键字
    VAR,
    FUNC,
    OP,
    GRAPH,
    IF,
    ELSE,
    FOR,
    RETURN,
    IMPORT,
    
    // 数据类型
    TYPE_INT,
    TYPE_FLOAT,
    TYPE_BOOL,
    TYPE_TENSOR,
    
    // 标识符和字面量
    IDENTIFIER,
    INT_LITERAL,
    FLOAT_LITERAL,
    STRING_LITERAL,
    BOOL_LITERAL,
    
    // 运算符
    PLUS,        // +
    MINUS,       // -
    STAR,        // *
    SLASH,       // /
    ASSIGN,      // =
    EQ,          // ==
    NEQ,         // !=
    LT,          // <
    GT,          // >
    LE,          // <=
    GE,          // >=
    AND,         // &&
    OR,          // ||
    NOT,         // !
    
    // 分隔符
    LPAREN,      // (
    RPAREN,      // )
    LBRACE,      // {
    RBRACE,      // }
    LBRACKET,    // [
    RBRACKET,    // ]
    COMMA,       // ,
    SEMICOLON,   // ;
    COLON,       // :
    DOT,         // .
    
    // 特殊标记
    EOF_TOKEN,
    ERROR
};

// 词法单元结构
struct Token {
    TokenType type;
    std::string lexeme;
    int line;
    
    Token(TokenType type, const std::string& lexeme, int line)
        : type(type), lexeme(lexeme), line(line) {}
        
    std::string toString() const;
};

// 词法分析器类
class Lexer {
public:
    Lexer(const std::string& source);
    
    // 获取下一个词法单元
    Token scanToken();
    
    // 扫描所有词法单元
    std::vector<Token> scanTokens();
    
private:
    std::string source;
    std::vector<Token> tokens;
    
    int start = 0;      // 当前词法单元的起始位置
    int current = 0;    // 当前扫描位置
    int line = 1;       // 当前行号
    
    // 关键字映射表
    static std::unordered_map<std::string, TokenType> keywords;
    
    // 辅助方法
    bool isAtEnd() const;
    char advance();
    bool match(char expected);
    char peek() const;
    char peekNext() const;
    
    // 处理各种词法单元的方法
    void scanToken();
    void addToken(TokenType type);
    void addToken(TokenType type, const std::string& lexeme);
    
    // 处理特定类型的词法单元
    void handleString();
    void handleNumber();
    void handleIdentifier();
    
    // 辅助判断字符类型的方法
    static bool isDigit(char c);
    static bool isAlpha(char c);
    static bool isAlphaNumeric(char c);
};

} // namespace ai_compiler

#endif // LEXER_H
