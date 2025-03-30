# 前端教程：词法分析、语法分析和AST构建

本教程将介绍AI编译器前端的实现，包括词法分析器、语法分析器和抽象语法树(AST)的构建。

## 1. 词法分析器 (Lexer)

词法分析器是编译器的第一个阶段，负责将源代码文本转换为词法单元(Token)序列。

### 1.1 词法单元的定义

在我们的AI编译器中，词法单元由以下几个部分组成：

```cpp
struct Token {
    TokenType type;     // 词法单元类型
    std::string lexeme; // 词法单元的文本表示
    int line;           // 行号
};
```

词法单元类型包括关键字、标识符、字面量、运算符和分隔符等：

```cpp
enum class TokenType {
    // 关键字
    VAR, FUNC, OP, GRAPH, IF, ELSE, FOR, RETURN, IMPORT,
    
    // 数据类型
    TYPE_INT, TYPE_FLOAT, TYPE_BOOL, TYPE_TENSOR,
    
    // 标识符和字面量
    IDENTIFIER, INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL, BOOL_LITERAL,
    
    // 运算符
    PLUS, MINUS, STAR, SLASH, ASSIGN, EQ, NEQ, LT, GT, LE, GE, AND, OR, NOT,
    
    // 分隔符
    LPAREN, RPAREN, LBRACE, RBRACE, LBRACKET, RBRACKET, COMMA, SEMICOLON, COLON, DOT,
    
    // 特殊标记
    EOF_TOKEN, ERROR
};
```

### 1.2 词法分析器的实现

词法分析器的核心是`scanToken`方法，它负责识别下一个词法单元：

```cpp
void Lexer::scanToken() {
    char c = advance();
    
    switch (c) {
        // 处理单字符词法单元
        case '(': addToken(TokenType::LPAREN); break;
        case ')': addToken(TokenType::RPAREN); break;
        // ... 其他单字符词法单元
        
        // 处理可能是单字符或双字符的词法单元
        case '!':
            addToken(match('=') ? TokenType::NEQ : TokenType::NOT);
            break;
        // ... 其他可能是单字符或双字符的词法单元
        
        // 处理注释
        case '/':
            if (match('/')) {
                // 单行注释
                while (peek() != '\n' && !isAtEnd()) advance();
            } else if (match('*')) {
                // 多行注释
                // ... 处理多行注释的代码
            } else {
                addToken(TokenType::SLASH);
            }
            break;
        
        // 处理空白字符
        case ' ':
        case '\r':
        case '\t':
            break;
        
        // 处理换行
        case '\n':
            line++;
            break;
        
        // 处理字符串字面量
        case '"':
            handleString();
            break;
        
        // 处理其他字符
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
```

词法分析器还包括一些辅助方法，如`handleString`、`handleNumber`和`handleIdentifier`，用于处理特定类型的词法单元。

### 1.3 使用词法分析器

使用词法分析器非常简单：

```cpp
Lexer lexer("var x: int = 5;");
std::vector<Token> tokens = lexer.scanTokens();

for (const auto& token : tokens) {
    std::cout << token.toString() << std::endl;
}
```

输出结果将是：

```
VAR 'var' at line 1
IDENTIFIER 'x' at line 1
COLON ':' at line 1
TYPE_INT 'int' at line 1
ASSIGN '=' at line 1
INT_LITERAL '5' at line 1
SEMICOLON ';' at line 1
EOF '' at line 1
```

## 2. 语法分析器 (Parser)

语法分析器是编译器的第二个阶段，负责将词法单元序列转换为抽象语法树(AST)。

### 2.1 递归下降解析

我们的语法分析器使用递归下降解析技术，为每个语法规则定义一个解析方法：

```cpp
std::shared_ptr<Program> Parser::parse() {
    try {
        return parseProgram();
    } catch (const ParseError& error) {
        std::cerr << error.what() << std::endl;
        return nullptr;
    }
}

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
```

每个解析方法负责解析特定的语法结构，并返回相应的AST节点。例如，`parseDeclaration`方法解析声明语句：

```cpp
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
```

### 2.2 错误处理

语法分析器包含错误处理机制，当遇到语法错误时，它会尝试恢复并继续解析：

```cpp
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
```

### 2.3 使用语法分析器

使用语法分析器解析源代码：

```cpp
Parser parser("var x: int = 5;");
std::shared_ptr<Program> program = parser.parse();

if (program) {
    std::cout << program->toString() << std::endl;
}
```

输出结果将是AST的文本表示：

```
var x: int = 5;
```

## 3. 抽象语法树 (AST)

抽象语法树是源代码的树形表示，每个节点代表源代码中的一个结构。

### 3.1 AST节点类型

我们的AST包含以下主要节点类型：

- **表达式**：代表计算值的代码片段
  - 字面量：整数、浮点数、布尔值、张量
  - 变量引用
  - 二元表达式：加、减、乘、除等
  - 一元表达式：负号、逻辑非等
  - 函数调用

- **语句**：代表执行操作的代码片段
  - 表达式语句
  - 变量声明
  - 块语句
  - If语句
  - For循环
  - 返回语句
  - 函数声明
  - 操作声明
  - 计算图声明
  - 导入语句

### 3.2 AST节点的实现

每个AST节点都继承自`ASTNode`基类：

```cpp
class ASTNode {
public:
    virtual ~ASTNode() = default;
    virtual std::string toString() const = 0;
};
```

例如，二元表达式节点的实现：

```cpp
class BinaryExpr : public Expression {
public:
    enum class Op {
        ADD, SUB, MUL, DIV, EQ, NEQ, LT, GT, LE, GE, AND, OR
    };
    
    Op op;
    std::shared_ptr<Expression> left;
    std::shared_ptr<Expression> right;
    
    BinaryExpr(Op op, std::shared_ptr<Expression> left, std::shared_ptr<Expression> right)
        : op(op), left(left), right(right) {}
    
    std::string toString() const override {
        // ... 实现toString方法
    }
    
    std::shared_ptr<Type> getType() const override {
        // ... 实现类型推断
    }
};
```

### 3.3 类型系统

AST包含一个类型系统，用于表示变量、表达式和函数的类型：

```cpp
class Type : public ASTNode {
public:
    enum class TypeKind {
        INT, FLOAT, BOOL, TENSOR, FUNCTION
    };

    TypeKind kind;

    Type(TypeKind kind) : kind(kind) {}
    virtual ~Type() = default;
};

class TensorType : public Type {
public:
    std::shared_ptr<Type> elementType;
    std::vector<int> shape;
    
    TensorType(std::shared_ptr<Type> elementType, const std::vector<int>& shape)
        : Type(TypeKind::TENSOR), elementType(elementType), shape(shape) {}
    
    std::string toString() const override {
        // ... 实现toString方法
    }
};
```

## 4. 实践：解析一个简单的DSL程序

让我们看一个完整的例子，解析以下DSL程序：

```
func add(a: int, b: int): int {
    return a + b;
}

func main() {
    var x: int = 5;
    var y: int = 10;
    var z: int = add(x, y);
    print(z);
}
```

1. 词法分析器将源代码转换为词法单元序列
2. 语法分析器将词法单元序列转换为AST
3. AST可以用于后续的编译阶段，如IR生成和优化

## 5. 练习

1. 修改词法分析器，支持新的关键字和运算符
2. 扩展语法分析器，支持新的语法结构
3. 实现一个简单的类型检查器，验证AST中的类型一致性

## 6. 总结

前端是编译器的入口，负责将源代码转换为编译器内部的表示形式。通过词法分析、语法分析和AST构建，我们可以将DSL程序转换为结构化的形式，为后续的IR生成和优化奠定基础。

在下一个教程中，我们将介绍如何将AST转换为中间表示(IR)，这是编译器优化的基础。
