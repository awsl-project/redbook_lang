import re
from io import StringIO

# --- 1. 词法分析 (Lexer) ---


class Token:
    """代表一个词法单元 (Token)"""

    def __init__(self, type, value, line=0, column=0):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)})"


class Lexer:
    """Herlang 词法分析器"""

    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        # 定义 Token 规则 (正则表达式)
        self.token_specs = [
            ('SKIP',      r'[ \t]+'),
            ('NEWLINE',   r'\n'),
            ('COMMENT',   r'#.*'),
            ('NUMBER',    r'\d+(\.\d*)?'),
            ('STRING',    r"'[^']+'"),      # ADDED: 字符串字面量用单引号包裹
            ('KW_ASSIGN', r'种草一个'),
            ('KW_IS',     r'是'),
            ('KW_PRINT',  r'听我说,'),
            ('KW_DEF',    r'开个新帖'),
            ('KW_CALL',   r'调用'),
            ('KW_NAME',   r'叫'),
            ('KW_PARAMS', r'艾特'),
            ('KW_WITH',   r'用'),
            ('KW_RETURN', r'绝绝子'),
            ('OP_PLUS',   r'加上'),
            ('OP_MINUS',  r'减去'),
            ('OP_MUL',    r'乘以'),
            ('OP_DIV',    r'除以'),
            ('LPAREN',    r'\('),
            ('RPAREN',    r'\)'),
            ('COLON',     r':'),
            ('COMMA',     r','),
            ('IDENTIFIER', r'"[^"]+"'),  # MODIFIED COMMENT: 标识符用双引号包裹
            ('MISMATCH',  r'.'),
        ]
        self.token_regex = re.compile(
            '|'.join('(?P<%s>%s)' % pair for pair in self.token_specs))

    def tokenize(self):
        """生成 Token 列表"""
        tokens = []
        while self.pos < len(self.text):
            match = self.token_regex.match(self.text, self.pos)
            if not match:
                raise SyntaxError(f"在行 {self.line} 非法字符")

            type = match.lastgroup
            value = match.group(type)

            # Store column at the start of the token for more accurate reporting if needed later.
            # The current implementation uses self.column which is usually the start of the line.
            # Basic assignment, lexer's self.column isn't fully updated per token.
            token_col = self.column

            if type == 'NEWLINE':
                tokens.append(Token('NEWLINE', '\n', self.line, token_col))
                self.line += 1
                self.column = 1
            elif type == 'SKIP' or type == 'COMMENT':
                # For SKIP and COMMENT, we advance position but don't generate tokens.
                # Column would advance here if we were tracking it precisely for errors within skipped parts.
                pass
            elif type == 'MISMATCH':
                raise SyntaxError(
                    # Use token_col for slightly better reporting
                    f"在行 {self.line}, 列 {token_col} 遇到未知 Token: {value}")
            else:
                if type == 'IDENTIFIER':
                    value = value[1:-1]  # 去掉双引号
                elif type == 'STRING':   # ADDED
                    value = value[1:-1]  # 去掉单引号
                tokens.append(Token(type, value, self.line, token_col))
                # self.column += len(match.group(type)) # If precise column tracking per token is desired

            self.pos = match.end()

        tokens.append(Token('EOF', None))
        return tokens

# --- 2. 语法分析 (Parser) & AST 定义 ---


class ASTNode:
    pass


class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements


class VarAssign(ASTNode):
    def __init__(self, name, value):
        self.name = name
        self.value = value


class Print(ASTNode):
    def __init__(self, values):
        self.values = values


class BinOp(ASTNode):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right


class Num(ASTNode):
    def __init__(self, token):
        self.token = token
        self.value = float(
            token.value) if '.' in token.value else int(token.value)


class Identifier(ASTNode):
    def __init__(self, token):
        self.token = token
        self.value = token.value  # Value is already stripped of quotes by Lexer


class StringLiteral(ASTNode):  # ADDED
    """代表一个字符串字面量"""

    def __init__(self, token):
        self.token = token
        self.value = token.value  # Value is already stripped of quotes by Lexer

    def __repr__(self):
        return f"StringLiteral({repr(self.value)})"


class FuncDef(ASTNode):
    def __init__(self, name, params, body):
        self.name = name
        self.params = params
        self.body = body


class FuncCall(ASTNode):
    def __init__(self, name, args):
        self.name = name
        self.args = args


class Return(ASTNode):
    def __init__(self, value):
        self.value = value


class Parser:
    """Herlang 语法分析器"""

    def __init__(self, tokens):
        self.tokens = [t for t in tokens if t.type != 'NEWLINE']  # 简化处理，忽略换行符
        self.pos = 0
        if self.tokens:  # Ensure tokens list is not empty
            self.current_token = self.tokens[self.pos]
        # Handle empty token list (e.g. empty input or input with only comments/whitespace)
        else:
            self.current_token = Token('EOF', None)

    def _advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = Token('EOF', None)

    def _eat(self, token_type):
        if self.current_token.type == token_type:
            self._advance()
        else:
            raise SyntaxError(
                f"语法下头了！期望得到 {token_type}"
                f"(在行 {self.current_token.line} 列 {self.current_token.column}),"
                f"但却是 {self.current_token.type} (值: {repr(self.current_token.value)})"
            )

    def _parse_factor(self):
        token = self.current_token
        if token.type == 'NUMBER':
            self._eat('NUMBER')
            return Num(token)
        elif token.type == 'STRING':    # ADDED
            self._eat('STRING')
            return StringLiteral(token)
        elif token.type == 'IDENTIFIER':
            self._eat('IDENTIFIER')
            return Identifier(token)
        elif token.type == 'KW_CALL':
            return self._parse_func_call()
        elif token.type == 'LPAREN':
            self._eat('LPAREN')
            node = self._parse_expression()
            self._eat('RPAREN')
            return node
        else:
            raise SyntaxError(
                f"语法下头了！在因子中遇到非预期Token: {token.type} (值: {repr(token.value)}) 在行 {token.line} 列 {token.column}")

    def _parse_term(self):
        node = self._parse_factor()
        while self.current_token.type in ('OP_MUL', 'OP_DIV'):
            op_token = self.current_token
            self._eat(op_token.type)
            node = BinOp(left=node, op=op_token, right=self._parse_factor())
        return node

    def _parse_expression(self):
        node = self._parse_term()
        while self.current_token.type in ('OP_PLUS', 'OP_MINUS'):
            op_token = self.current_token
            self._eat(op_token.type)
            node = BinOp(left=node, op=op_token, right=self._parse_term())
        return node

    def _parse_var_assign(self):
        self._eat('KW_ASSIGN')
        name_token = self.current_token
        self._eat('IDENTIFIER')
        self._eat('KW_IS')
        value = self._parse_expression()
        return VarAssign(name=Identifier(name_token), value=value)

    def _parse_print(self):
        self._eat('KW_PRINT')
        values = []
        if self.current_token.type == 'EOF':  # Handle print with no arguments if desired, or raise error
            pass  # Or raise SyntaxError("听我说后面需要内容！")
        else:
            values.append(self._parse_expression())
            while self.current_token.type == 'COMMA':
                self._eat('COMMA')
                values.append(self._parse_expression())
        return Print(values)

    def _parse_func_def(self):
        self._eat('KW_DEF')
        self._eat('KW_NAME')
        name_token = self.current_token
        self._eat('IDENTIFIER')
        self._eat('KW_PARAMS')
        self._eat('LPAREN')
        params = []
        if self.current_token.type == 'IDENTIFIER':
            params.append(Identifier(self.current_token))
            self._eat('IDENTIFIER')
            while self.current_token.type == 'COMMA':
                self._eat('COMMA')
                params.append(Identifier(self.current_token))
                self._eat('IDENTIFIER')
        self._eat('RPAREN')
        self._eat('COLON')

        # 简化版：函数体就是下一个语句
        body = self._parse_statement()
        return FuncDef(name=Identifier(name_token), params=params, body=body)

    def _parse_func_call(self):
        self._eat('KW_CALL')
        name_token = self.current_token
        self._eat('IDENTIFIER')
        self._eat('KW_WITH')
        self._eat('LPAREN')
        args = []
        if self.current_token.type != 'RPAREN':
            args.append(self._parse_expression())
            while self.current_token.type == 'COMMA':
                self._eat('COMMA')
                args.append(self._parse_expression())
        self._eat('RPAREN')
        return FuncCall(name=Identifier(name_token), args=args)

    def _parse_return(self):
        self._eat('KW_RETURN')
        value = self._parse_expression()
        return Return(value)

    def _parse_statement(self):
        if self.current_token.type == 'KW_ASSIGN':
            return self._parse_var_assign()
        if self.current_token.type == 'KW_PRINT':
            return self._parse_print()
        if self.current_token.type == 'KW_DEF':
            return self._parse_func_def()
        if self.current_token.type == 'KW_RETURN':
            return self._parse_return()
        # Allow independent expressions only if they are function calls or have side effects.
        # For simplicity, current code allows any expression.
        # If current_token is EOF, it means end of input, should not parse as expression.
        if self.current_token.type == 'EOF':
            raise SyntaxError("非预期的文件末尾，可能缺少语句。")
        return self._parse_expression()

    def parse(self):
        statements = []
        while self.current_token.type != 'EOF':
            statements.append(self._parse_statement())
        return Program(statements)

# --- 3. 解释器 (Interpreter) ---


class Scope:
    """作用域，用于存储变量和函数"""

    def __init__(self, parent=None):
        self.parent = parent
        self.symbols = {}

    def get(self, name):
        result = self.symbols.get(name)
        if result is None and self.parent:
            return self.parent.get(name)
        return result

    def set(self, name, value):
        self.symbols[name] = value


class Interpreter:
    """遍历 AST 并执行的解释器"""

    def __init__(self, parser):
        self.tree = parser.parse()
        self.global_scope = Scope()

    def _visit(self, node, scope):
        method_name = f'_visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self._generic_visit)
        return visitor(node, scope)

    def _generic_visit(self, node, scope):
        raise TypeError(f"OMG, 没有对应的节点访问方法 _visit_{type(node).__name__}")

    def _visit_Program(self, node, scope):
        result = None
        for statement in node.statements:
            result = self._visit(statement, scope)
        return result  # Result of the last statement in the program

    def _visit_VarAssign(self, node, scope):
        value = self._visit(node.value, scope)
        scope.set(node.name.value, value)
        # Variable assignment itself doesn't produce a value to be returned further up

    def _visit_Print(self, node, scope):
        values = [self._visit(v, scope) for v in node.values]
        self.output_buffer.write(" ".join(map(str, values)) + "\n")
        # Print statement itself doesn't produce a value

    def _visit_BinOp(self, node, scope):
        left = self._visit(node.left, scope)
        right = self._visit(node.right, scope)

        # Type checking for operations
        if isinstance(left, str) or isinstance(right, str):
            if node.op.type == 'OP_PLUS':  # Allow string concatenation with '加上'
                return str(left) + str(right)
            else:
                raise TypeError(f"下头了！操作 {node.op.value} 不支持字符串类型。")

        if node.op.type == 'OP_PLUS':
            return left + right
        if node.op.type == 'OP_MINUS':
            return left - right
        if node.op.type == 'OP_MUL':
            return left * right
        if node.op.type == 'OP_DIV':
            # Check type before division
            if not isinstance(right, (int, float)) or right == 0:
                raise ZeroDivisionError("下头了！不能除以零或非数值类型！")
            if not isinstance(left, (int, float)):
                raise TypeError("下头了！被除数必须是数值类型！")
            return left / right

    def _visit_Num(self, node, scope):
        return node.value

    def _visit_Identifier(self, node, scope):  # MODIFIED: Strict lookup
        val = scope.get(node.value)
        if val is None:
            raise NameError(f"这个“{node.value}”还没种草呢，找不到！")
        return val

    def _visit_StringLiteral(self, node, scope):  # ADDED
        return node.value

    def _visit_FuncDef(self, node, scope):
        scope.set(node.name.value, node)  # Store FuncDef object itself

    def _visit_Return(self, node, scope):
        raise ReturnValue(self._visit(node.value, scope))

    def _visit_FuncCall(self, node, scope):
        func_def = scope.get(node.name.value)
        if not isinstance(func_def, FuncDef):
            raise TypeError(f"“{node.name.value}”不是一个新帖（函数），不能调用！")

        if len(func_def.params) != len(node.args):
            raise TypeError(
                f"调用函数“{node.name.value}”时艾特的人数不对！期望 {len(func_def.params)} 个，收到了 {len(node.args)} 个")

        call_scope = Scope(parent=scope)
        for param_node, arg_node in zip(func_def.params, node.args):
            # Evaluate arg in call_scope's parent (current scope)
            arg_value = self._visit(arg_node, call_scope)
            call_scope.set(param_node.value, arg_value)

        try:
            # Execute function body in the new call_scope
            return self._visit(func_def.body, call_scope)
        except ReturnValue as ret:
            return ret.value
        # Implicit return None if no '绝绝子' or if body doesn't return through ReturnValue
        return None

    def interpret(self):
        self.output_buffer = StringIO()
        try:
            self._visit(self.tree, self.global_scope)
            return self.output_buffer.getvalue()
        except (SyntaxError, TypeError, NameError, ZeroDivisionError) as e:
            # Get line number if available from the exception or its cause
            line_info = ""
            # Simplistic error line reporting attempt, real AST nodes would need line info
            # For now, rely on parser/lexer error messages for line numbers
            return f"程序运行下头了！\n原因: {e}{line_info}"
        except ReturnValue:
            return "程序运行下头了！\n原因: '绝绝子' (返回)语句只能在新帖 (函数) 中使用。"


class ReturnValue(Exception):
    """自定义异常，用于从函数调用中返回"""

    def __init__(self, value):
        self.value = value
