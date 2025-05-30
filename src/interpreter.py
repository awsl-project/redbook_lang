import re
from io import StringIO
import math

# --- 1. 词法分析 (Lexer) ---


class Token:
    """代表一个词法单元 (Token)"""

    def __init__(self, type, value, line=0, column=0):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type}, {repr(self.value)}, L{self.line}C{self.column})"


class Lexer:
    """RedbookLang (薯言) 词法分析器"""

    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.token_specs = [
            ('SKIP',      r'[ \t]+'),
            ('NEWLINE',   r'\n'),
            ('COMMENT',   r'#.*'),
            ('NUMBER',    r'\d+(\.\d*)?'),
            ('STRING',    r"'[^']+'"),
            ('KW_ASSIGN', r'种草一个'),
            ('KW_IS',     r'是'),
            ('KW_PRINT',  r'听我说,'),
            ('KW_DEF',    r'开个新帖'),
            ('KW_CALL',   r'调用'),
            ('KW_NAME',   r'叫'),
            ('KW_PARAMS', r'艾特'),
            ('KW_WITH',   r'用'),
            ('KW_RETURN', r'绝绝子'),
            ('KW_IF',     r'康康是不是'),
            ('KW_ELIF',   r'或者康康'),
            ('KW_ELSE',   r'都不是呢'),
            ('KW_WHILE',  r'本宝宝坚持'),
            ('KW_FOR',    r'盘点清单'),
            ('KW_FOR_START_PARAM', r'启动:'),
            ('KW_FOR_END_PARAM',   r'目标:'),
            ('KW_FOR_STEP_PARAM',  r'跨步:'),
            ('LIT_TRUE',  r'当然啦'),
            ('LIT_FALSE', r'才不是'),
            ('OP_PLUS',   r'加上'),
            ('OP_MINUS',  r'减去'),
            ('OP_MUL',    r'乘以'),
            ('OP_DIV',    r'除以'),
            ('OPERATOR_SYMBOL_MINUS', r'-'),
            ('OP_EQ',     r'等于'),
            ('OP_NE',     r'不等于'),
            ('OP_LT',     r'小于'),
            ('OP_LE',     r'小于等于'),
            ('OP_GT',     r'大于'),
            ('OP_GE',     r'大于等于'),
            ('OP_AND',    r'并且'),
            ('OP_OR',     r'或者'),
            ('OP_NOT',    r'反转魅力'),
            ('LPAREN',    r'\('),
            ('RPAREN',    r'\)'),
            ('COLON',     r':'),
            ('COMMA',     r','),
            ('IDENTIFIER', r'"[^"]+"'),
            ('MISMATCH',  r'.'),
        ]
        self.token_regex = re.compile(
            '|'.join('(?P<%s>%s)' % pair for pair in self.token_specs))

    def tokenize(self):
        tokens = []
        while self.pos < len(self.text):
            current_char_col = self.column
            match = self.token_regex.match(self.text, self.pos)
            if not match:
                raise SyntaxError(
                    f"词法错误: 在行 {self.line} 列 {current_char_col} 遇到非法字符。")

            type = match.lastgroup
            value = match.group(type)
            token_line = self.line
            token_col_start = current_char_col

            if type == 'NEWLINE':
                tokens.append(
                    Token('NEWLINE', '\n', token_line, token_col_start))
                self.line += 1
                self.column = 1
            elif type == 'SKIP' or type == 'COMMENT':
                newline_count = value.count('\n')
                if newline_count > 0:
                    self.line += newline_count
                    self.column = len(value.split('\n')[-1]) + 1
                else:
                    self.column += len(value)
            elif type == 'MISMATCH':
                if value.strip() == '' and self.pos + len(value) == len(self.text):
                    pass
                else:
                    raise SyntaxError(
                        f"词法错误: 在行 {token_line} 列 {token_col_start} 遇到未知Token: '{value}'")
            else:
                original_token_text_len = len(value)
                if type == 'IDENTIFIER':
                    value = value[1:-1]
                elif type == 'STRING':
                    value = value[1:-1]

                tokens.append(Token(type, value, token_line, token_col_start))
                self.column += original_token_text_len
            self.pos = match.end()
        tokens.append(Token('EOF', None, self.line, self.column))
        return tokens

# --- 2. 语法分析 (Parser) & AST 定义 ---


class ASTNode:
    pass


class Program(ASTNode):
    def __init__(self, statements):
        self.statements = statements


class VarAssign(ASTNode):
    def __init__(self, name_token, value_node):
        self.name = Identifier(name_token)
        self.value = value_node


class Print(ASTNode):
    def __init__(self, values):
        self.values = values


class BinOp(ASTNode):
    def __init__(self, left, op_token, right):
        self.left = left
        self.op = op_token
        self.right = right


class UnaryOp(ASTNode):
    def __init__(self, op_token, operand):
        self.op = op_token
        self.operand = operand


class Num(ASTNode):
    def __init__(self, token):
        self.token = token
        self.value = float(
            token.value) if '.' in token.value else int(token.value)


class Identifier(ASTNode):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class StringLiteral(ASTNode):
    def __init__(self, token):
        self.token = token
        self.value = token.value


class BooleanLiteral(ASTNode):
    def __init__(self, token):
        self.token = token
        self.value = (token.type == 'LIT_TRUE')


class FuncDef(ASTNode):
    def __init__(self, name_token, params_tokens, body_node):
        self.name = Identifier(name_token)
        self.params = [Identifier(pt) for pt in params_tokens]
        self.body = body_node


class FuncCall(ASTNode):
    def __init__(self, name_node, args_nodes):
        self.name = name_node
        self.args = args_nodes


class Return(ASTNode):
    def __init__(self, value_node):
        self.value = value_node


class IfNode(ASTNode):
    def __init__(self, cases, else_statement):
        self.cases = cases
        self.else_statement = else_statement


class WhileNode(ASTNode):
    def __init__(self, condition_node, body_statement_node):
        self.condition = condition_node
        self.body_statement = body_statement_node


class ForNode(ASTNode):
    def __init__(self, var_name_token, start_expr_node, end_expr_node, step_expr_node, body_statement_node):
        self.var_name_token = var_name_token
        self.start_expr = start_expr_node
        self.end_expr = end_expr_node
        self.step_expr = step_expr_node
        self.body_statement = body_statement_node


class Parser:
    def __init__(self, tokens):
        self.tokens = [t for t in tokens if t.type not in (
            'SKIP', 'COMMENT', 'NEWLINE')]
        self.pos = 0
        self.current_token = self.tokens[self.pos] if self.tokens else Token(
            'EOF', None)

    def _error(self, message):
        token = self.current_token
        raise SyntaxError(
            f"{message} (在行 {token.line} 列 {token.column}, Token: {token})")

    def _advance(self):
        self.pos += 1
        self.current_token = self.tokens[self.pos] if self.pos < len(
            self.tokens) else Token('EOF', None)

    def _eat(self, token_type):
        if self.current_token.type == token_type:
            token = self.current_token
            self._advance()
            return token
        else:
            self._error(
                f"语法下头了！期望得到 {token_type}，但却是 {self.current_token.type}")

    def _parse_expression(self):
        return self._parse_logical_or()

    def _parse_logical_or(self):
        node = self._parse_logical_and()
        while self.current_token.type == 'OP_OR':
            op_token = self._eat('OP_OR')
            node = BinOp(left=node, op_token=op_token,
                         right=self._parse_logical_and())
        return node

    def _parse_logical_and(self):
        node = self._parse_comparison()
        while self.current_token.type == 'OP_AND':
            op_token = self._eat('OP_AND')
            node = BinOp(left=node, op_token=op_token,
                         right=self._parse_comparison())
        return node

    def _parse_comparison(self):
        node = self._parse_arithmetic_add_sub()
        comp_ops = ('OP_EQ', 'OP_NE', 'OP_LT', 'OP_LE', 'OP_GT', 'OP_GE')
        if self.current_token.type in comp_ops:
            op_token = self._eat(self.current_token.type)
            node = BinOp(left=node, op_token=op_token,
                         right=self._parse_arithmetic_add_sub())
        return node

    def _parse_arithmetic_add_sub(self):
        node = self._parse_arithmetic_mul_div()
        while self.current_token.type in ('OP_PLUS', 'OP_MINUS'):
            op_token = self._eat(self.current_token.type)
            node = BinOp(left=node, op_token=op_token,
                         right=self._parse_arithmetic_mul_div())
        return node

    def _parse_arithmetic_mul_div(self):
        node = self._parse_unary_operations()
        while self.current_token.type in ('OP_MUL', 'OP_DIV'):
            op_token = self._eat(self.current_token.type)
            node = BinOp(left=node, op_token=op_token,
                         right=self._parse_unary_operations())
        return node

    def _parse_unary_operations(self):
        token = self.current_token
        if token.type == 'OP_NOT':
            op_token = self._eat('OP_NOT')
            operand_node = self._parse_unary_operations()
            return UnaryOp(op_token=op_token, operand=operand_node)
        elif token.type == 'OPERATOR_SYMBOL_MINUS':
            op_token = self._eat('OPERATOR_SYMBOL_MINUS')
            operand_node = self._parse_unary_operations()
            return UnaryOp(op_token=op_token, operand=operand_node)
        else:
            return self._parse_primary()

    def _parse_primary(self):
        token = self.current_token
        if token.type == 'NUMBER':
            self._eat('NUMBER')
            return Num(token)
        elif token.type == 'STRING':
            self._eat('STRING')
            return StringLiteral(token)
        elif token.type == 'LIT_TRUE':
            self._eat('LIT_TRUE')
            return BooleanLiteral(token)
        elif token.type == 'LIT_FALSE':
            self._eat('LIT_FALSE')
            return BooleanLiteral(token)
        elif token.type == 'IDENTIFIER':
            self._eat('IDENTIFIER')
            return Identifier(token)
        elif token.type == 'KW_CALL':
            return self._parse_func_call_expression()
        elif token.type == 'LPAREN':
            self._eat('LPAREN')
            node = self._parse_expression()
            self._eat('RPAREN')
            return node
        else:
            self._error(
                f"表达式中遇到意想不到的宝藏Token: {token.type} (值: {repr(token.value)})")

    def _parse_statement(self):
        ttype = self.current_token.type
        if ttype == 'KW_ASSIGN':
            return self._parse_var_assign()
        elif ttype == 'KW_PRINT':
            return self._parse_print()
        elif ttype == 'KW_DEF':
            return self._parse_func_def()
        elif ttype == 'KW_RETURN':
            return self._parse_return()
        elif ttype == 'KW_IF':
            return self._parse_if_statement()
        elif ttype == 'KW_WHILE':
            return self._parse_while_statement()
        elif ttype == 'KW_FOR':
            return self._parse_for_statement()
        elif ttype == 'EOF':
            self._error("代码结尾还想有啥惊喜吗？可能少写了点东西哦~")
        return self._parse_expression()

    def _parse_var_assign(self):
        self._eat('KW_ASSIGN')
        name_token = self._eat('IDENTIFIER')
        self._eat('KW_IS')
        value_node = self._parse_expression()
        return VarAssign(name_token=name_token, value_node=value_node)

    def _parse_print(self):
        self._eat('KW_PRINT')
        values = []
        if self.current_token.type != 'EOF':
            values.append(self._parse_expression())
            while self.current_token.type == 'COMMA':
                self._eat('COMMA')
                values.append(self._parse_expression())
        return Print(values)

    def _parse_func_def(self):
        self._eat('KW_DEF')
        self._eat('KW_NAME')
        name_token = self._eat('IDENTIFIER')
        self._eat('KW_PARAMS')
        self._eat('LPAREN')
        params_tokens = []
        if self.current_token.type == 'IDENTIFIER':
            params_tokens.append(self._eat('IDENTIFIER'))
            while self.current_token.type == 'COMMA':
                self._eat('COMMA')
                params_tokens.append(self._eat('IDENTIFIER'))
        self._eat('RPAREN')
        self._eat('COLON')
        body_node = self._parse_statement()
        return FuncDef(name_token=name_token, params_tokens=params_tokens, body_node=body_node)

    def _parse_func_call_expression(self):
        self._eat('KW_CALL')
        name_token = self._eat('IDENTIFIER')
        name_node = Identifier(name_token)
        self._eat('KW_WITH')
        self._eat('LPAREN')
        args_nodes = []
        if self.current_token.type != 'RPAREN':
            args_nodes.append(self._parse_expression())
            while self.current_token.type == 'COMMA':
                self._eat('COMMA')
                args_nodes.append(self._parse_expression())
        self._eat('RPAREN')
        return FuncCall(name_node=name_node, args_nodes=args_nodes)

    def _parse_return(self):
        self._eat('KW_RETURN')
        value_node = self._parse_expression()
        return Return(value_node)

    def _parse_if_statement(self):
        cases = []
        self._eat('KW_IF')
        condition_node = self._parse_expression()
        self._eat('COLON')
        statement_node = self._parse_statement()
        cases.append((condition_node, statement_node))
        while self.current_token.type == 'KW_ELIF':
            self._eat('KW_ELIF')
            condition_node = self._parse_expression()
            self._eat('COLON')
            statement_node = self._parse_statement()
            cases.append((condition_node, statement_node))
        else_statement_node = None
        if self.current_token.type == 'KW_ELSE':
            self._eat('KW_ELSE')
            self._eat('COLON')
            else_statement_node = self._parse_statement()
        return IfNode(cases, else_statement_node)

    def _parse_while_statement(self):
        self._eat('KW_WHILE')
        condition_node = self._parse_expression()
        self._eat('COLON')
        body_statement_node = self._parse_statement()
        return WhileNode(condition_node, body_statement_node)

    def _parse_for_statement(self):
        self._eat('KW_FOR')
        var_name_token = self._eat('IDENTIFIER')
        self._eat('LPAREN')
        self._eat('KW_FOR_START_PARAM')
        start_expr_node = self._parse_expression()
        self._eat('COMMA')
        self._eat('KW_FOR_END_PARAM')
        end_expr_node = self._parse_expression()
        step_expr_node = None
        if self.current_token.type == 'COMMA':
            self._eat('COMMA')
            self._eat('KW_FOR_STEP_PARAM')
            step_expr_node = self._parse_expression()
        self._eat('RPAREN')
        self._eat('COLON')
        body_statement_node = self._parse_statement()
        return ForNode(var_name_token, start_expr_node, end_expr_node, step_expr_node, body_statement_node)

    def parse(self):
        statements = []
        while self.current_token.type != 'EOF':
            statements.append(self._parse_statement())
        return Program(statements)

# --- 3. 解释器 (Interpreter) ---


class Scope:
    def __init__(self, parent=None):
        self.parent = parent
        self.symbols = {}

    def get(self, name):
        result = self.symbols.get(name)
        return result if result is not None or not self.parent else self.parent.get(name)

    def set(self, name, value):
        self.symbols[name] = value


class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value


class Interpreter:
    MAX_LOOP_ITERATIONS = 2000

    def __init__(self, parser_or_code_string):
        if isinstance(parser_or_code_string, str):
            lexer = Lexer(parser_or_code_string)
            tokens = lexer.tokenize()
            parser = Parser(tokens)
        elif isinstance(parser_or_code_string, Parser):
            parser = parser_or_code_string
        else:
            raise TypeError("Interpreter需要Parser对象或代码字符串")
        self.parser_instance = parser
        self.global_scope = Scope()
        self.output_buffer = StringIO()

    def _type_error(self, message, token=None):
        context = ""
        if token and hasattr(token, 'line') and hasattr(token, 'column'):
            context = f" (大约在行 {token.line} 列 {token.column})"
        raise TypeError(message + context)

    def _visit(self, node, scope):
        method_name = f'_visit_{type(node).__name__}'
        visitor = getattr(self, method_name, self._generic_visit)
        return visitor(node, scope)

    def _generic_visit(self, node, scope):
        raise TypeError(f"OMG! 薯言不支持这种操作哦: _visit_{type(node).__name__}")

    def _to_boolean(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return len(value) > 0
        return value is not None

    def _visit_Program(self, node, scope):
        result = None
        for statement in node.statements:
            try:
                result = self._visit(statement, scope)
            except ReturnValue:
                raise SyntaxError("'绝绝子' (返回)只能在新帖 (函数) 内部使用哦!")
        return result

    def _visit_VarAssign(self, node, scope):
        value = self._visit(node.value, scope)
        scope.set(node.name.value, value)
        return None

    def _visit_Print(self, node, scope):
        values = [self._visit(v, scope) for v in node.values]
        self.output_buffer.write(" ".join(map(str, values)) + "\n")
        return None

    def _visit_BinOp(self, node, scope):
        left_val = self._visit(node.left, scope)

        if node.op.type == 'OP_AND':
            if not self._to_boolean(left_val):
                return False
            right_val = self._visit(node.right, scope)
            return self._to_boolean(right_val)
        elif node.op.type == 'OP_OR':
            if self._to_boolean(left_val):
                return True
            right_val = self._visit(node.right, scope)
            return self._to_boolean(right_val)

        right_val = self._visit(node.right, scope)
        op_type = node.op.type

        if op_type == 'OP_PLUS':
            if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                return left_val + right_val
            elif isinstance(left_val, str) or isinstance(right_val, str):
                return str(left_val) + str(right_val)
            else:
                self._type_error(
                    f"“加上”操作不支持这两种东东: {type(left_val).__name__} 和 {type(right_val).__name__}", node.op)
        elif op_type == 'OP_MINUS':
            if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                return left_val - right_val
            else:
                self._type_error(f"“减去”只认数字宝宝哦", node.op)
        elif op_type == 'OP_MUL':
            if isinstance(left_val, (int, float)) and isinstance(right_val, (int, float)):
                return left_val * right_val
            else:
                self._type_error(f"“乘以”只认数字宝宝哦", node.op)
        elif op_type == 'OP_DIV':
            if not isinstance(left_val, (int, float)) or not isinstance(right_val, (int, float)):
                self._type_error(f"“除以”的左右两边都必须是数字哦", node.op)
            if right_val == 0:
                raise ZeroDivisionError("除以零？姐妹这可不兴学啊！")
            return left_val / right_val
        elif op_type == 'OP_EQ':
            return left_val == right_val
        elif op_type == 'OP_NE':
            return left_val != right_val
        elif op_type == 'OP_LT':
            if not (type(left_val) == type(right_val) and isinstance(left_val, (int, float, str))):
                self._type_error(f"“小于”比较时两边类型要一样哦 (数字或字符串)", node.op)
            return left_val < right_val
        elif op_type == 'OP_LE':
            if not (type(left_val) == type(right_val) and isinstance(left_val, (int, float, str))):
                self._type_error(f"“小于等于”比较时两边类型要一样哦", node.op)
            return left_val <= right_val
        elif op_type == 'OP_GT':
            if not (type(left_val) == type(right_val) and isinstance(left_val, (int, float, str))):
                self._type_error(f"“大于”比较时两边类型要一样哦", node.op)
            return left_val > right_val
        elif op_type == 'OP_GE':
            if not (type(left_val) == type(right_val) and isinstance(left_val, (int, float, str))):
                self._type_error(f"“大于等于”比较时两边类型要一样哦", node.op)
            return left_val >= right_val
        else:
            self._generic_visit(node, scope)

    def _visit_UnaryOp(self, node, scope):
        operand_val = self._visit(node.operand, scope)
        if node.op.type == 'OP_NOT':
            return not self._to_boolean(operand_val)
        elif node.op.type == 'OPERATOR_SYMBOL_MINUS':
            if not isinstance(operand_val, (int, float)):
                self._type_error(
                    f"一元负号 '-' 只能用于数字哦，宝！但你给我的是 {type(operand_val).__name__}", node.op)
            return -operand_val
        else:
            self._generic_visit(node, scope)

    def _visit_Num(self, node, scope): return node.value

    def _visit_Identifier(self, node, scope):
        val = scope.get(node.value)
        if val is None:
            raise NameError(
                f"这个“{node.value}”还没在我这里种草呢，找不到呀宝！ (在行 {node.token.line} 列 {node.token.column})")
        return val

    def _visit_StringLiteral(self, node, scope): return node.value
    def _visit_BooleanLiteral(self, node, scope): return node.value

    def _visit_FuncDef(self, node, scope):
        scope.set(node.name.value, node)
        return None

    def _visit_FuncCall(self, node, scope):
        func_name = node.name.value
        func_def = scope.get(func_name)
        if not isinstance(func_def, FuncDef):
            raise TypeError(
                f"“{func_name}”好像不是一个新帖笔记（函数），不能调用哦！ (在行 {node.name.token.line} 列 {node.name.token.column})")
        if len(func_def.params) != len(node.args):
            raise TypeError(
                f"调用新帖“{func_name}”时艾特的人数不对！期望 {len(func_def.params)} 个，实际 {len(node.args)} 个 (在行 {node.name.token.line} 列 {node.name.token.column})")

        call_scope = Scope(parent=scope)
        for param_node, arg_node in zip(func_def.params, node.args):
            arg_value = self._visit(arg_node, scope)
            call_scope.set(param_node.value, arg_value)

        try:
            return self._visit(func_def.body, call_scope)
        except ReturnValue as ret:
            return ret.value
        return None

    def _visit_Return(self, node, scope):
        raise ReturnValue(self._visit(node.value, scope))

    def _visit_IfNode(self, node, scope):
        for condition_node, statement_node in node.cases:
            if self._to_boolean(self._visit(condition_node, scope)):
                return self._visit(statement_node, scope)
        if node.else_statement:
            return self._visit(node.else_statement, scope)
        return None

    def _visit_WhileNode(self, node, scope):
        iterations = 0
        while self._to_boolean(self._visit(node.condition, scope)):
            if iterations >= self.MAX_LOOP_ITERATIONS:
                raise RuntimeError(
                    f"本宝宝坚持不住啦！循环超过 {self.MAX_LOOP_ITERATIONS} 次，可能写了个无限循环哦！")
            self._visit(node.body_statement, scope)
            iterations += 1
        return None

    def _visit_ForNode(self, node, scope):
        start_val = self._visit(node.start_expr, scope)
        end_val = self._visit(node.end_expr, scope)
        step_val = 1
        if node.step_expr:
            step_val = self._visit(node.step_expr, scope)

        if not (isinstance(start_val, (int, float)) and
                isinstance(end_val, (int, float)) and
                isinstance(step_val, (int, float))):
            self._type_error("盘点清单的参数(启动、目标、跨步)都必须是数字哦！", node.var_name_token)
        if step_val == 0:
            raise ValueError("盘点清单的“跨步”可不能是零呀！")

        loop_var_name = node.var_name_token.value
        original_loop_var_value = scope.get(loop_var_name)
        had_original_value = loop_var_name in scope.symbols

        is_float_iter = any(isinstance(v, float)
                            for v in [start_val, end_val, step_val])

        current_val = float(start_val)
        target_val = float(end_val)
        actual_step = float(step_val)

        iterations = 0

        while True:
            if actual_step > 0 and current_val >= target_val:
                break
            if actual_step < 0 and current_val <= target_val:
                break

            if iterations >= self.MAX_LOOP_ITERATIONS:
                raise RuntimeError(
                    f"盘点清单也太长了叭！超过 {self.MAX_LOOP_ITERATIONS} 次，停一停！")

            loop_var_assign_val = current_val
            if not is_float_iter:
                if current_val == math.trunc(current_val):
                    loop_var_assign_val = int(current_val)

            scope.set(loop_var_name, loop_var_assign_val)
            self._visit(node.body_statement, scope)

            current_val += actual_step
            iterations += 1

        if had_original_value:
            scope.set(loop_var_name, original_loop_var_value)
        elif loop_var_name in scope.symbols:
            del scope.symbols[loop_var_name]

        return None

    def interpret(self, code_string_to_run=None):
        self.output_buffer = StringIO()

        current_parser = self.parser_instance
        if code_string_to_run is not None:
            lexer = Lexer(code_string_to_run)
            tokens = lexer.tokenize()
            current_parser = Parser(tokens)

        try:
            tree = current_parser.parse()
            self._visit(tree, self.global_scope)
            return {"output": self.output_buffer.getvalue(), "error": None}
        except ReturnValue as e:
            return {
                "output": self.output_buffer.getvalue(),
                "error": f"程序运行下头了！\n原因: '绝绝子' (返回)语句可能在非函数内部的顶层使用了: {e.value}"
            }
        except (SyntaxError, TypeError, NameError, ZeroDivisionError, ValueError, RuntimeError) as e:
            return {"output": self.output_buffer.getvalue(),
                    "error": f"程序运行下头了！\n原因({type(e).__name__}): {e}"}
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            return {
                "output": self.output_buffer.getvalue(),
                "error": f"解释器遇到一个未知的内部意外！\n原因({type(e).__name__}): {e}\n{tb_str}"
            }
