
# --- Main Execution ---
from src.interpreter import Interpreter, Lexer, Parser


if __name__ == "__main__":
    herlang_code_example = """
# 这是我的第一个 Herlang 程序, yyds!

种草一个 "单价" 是 12.5
种草一个 "数量" 是 10
种草一个 "欢迎消息" 是 'Hello Herlang, yyds!' # 使用单引号定义字符串

# 定义一个计算总价的函数
开个新帖 叫 "计算折扣价" 艾特 ("原价", "折扣率"):
    绝绝子 "原价" 乘以 "折扣率"

# 调用函数并赋值
种草一个 "总价" 是 调用 "计算折扣价" 用 ("单价" 乘以 "数量", 0.8)

听我说, '最终价格是:', "总价"   # 单引号字符串 和 双引号标识符
听我说, "欢迎消息"             # 打印字符串变量
听我说, '你好', '世界', "单价" 加上 2.5 # 多个字符串和表达式

# 字符串拼接
种草一个 "称呼" 是 '我的朋友'
听我说, "欢迎消息" 加上 ', ' 加上 "称呼" 加上 '!'

# 试试错误处理
种草一个 "空篮子" 是 0
# 下面这行会触发 ZeroDivisionError, 如果取消注释
# 听我说, 100 除以 "空篮子"

# 试试未定义变量
# 听我说, "未定义变量" # 这会触发 NameError
"""

    print("---------- [Herlang 源码] ----------")
    print(herlang_code_example)

    lexer = Lexer(herlang_code_example)
    tokens = lexer.tokenize()

    print("\n---------- [词法分析结果 (Tokens)] ----------")
    # print(tokens) # Optionally print all tokens for debugging

    parser = Parser(tokens)
    # try:
    #     ast = parser.parse()
    #     print("\n---------- [AST (部分表示)] ----------")
    #     for stmt in ast.statements:
    #         print(stmt) # Basic AST print
    # except SyntaxError as e:
    #     print(f"语法分析错误: {e}")
    #     sys.exit(1)

    interpreter = Interpreter(parser)  # Parser is passed, it will call parse()

    print("\n---------- [程序输出] ----------")
    output = interpreter.interpret()
    print(output)
