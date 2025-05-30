import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any

from src.interpreter import Lexer, Parser, Interpreter

app = FastAPI(
    title="RedbookLang Playground API",
    description="API for executing RedbookLang code.",
    version="1.0.0"
)


class CodePayload(BaseModel):
    code: str


def execute_redbooklang_code(code_string: str) -> Dict[str, Any]:
    """
    执行 RedbookLang 代码。
    使用修改后的 interpreter.interpret() 方法，该方法应返回一个字典。
    """
    try:
        lexer = Lexer(code_string)
        tokens = lexer.tokenize()
        parser = Parser(tokens)  # Parser.__init__ 应该处理空 token 列表的情况

        # Interpreter.__init__ 调用 parser.parse() 来生成 AST
        interpreter = Interpreter(parser)

        # interpret() 方法现在应该返回 {"output": "...", "error": "..."}
        output = interpreter.interpret()
        return output

    # 捕获在 Lexer 或 Parser 初始化/调用期间可能发生的错误
    # 这些错误可能在 interpreter.interpret() 被调用之前发生
    except SyntaxError as e:
        return {"output": "", "error": f"语法分析阶段错误 (SyntaxError):\n{e}"}
    except Exception as e:
        # 捕获在解释器设置或执行之外发生的任何其他意外错误
        return {"output": "", "error": f"代码准备阶段发生未知错误:\n{type(e).__name__}: {e}"}


@app.get("/", include_in_schema=False)
async def get_index_page():
    """提供前端 HTML 页面"""
    current_dir = os.path.dirname(os.path.realpath(__file__))
    index_html_path = os.path.join(current_dir, "templates", "index.html")
    if not os.path.exists(index_html_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_html_path)


@app.post("/run", summary="Run RedbookLang Code")
async def run_code_endpoint(payload: CodePayload) -> Dict[str, Any]:
    """
    接收 RedbookLang 代码，执行它，并返回输出或错误。
    """
    code = payload.code

    # code 为空字符串是允许的，解释器应该能处理
    # Pydantic 模型 CodePayload 已经确保 'code' 字段存在且为字符串

    execution_result = execute_redbooklang_code(code)

    # FastAPI 会自动将字典转换为 JSON 响应
    # 如果执行中发生错误，错误信息已包含在 execution_result["error"] 中
    # 如果需要，可以根据 execution_result["error"] 的存在情况设置 HTTP 状态码
    if execution_result.get("error"):
        # 可以选择返回不同的状态码，例如 400，如果错误是用户代码相关的
        # 但为了简单起见，这里总是返回 200，错误信息在 JSON 体内
        pass  # return JSONResponse(content=execution_result, status_code=400)

    return execution_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
