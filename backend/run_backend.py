from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import sys
import tempfile
import shutil


# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.agent_service import AgentService
from utils.web_print import get_process_messages, clear_process_messages

app = FastAPI(title="PaperMind API", version="1.0.0")

# 全局变量
is_test =  True # 默认为True，表示开发者调试模式，使用.env中的API-KEY
user_api_key = None  # 用户输入的API-KEY

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务 - 提供assets目录下的文件
assets_path = os.path.join(os.path.dirname(__file__), "..", "assets")
if os.path.exists(assets_path):
    app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

# 初始化Agent服务
agent_service = AgentService()

class ChatRequest(BaseModel):
    message: str
    mode: str = "existing"
    pdf_path: Optional[str] = None
    thread_id: str = "default_session"

class ChatResponse(BaseModel):
    response: str
    pdf_path: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "PaperMind API", "version": "1.0.0"}

@app.get("/modes")
async def get_modes():
    """获取可用的问答模式"""
    return {
        "modes": [
            {"key": "existing", "name": "💾 现有知识库问答"},
            {"key": "upload", "name": "📤 上传知识库问答"},
            {"key": "research", "name": "🔍 论文检索问答"}
        ]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理聊天消息"""
    # 检查API-KEY状态
    if not is_test and not user_api_key:
        return ChatResponse(
            response="请先在右侧API-KEY设置中输入您的API-KEY",
            pdf_path=request.pdf_path
        )
    
    try:
        response, new_pdf_path = await agent_service.get_response(
            user_input=request.message,
            mode=request.mode,
            pdf_path=request.pdf_path,
            thread_id=request.thread_id,
            api_key=user_api_key if not is_test else None
        )
        return ChatResponse(response=response, pdf_path=new_pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """上传PDF文件"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF文件")
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        return {
            "filename": file.filename,
            "path": tmp_path,
            "size": os.path.getsize(tmp_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")
    finally:
        file.file.close()

@app.get("/pdf/{filename}")
async def get_pdf(filename: str):
    """获取PDF文件（用于预览）"""
    # 这里需要实现安全的文件访问
    # 为了安全起见，只允许访问特定目录下的文件
    allowed_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "assets", "awesome_papers"),
        tempfile.gettempdir()
    ]
    
    # 安全检查：防止目录遍历攻击
    for allowed_dir in allowed_dirs:
        full_path = os.path.join(allowed_dir, filename)
        if os.path.exists(full_path) and os.path.isfile(full_path):
            return FileResponse(
                full_path, 
                media_type='application/pdf',
                headers={
                    'Content-Disposition': 'inline',
                    'X-Content-Type-Options': 'nosniff',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            )
    
    raise HTTPException(status_code=404, detail="PDF文件不存在")


class ResetRequest(BaseModel):
    mode: Optional[str] = None

@app.post("/reset-agent")
async def reset_agent(request: ResetRequest):
    """重置agent实例"""
    try:
        agent_service.reset_agent(request.mode)
        return {"success": True, "message": f"Agent重置成功{'（所有模式）' if request.mode is None else f'（模式：{request.mode}）'}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置失败: {str(e)}")


class SetApiKeyRequest(BaseModel):
    api_key: str

@app.post("/set-api-key")
async def set_api_key(request: SetApiKeyRequest):
    """设置用户API-KEY"""
    global user_api_key, is_test
    
    if not request.api_key:
        return {"success": False, "message": "API-KEY不能为空"}
    
    # 简单的验证：检查API-KEY格式（以'sk-'开头）
    if not request.api_key.startswith('sk-'):
        return {"success": False, "message": "API-KEY格式不正确，应以'sk-'开头"}
    
    user_api_key = request.api_key
    is_test = False  # 用户输入了API-KEY，切换到非测试模式
    
    return {"success": True, "message": "API-KEY设置成功"}

class ProcessMessagesRequest(BaseModel):
    session_id: str = "default_session"
    clear: bool = False

@app.post("/get-process-messages")
async def get_process_messages_api(request: ProcessMessagesRequest):
    """获取中间过程消息"""
    try:
        messages = get_process_messages(request.session_id, request.clear)
        return {"success": True, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取消息失败: {str(e)}")


@app.post("/clear-process-messages")
async def clear_process_messages_api(request: ProcessMessagesRequest):
    """清空中间过程消息"""
    try:
        clear_process_messages(request.session_id)
        return {"success": True, "message": f"已清空会话 {request.session_id} 的中间过程消息"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空消息失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
