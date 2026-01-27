import asyncio
import time
from typing import Optional, List
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import logging

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RequestItem:
    """单个请求项"""
    request_id: str
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    timestamp: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)


class InferenceRequest(BaseModel):
    """推理请求"""
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7


class InferenceResponse(BaseModel):
    """推理响应"""
    request_id: str
    prompt: str
    generated_text: str
    tokens: int
    latency_ms: float


class RequestQueue:
    """FIFO请求队列"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = deque(maxlen=max_size)
        self.lock = asyncio.Lock()
        self.event = asyncio.Event()
    
    async def put(self, item: RequestItem):
        """添加请求到队列"""
        async with self.lock:
            self.queue.append(item)
            self.event.set()
    
    async def get_batch(self, batch_size: int) -> List[RequestItem]:
        """获取一批请求（静态批处理）"""
        batch = []
        
        # 等待至少有一个请求
        await self.event.wait()
        
        async with self.lock:
            while len(batch) < batch_size and len(self.queue) > 0:
                batch.append(self.queue.popleft())
            
            if len(self.queue) == 0:
                self.event.clear()
        
        return batch
    
    async def size(self) -> int:
        """获取队列大小"""
        async with self.lock:
            return len(self.queue)


class DummyLLM:
    """模拟LLM模型（用于演示）"""
    
    def __init__(self, model_name: str = "dummy-llm"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Model loaded on {self.device}")
    
    def generate(self, prompts: List[str], max_tokens: int, 
                 temperature: float) -> List[str]:
        """批量生成文本"""
        outputs = []
        
        # 模拟推理延迟
        batch_size = len(prompts)
        inference_time = 0.1 * batch_size + np.random.uniform(0.05, 0.2)
        time.sleep(inference_time)
        
        for prompt in prompts:
            # 模拟生成的文本
            generated = f"Generated response for: {prompt[:30]}..."
            outputs.append(generated)
        
        return outputs


class BatchingScheduler:
    """静态批处理调度器"""
    
    def __init__(self, batch_size: int = 8, timeout_ms: int = 100):
        self.batch_size = batch_size
        self.timeout_s = timeout_ms / 1000.0
        self.request_queue = RequestQueue()
        self.model = DummyLLM()
        self.running = False
    
    async def process_batch(self):
        """处理一批请求的核心逻辑"""
        batch = []
        start_wait = time.time()
        
        # 等待直到批次满或超时
        while len(batch) < self.batch_size:
            remaining_time = self.timeout_s - (time.time() - start_wait)
            
            if remaining_time <= 0:
                # 超时，处理部分批次
                break
            
            try:
                # 尝试在超时内获取请求
                wait_task = asyncio.create_task(
                    self.request_queue.get_batch(self.batch_size - len(batch))
                )
                new_requests = await asyncio.wait_for(
                    wait_task, 
                    timeout=remaining_time
                )
                batch.extend(new_requests)
            except asyncio.TimeoutError:
                # 超时，使用当前批次
                break
        
        if not batch:
            return
        
        # 执行推理
        prompts = [req.prompt for req in batch]
        inference_start = time.time()
        
        try:
            generated_texts = self.model.generate(
                prompts,
                batch[0].max_tokens,  # 使用第一个请求的max_tokens
                batch[0].temperature
            )
            inference_time = (time.time() - inference_start) * 1000
            
            # 设置结果
            for req, text in zip(batch, generated_texts):
                latency = (time.time() - req.timestamp) * 1000
                result = InferenceResponse(
                    request_id=req.request_id,
                    prompt=req.prompt,
                    generated_text=text,
                    tokens=req.max_tokens,
                    latency_ms=latency
                )
                
                if not req.future.done():
                    req.future.set_result(result)
            
            logger.info(
                f"Batch processed: size={len(batch)}, "
                f"inference_time={inference_time:.2f}ms"
            )
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
    
    async def scheduler_loop(self):
        """调度器主循环"""
        self.running = True
        logger.info("Scheduler started")
        
        while self.running:
            try:
                await self.process_batch()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(0.01)
    
    def start(self):
        """启动调度器"""
        asyncio.create_task(self.scheduler_loop())
    
    def stop(self):
        """停止调度器"""
        self.running = False


# 全局调度器实例
scheduler = BatchingScheduler(batch_size=8, timeout_ms=100)

# FastAPI应用
app = FastAPI(title="dLLM Online Serving Framework")


@app.on_event("startup")
async def startup_event():
    """应用启动"""
    scheduler.start()
    logger.info("Application started")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭"""
    scheduler.stop()
    logger.info("Application shutdown")


@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """推理端点"""
    import uuid
    
    request_id = str(uuid.uuid4())[:8]
    
    # 创建请求项
    future = asyncio.Future()
    req = RequestItem(
        request_id=request_id,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        future=future
    )
    
    # 添加到队列
    try:
        await scheduler.request_queue.put(req)
        logger.info(f"Request {request_id} queued")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Queue full: {e}")
    
    # 等待结果（添加超时防止请求挂起）
    try:
        result = await asyncio.wait_for(future, timeout=30.0)
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Inference timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue_size")
async def get_queue_size():
    """获取队列大小"""
    size = await scheduler.request_queue.size()
    return {"queue_size": size}


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "scheduler_running": scheduler.running,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # 使用单个worker避免多进程问题
    )