#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from process_wechat_data import WeChatDataProcessor

class MockLLMHandler(BaseHTTPRequestHandler):
    """模拟大模型API服务器"""
    
    def do_POST(self):
        if self.path == '/v1/chat/completions':
            # 读取请求数据
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # 模拟处理延迟
            time.sleep(random.uniform(0.1, 0.5))
            
            # 随机生成评分
            score = random.uniform(3.0, 9.5)
            
            # 返回模拟响应
            response = {
                "choices": [
                    {
                        "message": {
                            "content": f"{score:.1f}"
                        }
                    }
                ]
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        # 禁用日志输出
        pass

def start_mock_server():
    """启动模拟服务器"""
    server = HTTPServer(('localhost', 8888), MockLLMHandler)
    print("模拟LLM服务器启动在 http://localhost:8888")
    server.serve_forever()

def test_llm_evaluation():
    """测试大模型评估功能"""
    
    # 在后台启动模拟服务器
    server_thread = threading.Thread(target=start_mock_server, daemon=True)
    server_thread.start()
    time.sleep(1)  # 等待服务器启动
    
    # 创建测试配置
    test_config = {
        "data_processing": {
            "data_dir": "data",
            "output_file": "test_training_data.json",
            "my_wxid": "wxid_twmzyezhlsj022",
            "max_inter_message_gap": 90,
            "max_reply_delay": 300,
            "use_multiprocessing": False,  # 关闭多进程以便调试
            "max_workers": 1
        },
        "llm_evaluation": {
            "enabled": True,
            "api_url": "http://localhost:8888/v1/chat/completions",
            "api_key": "test_key",
            "model": "test-model",
            "max_workers": 2,
            "min_score": 6.0,
            "timeout": 10,
            "retry_attempts": 2,
            "evaluation_prompt": "评估对话质量: {instruction} | {input} | {output}"
        },
        "output": {
            "save_detailed_results": True,
            "save_evaluation_scores": True,
            "filtered_data_file": "test_training_data_filtered.json",
            "evaluation_report_file": "test_evaluation_report.json"
        }
    }
    
    # 保存测试配置
    with open('test_config.json', 'w', encoding='utf-8') as f:
        json.dump(test_config, f, ensure_ascii=False, indent=2)
    
    # 创建测试训练数据
    test_training_data = [
        {
            "instruction": "你是骆明宇，正在和朋友聊天。",
            "input": "历史记录:\n[14:30:00] 朋友: 你在干什么？\n\n对方最新消息:\n今天天气真好",
            "output": "是啊<return>出去走走吧"
        },
        {
            "instruction": "你是骆明宇，正在和朋友聊天。",
            "input": "历史记录:\n[15:00:00] 朋友: 饿了\n\n对方最新消息:\n去吃饭吗",
            "output": "走走走<return>吃什么"
        },
        {
            "instruction": "你是骆明宇，正在和朋友聊天。",
            "input": "历史记录:\n[16:00:00] 朋友: 作业好难\n\n对方最新消息:\n你会做吗",
            "output": "不会<return>一起研究研究"
        }
    ]
    
    # 使用测试配置创建处理器
    processor = WeChatDataProcessor('test_config.json')
    
    # 测试大模型评估功能
    print("开始测试大模型评估功能...")
    filtered_data, evaluation_results = processor.evaluate_training_data(test_training_data)
    
    print(f"\n评估结果:")
    print(f"原始数据: {evaluation_results['total_samples']} 条")
    print(f"通过筛选: {evaluation_results['passed_samples']} 条")
    print(f"平均分数: {evaluation_results['average_score']:.2f}")
    print(f"分数分布: {evaluation_results['score_distribution']}")
    
    # 保存测试结果
    with open('test_evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    print("\n测试完成！结果已保存到 test_evaluation_report.json")

if __name__ == "__main__":
    test_llm_evaluation() 