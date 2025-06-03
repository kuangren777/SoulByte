#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time
import re
from typing import Dict, Tuple
from managers.config_manager import ConfigManager
from managers.evaluation_cache import EvaluationCache


class LLMEvaluator:
    """大模型评价器"""
    
    def __init__(self, config: ConfigManager, output_dir: str = "output"):
        self.config = config
        self.api_url = config.get('llm_evaluation.api_url')
        self.api_key = config.get('llm_evaluation.api_key')
        self.model = config.get('llm_evaluation.model')
        self.timeout = config.get('llm_evaluation.timeout', 30)
        self.retry_attempts = config.get('llm_evaluation.retry_attempts', 3)
        self.evaluation_prompt = config.get('llm_evaluation.evaluation_prompt')
        
        # 初始化缓存系统（使用output目录）
        self.cache = EvaluationCache(output_dir)
        
    def evaluate_sample(self, sample: Dict) -> Tuple[float, str]:
        """评估单个训练样本"""
        # 先检查缓存
        cached_result = self.cache.get_score(sample)
        if cached_result:
            return cached_result
        
        prompt = self.evaluation_prompt.format(
            instruction=sample['instruction'],
            input=sample['input'],
            output=sample['output']
        )
        
        for attempt in range(self.retry_attempts):
            try:
                response = self._call_api(prompt)
                score = self._parse_score(response)
                
                # 保存到缓存
                self.cache.save_score(sample, score, response)
                return score, response
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    print(f"评估失败: {e}")
                    return 0.0, f"错误: {e}"
                time.sleep(1)  # 重试前等待
        
        return 0.0, "评估失败"
    
    def _call_api(self, prompt: str) -> str:
        """调用大模型API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': self.model,
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'max_tokens': 10000,
            'temperature': 0.1,
            'chat_template_kwargs': {'enable_thinking': False},
            'top_k': 20,
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        return result['choices'][0]['message']['content'].strip()
    
    def _parse_score(self, response: str) -> float:
        """解析评分结果"""
        # 先尝试提取最后一行的数字（通常分数在最后）
        lines = response.strip().split('\n')
        for line in reversed(lines):
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', line)
            if numbers:
                score = float(numbers[-1])  # 取最后一个数字
                if 0 <= score <= 10:
                    return score
        
        # 如果上面没找到，尝试在整个响应中查找
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response)
        if numbers:
            for num_str in reversed(numbers):  # 从后往前找
                score = float(num_str)
                if 0 <= score <= 10:
                    return score
        
        print(f"无法解析评分响应: {response[:100]}...")
        return 0.0
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return self.cache.get_cache_size()
    
    def clear_cache(self) -> None:
        """清空评估缓存"""
        self.cache.clear_cache() 