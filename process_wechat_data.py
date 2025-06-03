#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
import os
import glob
import requests
import time
import re
import hashlib
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count, Value, Lock
from tqdm import tqdm
import jieba

# 多进程工作函数（必须在全局作用域，避免序列化问题）

def extract_message_blocks_process_worker(args):
    """多进程提取对话回合的工作函数"""
    messages, start_idx, end_idx, process_id, MAX_INTER_MESSAGE_GAP, MAX_REPLY_DELAY, progress_dict = args
    conversation_rounds = []
    i = start_idx
    
    # 进程局部的联系人更新缓存
    local_contact_updates = {}
    total_to_process = end_idx - start_idx
    processed = 0
    
    def find_other_message_block(messages, start_idx):
        """寻找对方的消息块"""
        if start_idx >= len(messages):
            return None
        
        msg = messages[start_idx]
        if msg['is_sender'] == 1:
            return None
        
        block = [{'index': start_idx, **msg}]
        current_talker = msg['talker']
        last_time = msg['create_time']
        
        for i in range(start_idx + 1, len(messages)):
            next_msg = messages[i]
            
            if next_msg['is_sender'] == 1:
                break
            
            if next_msg['talker'] != current_talker:
                break
            
            time_gap = (next_msg['create_time'] - last_time).total_seconds()
            if time_gap > MAX_INTER_MESSAGE_GAP:
                break
            
            block.append({'index': i, **next_msg})
            last_time = next_msg['create_time']
        
        return block if len(block) > 0 else None
    
    def find_my_reply_block(messages, start_idx, last_other_msg):
        """寻找我的回复块"""
        if start_idx >= len(messages):
            return None
        
        first_my_msg = None
        first_my_idx = start_idx
        
        for i in range(start_idx, min(start_idx + 10, len(messages))):
            if messages[i]['is_sender'] == 1:
                reply_delay = (messages[i]['create_time'] - last_other_msg['create_time']).total_seconds()
                if reply_delay <= MAX_REPLY_DELAY:
                    first_my_msg = messages[i]
                    first_my_idx = i
                    break
        
        if not first_my_msg:
            return None
        
        block = [{'index': first_my_idx, **first_my_msg}]
        last_time = first_my_msg['create_time']
        
        for i in range(first_my_idx + 1, len(messages)):
            next_msg = messages[i]
            
            if next_msg['is_sender'] != 1:
                break
            
            time_gap = (next_msg['create_time'] - last_time).total_seconds()
            if time_gap > MAX_INTER_MESSAGE_GAP:
                break
            
            block.append({'index': i, **next_msg})
            last_time = next_msg['create_time']
        
        return block if len(block) > 0 else None
    
    def is_valid_reply_timing(last_other_msg, first_my_msg):
        """检查回复时间是否合理"""
        reply_delay = (first_my_msg['create_time'] - last_other_msg['create_time']).total_seconds()
        return 0 <= reply_delay <= MAX_REPLY_DELAY
    
    while i < min(end_idx, len(messages)):
        other_block = find_other_message_block(messages, i)
        if not other_block:
            i += 1
            processed += 1
            if processed % 100 == 0:  # 每处理100条消息更新一次进度
                progress_dict[process_id] = processed
            continue
        
        i = other_block[-1]['index'] + 1
        processed = i - start_idx  # 更准确的进度计算
        
        my_block = find_my_reply_block(messages, i, other_block[-1])
        if not my_block:
            continue
        
        if is_valid_reply_timing(other_block[-1], my_block[0]):
            conversation_rounds.append((other_block, my_block))
            
            # 缓存联系人信息更新
            for msg in other_block:
                talker = msg['talker']
                if talker not in local_contact_updates:
                    local_contact_updates[talker] = {
                        'message_count': 0,
                        'last_contact_date': msg['create_time']
                    }
                local_contact_updates[talker]['message_count'] += 1
                if msg['create_time'] > local_contact_updates[talker]['last_contact_date']:
                    local_contact_updates[talker]['last_contact_date'] = msg['create_time']
        
        i = my_block[-1]['index'] + 1
        processed = i - start_idx
        if processed % 100 == 0:  # 每处理100条消息更新一次进度
            progress_dict[process_id] = processed
    
    # 最后更新一次进度为总数
    progress_dict[process_id] = processed
    
    return conversation_rounds, local_contact_updates

def format_training_data_process_worker(args):
    """多进程格式化训练数据的工作函数"""
    conversation_rounds_chunk, all_messages, contact_data, my_name, process_id, progress_dict = args
    training_data = []
    total_to_process = len(conversation_rounds_chunk)
    processed = 0
    
    def format_message_block_content(message_block):
        """格式化消息块内容，使用<return>分割连续消息"""
        contents = [msg['content'] for msg in message_block]
        return "<return>".join(contents)
    
    def build_context_for_process(messages, reply_time):
        """构建历史上下文（进程版本）"""
        reply_date = reply_time.date()
        context_messages = []
        
        for msg in messages:
            msg_date = msg['create_time'].date()
            if msg_date == reply_date and msg['create_time'] < reply_time:
                context_messages.append(msg)
        
        return context_messages
    
    def get_other_name_for_process(msg, contact_data):
        """从预传递的数据中获取对方的昵称"""
        talker = msg['talker']
        
        contact_info = contact_data.get(talker, {})
        if contact_info.get('remark') and contact_info['remark'].strip():
            return contact_info['remark']
        elif contact_info.get('nickname') and contact_info['nickname'].strip():
            return contact_info['nickname']
        
        return talker[-8:] if len(talker) > 8 else talker
    
    def build_history_text_for_process(context_messages, current_other_block, contact_data):
        """构建历史对话文本"""
        history_lines = []
        
        for msg in context_messages:
            time_str = msg['create_time'].strftime("%H:%M:%S")
            sender = "我" if msg['is_sender'] == 1 else get_other_name_for_process(msg, contact_data)
            history_lines.append(f"[{time_str}] {sender}: {msg['content']}")
        
        for msg in current_other_block:
            time_str = msg['create_time'].strftime("%H:%M:%S")
            sender = get_other_name_for_process(msg, contact_data)
            history_lines.append(f"[{time_str}] {sender}: {msg['content']}")
        
        return "\n".join(history_lines[-20:])
    
    for idx, (other_block, my_block) in enumerate(conversation_rounds_chunk):
        try:
            reply_time = my_block[0]['create_time']
            context_messages = build_context_for_process(all_messages, reply_time)
            
            other_content = format_message_block_content(other_block)
            my_reply = format_message_block_content(my_block)
            history_text = build_history_text_for_process(context_messages, other_block, contact_data)
            
            # 获取对方的详细联系人信息
            talker_id = other_block[0]['talker']
            contact_info = contact_data.get(talker_id, {})
            other_name = get_other_name_for_process(other_block[0], contact_data)
            
            # 构建详细的联系人信息
            relationship = contact_info.get('relationship', '朋友')
            relationship_detail = contact_info.get('relationship_detail', '')
            first_contact_date = contact_info.get('first_contact_date', '')
            
            # 格式化首次联系时间
            first_contact_str = ""
            if first_contact_date:
                try:
                    if isinstance(first_contact_date, str):
                        from datetime import datetime
                        first_contact_dt = datetime.fromisoformat(first_contact_date.replace('Z', '+00:00'))
                        first_contact_str = first_contact_dt.strftime("%Y年%m月%d日")
                    else:
                        first_contact_str = first_contact_date.strftime("%Y年%m月%d日")
                except:
                    first_contact_str = ""
            
            # 格式化当前回复时间
            current_time_str = reply_time.strftime("%Y年%m月%d日 %H:%M")
            
            # 构建包含联系人信息的instruction
            instruction_parts = [
                f"你是{my_name}，正在和{other_name}聊天。"
            ]
            
            # 添加关系信息
            if relationship and relationship != '朋友':
                instruction_parts.append(f"对方是你的{relationship}。")
            elif relationship == '朋友':
                instruction_parts.append(f"对方是你的朋友。")
            
            # 添加关系详细描述
            if relationship_detail and relationship_detail.strip():
                instruction_parts.append(f"{relationship_detail}。")
            
            # 添加首次联系时间
            if first_contact_str:
                instruction_parts.append(f"你们从{first_contact_str}开始联系。")
            
            # 添加当前时间
            instruction_parts.append(f"现在是{current_time_str}。")
            
            # 添加回复指引
            instruction_parts.append("根据聊天记录和对方的最新消息，用你的风格回复。")
            
            instruction = " ".join(instruction_parts)
            
            training_sample = {
                "instruction": instruction,
                "input": f"历史记录:\n{history_text}\n\n对方最新消息:\n{other_content}",
                "output": my_reply
            }
            
            training_data.append(training_sample)
            
        except Exception as e:
            pass
        
        processed = idx + 1
        if processed % 10 == 0 or processed == total_to_process:  # 每10条或处理完所有数据时更新进度
            progress_dict[process_id] = processed
    
    return training_data

class ConfigManager:
    """配置管理器"""
    def __init__(self, config_file: str = "config.json"):
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
    
    def get(self, key_path: str, default=None):
        """获取配置值，支持点分隔的路径"""
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

class EvaluationCache:
    """评估结果缓存管理器"""
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.cache_file = os.path.join(output_dir, "evaluation_cache.json")
        self.cache = self._load_cache()
        # 添加线程锁保护缓存访问
        self._lock = threading.Lock()
    
    def _load_cache(self) -> Dict:
        """加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载评估缓存失败: {e}")
        return {}
    
    def _generate_key(self, sample: Dict) -> str:
        """生成样本的唯一键"""
        content = f"{sample['instruction']}|{sample['input']}|{sample['output']}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_score(self, sample: Dict) -> Optional[Tuple[float, str]]:
        """获取缓存的评分"""
        key = self._generate_key(sample)
        with self._lock:
            if key in self.cache:
                cached_data = self.cache[key]
                return cached_data['score'], cached_data['response']
        return None
    
    def save_score(self, sample: Dict, score: float, response: str) -> None:
        """保存评分到缓存"""
        key = self._generate_key(sample)
        cache_entry = {
            'score': score,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'sample': sample
        }
        
        with self._lock:
            self.cache[key] = cache_entry
            # 立即保存到文件
            self._save_cache_unsafe()
    
    def _save_cache_unsafe(self) -> None:
        """保存缓存到文件（不加锁，内部使用）"""
        try:
            # 创建缓存的副本以避免在序列化过程中被修改
            cache_copy = self.cache.copy()
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_copy, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存评估缓存失败: {e}")
    
    def _save_cache(self) -> None:
        """保存缓存到文件（公共接口）"""
        with self._lock:
            self._save_cache_unsafe()
    
    def filter_by_score(self, min_score: float) -> List[Dict]:
        """根据最小分数筛选样本"""
        filtered_samples = []
        stats = {
            'total': len(self.cache),
            'passed': 0,
            'failed': 0,
            'score_distribution': {}
        }
        
        # 分数分布统计初始化
        for score_range in [(0,2), (2,4), (4,6), (6,8), (8,10)]:
            stats['score_distribution'][f"{score_range[0]}-{score_range[1]}"] = 0
        
        for key, entry in self.cache.items():
            score = entry.get('score', 0)
            sample = entry.get('sample')
            
            # 统计分数分布
            for min_r, max_r in [(0,2), (2,4), (4,6), (6,8), (8,10)]:
                if min_r <= score < max_r:
                    stats['score_distribution'][f"{min_r}-{max_r}"] += 1
            
            if score >= min_score and sample:
                filtered_samples.append(sample)
                stats['passed'] += 1
            else:
                stats['failed'] += 1
        
        print(f"从缓存中筛选：总计 {stats['total']} 条数据，通过 {stats['passed']} 条，未通过 {stats['failed']} 条")
        print("分数分布:")
        for range_name, count in stats['score_distribution'].items():
            percent = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {range_name}: {count} ({percent:.1f}%)")
        
        return filtered_samples

class ContactManager:
    """联系人关系管理器"""
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.contacts_file = os.path.join(output_dir, "contacts.json")
        self.ensure_output_dir()
        self.contacts_data = self.load_all_contacts()
    
    def ensure_output_dir(self) -> None:
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def load_all_contacts(self) -> Dict:
        """加载所有联系人信息"""
        if os.path.exists(self.contacts_file):
            try:
                with open(self.contacts_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载联系人文件 {self.contacts_file} 失败: {e}")
        return {}
    
    def save_all_contacts(self) -> None:
        """保存所有联系人信息"""
        try:
            with open(self.contacts_file, 'w', encoding='utf-8') as f:
                json.dump(self.contacts_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存联系人文件 {self.contacts_file} 失败: {e}")
    
    def load_contact(self, contact_id: str) -> Dict:
        """加载单个联系人信息"""
        if contact_id in self.contacts_data:
            return self.contacts_data[contact_id].copy()
        
        # 返回默认联系人信息
        return {
            "contact_id": contact_id,
            "nickname": "",
            "remark": "",
            "relationship": "朋友",
            "relationship_detail": "",
            "first_contact_date": None,
            "last_contact_date": None,
            "message_count": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def save_contact(self, contact_info: Dict) -> None:
        """保存单个联系人信息"""
        contact_id = contact_info['contact_id']
        contact_info['updated_at'] = datetime.now().isoformat()
        self.contacts_data[contact_id] = contact_info
        self.save_all_contacts()
    
    def update_contact_from_message(self, msg: Dict) -> None:
        """从消息更新联系人信息"""
        contact_id = msg['talker']
        contact_info = self.load_contact(contact_id)
        
        # 更新统计信息（直接增加，不累加历史数据）
        contact_info['message_count'] += 1
        
        msg_date = msg['create_time'].isoformat()
        if not contact_info['first_contact_date']:
            contact_info['first_contact_date'] = msg_date
        
        # 更新最后联系时间（如果更新的时间更晚）
        if (not contact_info['last_contact_date'] or 
            msg_date > contact_info['last_contact_date']):
            contact_info['last_contact_date'] = msg_date
        
        self.save_contact(contact_info)
    
    def update_contact_from_users_data(self, contact_id: str, users_data: Dict) -> None:
        """从users.json数据更新联系人基本信息"""
        contact_info = self.load_contact(contact_id)
        
        if contact_id in users_data:
            user_data = users_data[contact_id]
            
            # 更新昵称和备注（只在为空时更新，保护手动设置）
            if user_data.get('nickname') and not contact_info.get('nickname'):
                contact_info['nickname'] = user_data['nickname']
            
            if user_data.get('remark') and not contact_info.get('remark'):
                contact_info['remark'] = user_data['remark']
        
        self.save_contact(contact_info)
    
    def batch_update_from_users_data(self, users_data: Dict) -> None:
        """批量从users.json数据更新联系人信息"""
        updated_count = 0
        for contact_id in users_data:
            old_contact = self.load_contact(contact_id)
            self.update_contact_from_users_data(contact_id, users_data)
            updated_count += 1
        
        # 批量保存，提高性能
        if updated_count > 0:
            self.save_all_contacts()
            print(f"批量更新了 {updated_count} 个联系人的基本信息")
    
    def list_all_contacts(self) -> List[Dict]:
        """列出所有联系人"""
        return list(self.contacts_data.values())
    
    def update_relationship(self, contact_id: str, relationship: str, relationship_detail: str = "") -> None:
        """更新联系人关系"""
        contact_info = self.load_contact(contact_id)
        contact_info['relationship'] = relationship
        contact_info['relationship_detail'] = relationship_detail
        self.save_contact(contact_info)
        print(f"已更新 {contact_id} 的关系为: {relationship}")
        if relationship_detail:
            print(f"详细备注: {relationship_detail}")
    
    def get_contacts_summary(self) -> Dict:
        """获取联系人统计摘要"""
        contacts = self.list_all_contacts()
        total_contacts = len(contacts)
        total_messages = sum(c.get('message_count', 0) for c in contacts)
        
        # 关系分布
        relationship_dist = {}
        for contact in contacts:
            rel = contact.get('relationship', '朋友')
            relationship_dist[rel] = relationship_dist.get(rel, 0) + 1
        
        # 最活跃联系人
        top_contacts = sorted(contacts, key=lambda x: x.get('message_count', 0), reverse=True)[:5]
        
        return {
            "total_contacts": total_contacts,
            "total_messages": total_messages,
            "relationship_distribution": relationship_dist,
            "top_active_contacts": [
                {
                    "name": c.get('remark') or c.get('nickname') or c['contact_id'][-8:],
                    "contact_id": c['contact_id'],
                    "message_count": c.get('message_count', 0)
                } for c in top_contacts
            ]
        }
    
    def reset_all_message_counts(self) -> None:
        """重置所有联系人的消息计数为0（在重新计算前调用）"""
        print("重置所有联系人的消息计数...")
        contacts = self.list_all_contacts()
        updated_count = 0
        
        for contact in contacts:
            contact_id = contact['contact_id']
            contact_info = self.load_contact(contact_id)
            contact_info['message_count'] = 0
            self.save_contact(contact_info)
            updated_count += 1
        
        print(f"已重置 {updated_count} 个联系人的消息计数")

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

class WeChatDataProcessor:
    def __init__(self, config_file: str = "config.json"):
        # 检查配置文件是否存在
        if not os.path.exists(config_file):
            print(f"配置文件 {config_file} 不存在，使用默认配置")
            self.config = None
            self.use_llm_evaluation = False
        else:
            self.config = ConfigManager(config_file)
            # 严格检查评估配置
            self.use_llm_evaluation = self.config.get('llm_evaluation.enabled', False)
            
            # 如果启用了评估，检查必要的配置
            if self.use_llm_evaluation:
                api_key = self.config.get('llm_evaluation.api_key')
                api_url = self.config.get('llm_evaluation.api_url')
                
                if not api_key or api_key == "your_api_key_here":
                    print("警告：API密钥未正确配置，禁用大模型评估")
                    self.use_llm_evaluation = False
                elif not api_url:
                    print("警告：API URL未配置，禁用大模型评估")
                    self.use_llm_evaluation = False
        
        # 创建output目录
        self.output_dir = "output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 设置默认值
        if self.config:
            self.data_dir = self.config.get('data_processing.data_dir', 'data')
            self.output_file = os.path.join(self.output_dir, 
                                          self.config.get('data_processing.output_file', 'training_data.json'))
            self.my_wxid = self.config.get('data_processing.my_wxid')
            self.MAX_INTER_MESSAGE_GAP = self.config.get('data_processing.max_inter_message_gap', 90)
            self.MAX_REPLY_DELAY = self.config.get('data_processing.max_reply_delay', 300)
            self.use_multiprocessing = self.config.get('data_processing.use_multiprocessing', True)
            self.max_workers = self.config.get('data_processing.max_workers', cpu_count())
            self.llm_max_workers = self.config.get('llm_evaluation.max_workers', 3)
            self.min_score = self.config.get('llm_evaluation.min_score', 5.0)
        else:
            # 默认配置
            self.data_dir = 'data'
            self.output_file = os.path.join(self.output_dir, 'training_data.json')
            self.my_wxid = None
            self.MAX_INTER_MESSAGE_GAP = 90
            self.MAX_REPLY_DELAY = 300
            self.use_multiprocessing = True
            self.max_workers = cpu_count()
            self.llm_max_workers = 3
            self.min_score = 5.0
        
        # 联系人管理器（使用output目录）
        self.contact_manager = ContactManager(self.output_dir)
        
        # 联系人信息数据库
        self.contact_database = {}
        
        # 增量处理状态文件（也放在output目录）
        self.state_file = os.path.join(self.output_dir, "processing_state.json")
        self.processed_files = self.load_processing_state()
        
        # 消息类型映射
        self.message_type_mapping = {
            "语音": "[语音消息]",
            "图片": "[图片]", 
            "文件": "[文件]",
            "动画表情": "[动画表情]",
            "(分享)卡片式链接": "[分享链接]",
            "合并转发的聊天记录": "[合并转发的聊天记录]",
            "引用回复": "引用回复"
        }
        
        # 初始化评估缓存（无论是否启用评估）
        self.evaluation_cache = EvaluationCache(self.output_dir)
        
        # 只有在启用评估时才初始化评估器
        if self.use_llm_evaluation and self.config:
            self.llm_evaluator = LLMEvaluator(self.config, self.output_dir)
    
    def filter_from_cache(self) -> None:
        """从评估缓存中筛选训练数据"""
        print("=== 从评估缓存筛选训练数据 ===")
        
        # 检查缓存文件是否存在
        if not os.path.exists(self.evaluation_cache.cache_file):
            print(f"错误: 评估缓存文件 {self.evaluation_cache.cache_file} 不存在")
            return
            
        # 获取配置的最低分数
        min_score = self.min_score
        print(f"使用最低分数阈值: {min_score}")
        
        # 从缓存中筛选数据
        filtered_data = self.evaluation_cache.filter_by_score(min_score)
        
        if not filtered_data:
            print("警告: 没有符合条件的数据")
            return
        
        # 获取联系人统计摘要
        contacts_summary = self.contact_manager.get_contacts_summary()
        
        # 分析语言模式
        language_patterns = self.analyze_filtered_data(filtered_data)
        
        # 保存筛选后的数据
        results = {
            "training_data": filtered_data,
            "language_patterns": language_patterns,
            "contacts_summary": contacts_summary,
            "metadata": {
                "stage": "cache_filtered",
                "total_samples": len(filtered_data),
                "processing_time": datetime.now().isoformat(),
                "parameters": {
                    "min_score": min_score
                }
            }
        }
        
        # 保存到文件
        filtered_file = os.path.join(self.output_dir, f'training_data_filtered_{min_score}.json')
        with open(filtered_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存纯训练数据
        training_only_file = os.path.join(self.output_dir, f'training_only_filtered_{min_score}.json')
        with open(training_only_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"筛选完成！从缓存中筛选出 {len(filtered_data)} 条训练数据")
        print(f"完整结果已保存到: {filtered_file}")
        print(f"纯训练数据已保存到: {training_only_file}")
    
    def analyze_filtered_data(self, filtered_data: List[Dict]) -> Dict:
        """分析筛选后的数据语言模式"""
        all_text = ""
        for sample in filtered_data:
            all_text += sample.get('output', '') + " "
        
        frequent_words = self.analyze_frequent_words(all_text, top_n=20)
        
        patterns = {
            "高频词汇": [{"词汇": word, "出现次数": count} for word, count in frequent_words],
            "样本统计": {
                "总样本数": len(filtered_data),
                "平均输出长度": sum(len(sample.get('output', '')) for sample in filtered_data) / len(filtered_data) if filtered_data else 0
            }
        }
        
        emoji_pattern = r'\[([^\]]+)\]'
        emojis = re.findall(emoji_pattern, all_text)
        emoji_count = {}
        for emoji in emojis:
            emoji_count[emoji] = emoji_count.get(emoji, 0) + 1
        
        patterns["常用表情"] = [{"表情": emoji, "出现次数": count} 
                            for emoji, count in sorted(emoji_count.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        return patterns
    
    def load_processing_state(self) -> Dict:
        """加载处理状态"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载处理状态失败: {e}")
        return {}
    
    def save_processing_state(self) -> None:
        """保存处理状态"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存处理状态失败: {e}")
    
    def get_file_hash(self, file_path: str) -> str:
        """获取文件哈希值用于增量处理"""
        try:
            stat = os.stat(file_path)
            content = f"{file_path}|{stat.st_mtime}|{stat.st_size}"
            return hashlib.md5(content.encode('utf-8')).hexdigest()
        except Exception:
            return ""
        
    def load_contact_database(self) -> None:
        """从所有users.json文件中加载联系人信息，并更新contacts目录"""
        contact_dirs = glob.glob(os.path.join(self.data_dir, "*/"))
        
        for contact_dir in contact_dirs:
            users_file = os.path.join(contact_dir, "users.json")
            if os.path.exists(users_file):
                try:
                    with open(users_file, 'r', encoding='utf-8') as f:
                        users_data = json.load(f)
                        
                        # 更新全局联系人数据库
                        self.contact_database.update(users_data)
                        
                        # 批量更新联系人文件
                        self.contact_manager.batch_update_from_users_data(users_data)
                        
                        print(f"从 {users_file} 加载并更新了 {len(users_data)} 个联系人信息")
                        
                except Exception as e:
                    print(f"加载联系人文件 {users_file} 时出错: {e}")
        
        print(f"联系人数据库总共包含 {len(self.contact_database)} 个联系人")
        
    def load_all_csv_files(self) -> List[Dict]:
        """增量加载data文件夹中所有子目录的CSV文件"""
        all_messages = []
        new_files_count = 0
        
        contact_dirs = glob.glob(os.path.join(self.data_dir, "*/"))
        
        print("正在加载CSV文件...")
        for contact_dir in tqdm(contact_dirs, desc="处理联系人目录"):
            csv_files = glob.glob(os.path.join(contact_dir, "*.csv"))
            
            for csv_file in csv_files:
                try:
                    file_hash = self.get_file_hash(csv_file)
                    
                    # 检查文件是否已处理过
                    if csv_file in self.processed_files and self.processed_files[csv_file] == file_hash:
                        continue  # 跳过已处理的文件
                    
                    df = pd.read_csv(csv_file)
                    messages = df.to_dict('records')
                    all_messages.extend(messages)
                    
                    # 记录已处理的文件
                    self.processed_files[csv_file] = file_hash
                    new_files_count += 1
                    
                except Exception as e:
                    print(f"处理文件 {csv_file} 时出错: {e}")
        
        if new_files_count == 0:
            print("没有发现新的CSV文件，使用缓存数据")
        else:
            print(f"处理了 {new_files_count} 个新文件")
            self.save_processing_state()
                    
        return all_messages
    
    def clean_and_preprocess(self, messages: List[Dict]) -> List[Dict]:
        """数据清洗与预处理，延迟联系人信息更新以提高性能"""
        cleaned_messages = []
        contact_updates = {}  # 缓存联系人更新信息
        
        # 先重置所有联系人的消息计数，确保只计算当前数据集的消息数
        self.contact_manager.reset_all_message_counts()
        
        print("正在清洗和预处理数据...")
        for msg in tqdm(messages, desc="清洗消息"):
            try:
                create_time = pd.to_datetime(msg['CreateTime'])
                msg_content = self.process_message_content(msg)
                
                cleaned_msg = {
                    'id': msg['id'],
                    'create_time': create_time,
                    'is_sender': int(msg['is_sender']),
                    'talker': msg['talker'],
                    'room_name': msg['room_name'] if pd.notna(msg['room_name']) else None,
                    'type_name': msg['type_name'],
                    'content': msg_content,
                    'original_msg': msg['msg'] if pd.notna(msg['msg']) else ""
                }
                
                # 缓存联系人信息更新（避免频繁IO）
                talker = cleaned_msg['talker']
                if talker not in contact_updates:
                    contact_updates[talker] = {
                        'message_count': 0,
                        'first_contact_date': create_time,
                        'last_contact_date': create_time
                    }
                contact_updates[talker]['message_count'] += 1
                if create_time < contact_updates[talker]['first_contact_date']:
                    contact_updates[talker]['first_contact_date'] = create_time
                if create_time > contact_updates[talker]['last_contact_date']:
                    contact_updates[talker]['last_contact_date'] = create_time
                
                cleaned_messages.append(cleaned_msg)
                
            except Exception as e:
                continue
        
        # 批量更新联系人信息
        self._batch_update_contacts_from_preprocessing(contact_updates)
        
        cleaned_messages.sort(key=lambda x: x['create_time'])
        print(f"清洗后共有 {len(cleaned_messages)} 条消息")
        return cleaned_messages
    
    def _batch_update_contacts_from_preprocessing(self, contact_updates: Dict) -> None:
        """从预处理阶段批量更新联系人信息"""
        if not contact_updates:
            return
            
        print(f"批量更新 {len(contact_updates)} 个联系人的预处理信息...")
        
        for talker, updates in tqdm(contact_updates.items(), desc="更新联系人预处理信息"):
            try:
                contact_info = self.contact_manager.load_contact(talker)
                
                # 直接设置消息计数，而不是累加（因为已经在reset_all_message_counts中重置了）
                contact_info['message_count'] = updates['message_count']
                
                # 更新首次联系时间
                first_date_str = updates['first_contact_date'].isoformat()
                if (not contact_info.get('first_contact_date') or 
                    first_date_str < contact_info.get('first_contact_date', '')):
                    contact_info['first_contact_date'] = first_date_str
                
                # 更新最后联系时间  
                last_date_str = updates['last_contact_date'].isoformat()
                if (not contact_info.get('last_contact_date') or 
                    last_date_str > contact_info.get('last_contact_date', '')):
                    contact_info['last_contact_date'] = last_date_str
                
                self.contact_manager.save_contact(contact_info)
                
            except Exception as e:
                print(f"更新联系人 {talker} 预处理信息时出错: {e}")
    
    def process_message_content(self, msg: Dict) -> str:
        """处理不同类型的消息内容"""
        type_name = msg['type_name']
        msg_content = msg['msg'] if pd.notna(msg['msg']) else ""
        
        if type_name == "文本":
            return msg_content
        
        if type_name == "语音":
            if pd.notna(msg.get('src')) and msg['src'].strip():
                return f"[语音转文字: {msg['src']}]"
            else:
                return "[语音消息]"
        
        return self.message_type_mapping.get(type_name, f"[{type_name}]")
    
    def extract_message_blocks(self, messages: List[Dict]) -> List[Tuple[List[Dict], List[Dict]]]:
        """提取对话回合：(对方消息块, 我的回复块) 的配对"""
        # 检查是否可以使用多进程
        can_use_multiprocess = len(messages) >= 1000
        
        if not self.use_multiprocessing or not can_use_multiprocess:
            return self._extract_message_blocks_single_process(messages)
        
        # 多进程处理
        chunk_size = max(len(messages) // self.max_workers, 1000)
        chunks = []
        
        # 创建进程间共享的进度字典
        manager = Manager()
        progress_dict = manager.dict()
        
        for i in range(0, len(messages), chunk_size):
            end_idx = min(i + chunk_size, len(messages))
            process_id = len(chunks)
            # 初始化进度为0
            progress_dict[process_id] = 0
            # 传递参数及进度字典
            chunks.append((messages, i, end_idx, process_id, self.MAX_INTER_MESSAGE_GAP, self.MAX_REPLY_DELAY, progress_dict))
        
        conversation_rounds = []
        all_contact_updates = {}
        
        print(f"使用多进程处理，分为 {len(chunks)} 个块...")
        print(f"每个进程将处理约 {chunk_size} 条消息")
        
        # 创建一个总进度条
        total_messages = len(messages)
        with tqdm(total=total_messages, desc="多进程提取对话", smoothing=0.1) as pbar:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(extract_message_blocks_process_worker, chunk) for chunk in chunks]
                
                # 持续更新进度条，直到所有进程完成
                last_total = 0
                while not all(future.done() for future in futures):
                    # 计算所有进程的进度总和
                    current_total = sum(progress_dict.values())
                    # 更新进度条增量
                    if current_total > last_total:
                        pbar.update(current_total - last_total)
                        last_total = current_total
                    time.sleep(0.1)  # 避免过度刷新
                
                # 处理结果
                for future in as_completed(futures):
                    try:
                        chunk_rounds, contact_updates = future.result()
                        conversation_rounds.extend(chunk_rounds)
                        
                        # 合并联系人更新（这里不需要修改，因为这是统计对话回合数据，不是累加message_count）
                        for talker, updates in contact_updates.items():
                            if talker not in all_contact_updates:
                                all_contact_updates[talker] = {'message_count': 0, 'last_contact_date': None}
                            all_contact_updates[talker]['message_count'] += updates['message_count']
                            if (all_contact_updates[talker]['last_contact_date'] is None or 
                                updates['last_contact_date'] > all_contact_updates[talker]['last_contact_date']):
                                all_contact_updates[talker]['last_contact_date'] = updates['last_contact_date']
                    except Exception as e:
                        print(f"处理进程块时出错: {e}")
                
                # 确保进度条完成
                pbar.update(total_messages - pbar.n)
        
        # 注意：这里不再进行联系人信息更新，因为已经在clean_and_preprocess阶段完成
        # self._batch_update_contacts_threadsafe(all_contact_updates)
        
        print(f"多进程提取完成！总共提取到 {len(conversation_rounds)} 个对话回合")
        return conversation_rounds
    
    def _extract_message_blocks_single_process(self, messages: List[Dict]) -> List[Tuple[List[Dict], List[Dict]]]:
        """单进程提取对话回合"""
        conversation_rounds = []
        i = 0
        
        print("正在提取对话回合...")
        with tqdm(total=len(messages), desc="处理消息") as pbar:
            while i < len(messages):
                other_block = self.find_other_message_block(messages, i)
                if not other_block:
                    i += 1
                    pbar.update(1)
                    continue
                
                i = other_block[-1]['index'] + 1
                
                my_block = self.find_my_reply_block(messages, i, other_block[-1])
                if not my_block:
                    pbar.update(len(other_block))
                    continue
                
                if self.is_valid_reply_timing(other_block[-1], my_block[0]):
                    conversation_rounds.append((other_block, my_block))
                
                i = my_block[-1]['index'] + 1
                pbar.update(len(other_block) + len(my_block))
        
        print(f"总共提取到 {len(conversation_rounds)} 个对话回合")
        return conversation_rounds
    
    def find_other_message_block(self, messages: List[Dict], start_idx: int) -> Optional[List[Dict]]:
        """寻找对方的消息块"""
        if start_idx >= len(messages):
            return None
        
        msg = messages[start_idx]
        if msg['is_sender'] == 1:
            return None
        
        block = [{'index': start_idx, **msg}]
        current_talker = msg['talker']
        last_time = msg['create_time']
        
        for i in range(start_idx + 1, len(messages)):
            next_msg = messages[i]
            
            if next_msg['is_sender'] == 1:
                break
            
            if next_msg['talker'] != current_talker:
                break
            
            time_gap = (next_msg['create_time'] - last_time).total_seconds()
            if time_gap > self.MAX_INTER_MESSAGE_GAP:
                break
            
            block.append({'index': i, **next_msg})
            last_time = next_msg['create_time']
        
        return block if len(block) > 0 else None
    
    def find_my_reply_block(self, messages: List[Dict], start_idx: int, last_other_msg: Dict) -> Optional[List[Dict]]:
        """寻找我的回复块"""
        if start_idx >= len(messages):
            return None
        
        first_my_msg = None
        first_my_idx = start_idx
        
        for i in range(start_idx, min(start_idx + 10, len(messages))):
            if messages[i]['is_sender'] == 1:
                reply_delay = (messages[i]['create_time'] - last_other_msg['create_time']).total_seconds()
                if reply_delay <= self.MAX_REPLY_DELAY:
                    first_my_msg = messages[i]
                    first_my_idx = i
                    break
        
        if not first_my_msg:
            return None
        
        block = [{'index': first_my_idx, **first_my_msg}]
        last_time = first_my_msg['create_time']
        
        for i in range(first_my_idx + 1, len(messages)):
            next_msg = messages[i]
            
            if next_msg['is_sender'] != 1:
                break
            
            time_gap = (next_msg['create_time'] - last_time).total_seconds()
            if time_gap > self.MAX_INTER_MESSAGE_GAP:
                break
            
            block.append({'index': i, **next_msg})
            last_time = next_msg['create_time']
        
        return block if len(block) > 0 else None
    
    def is_valid_reply_timing(self, last_other_msg: Dict, first_my_msg: Dict) -> bool:
        """检查回复时间是否合理"""
        reply_delay = (first_my_msg['create_time'] - last_other_msg['create_time']).total_seconds()
        return 0 <= reply_delay <= self.MAX_REPLY_DELAY
    
    def build_context(self, messages: List[Dict], reply_time: datetime) -> List[Dict]:
        """构建历史上下文（当日、当前时间点之前）"""
        reply_date = reply_time.date()
        context_messages = []
        
        for msg in messages:
            msg_date = msg['create_time'].date()
            if msg_date == reply_date and msg['create_time'] < reply_time:
                context_messages.append(msg)
        
        return context_messages
    
    def format_message_block_content(self, message_block: List[Dict]) -> str:
        """格式化消息块内容，使用<return>分割连续消息"""
        contents = [msg['content'] for msg in message_block]
        return "<return>".join(contents)
    
    def format_training_data(self, conversation_rounds: List[Tuple], all_messages: List[Dict]) -> List[Dict]:
        """格式化为训练数据"""
        if len(conversation_rounds) < 100 or not self.use_multiprocessing:
            # 少量数据或禁用多处理时使用单线程
            return self._format_training_data_single_thread(conversation_rounds, all_messages)
        
        # 预先提取所有联系人信息（避免在子进程中使用包含锁的对象）
        print("正在预提取联系人信息...")
        contact_data = {}
        all_talkers = set()
        for other_block, my_block in conversation_rounds:
            for msg in other_block:
                all_talkers.add(msg['talker'])
        
        for talker in tqdm(all_talkers, desc="提取联系人信息"):
            contact_info = self.contact_manager.load_contact(talker)
            contact_data[talker] = contact_info
        
        # 获取我的名字
        my_name = self.get_my_name()
        
        # 多进程处理
        chunk_size = max(len(conversation_rounds) // self.max_workers, 10)
        chunks = []
        
        # 创建进程间共享的进度字典
        manager = Manager()
        progress_dict = manager.dict()
        
        for i in range(0, len(conversation_rounds), chunk_size):
            end_idx = min(i + chunk_size, len(conversation_rounds))
            chunk_rounds = conversation_rounds[i:end_idx]
            process_id = len(chunks)
            # 初始化进度为0
            progress_dict[process_id] = 0
            chunks.append((chunk_rounds, all_messages, contact_data, my_name, process_id, progress_dict))
        
        training_data = []
        
        print(f"使用多进程格式化训练数据，分为 {len(chunks)} 个块...")
        print(f"每个进程将处理约 {chunk_size} 个对话回合")
        
        # 创建一个总进度条
        total_rounds = len(conversation_rounds)
        with tqdm(total=total_rounds, desc="多进程格式化", smoothing=0.1) as pbar:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(format_training_data_process_worker, chunk) for chunk in chunks]
                
                # 持续更新进度条，直到所有进程完成
                last_total = 0
                while not all(future.done() for future in futures):
                    # 计算所有进程的进度总和
                    current_total = sum(progress_dict.values())
                    # 更新进度条增量
                    if current_total > last_total:
                        pbar.update(current_total - last_total)
                        last_total = current_total
                    time.sleep(0.1)  # 避免过度刷新
                
                # 处理结果
                for future in as_completed(futures):
                    try:
                        chunk_training_data = future.result()
                        training_data.extend(chunk_training_data)
                    except Exception as e:
                        print(f"格式化进程块时出错: {e}")
                
                # 确保进度条完成
                pbar.update(total_rounds - pbar.n)
        
        print(f"多进程格式化完成！生成了 {len(training_data)} 条训练数据")
        return training_data
    
    def _format_training_data_single_thread(self, conversation_rounds: List[Tuple], all_messages: List[Dict]) -> List[Dict]:
        """单线程格式化训练数据（原方法）"""
        training_data = []
        
        print("正在格式化训练数据...")
        for other_block, my_block in tqdm(conversation_rounds, desc="格式化对话"):
            try:
                reply_time = my_block[0]['create_time']
                context_messages = self.build_context(all_messages, reply_time)
                
                other_content = self.format_message_block_content(other_block)
                my_reply = self.format_message_block_content(my_block)
                history_text = self.build_history_text(context_messages, other_block)
                
                # 获取对方的详细联系人信息
                talker_id = other_block[0]['talker']
                contact_info = self.contact_manager.load_contact(talker_id)
                other_name = self.get_other_name(other_block[0])
                
                # 构建详细的联系人信息
                relationship = contact_info.get('relationship', '朋友')
                relationship_detail = contact_info.get('relationship_detail', '')
                first_contact_date = contact_info.get('first_contact_date', '')
                
                # 格式化首次联系时间
                first_contact_str = ""
                if first_contact_date:
                    try:
                        if isinstance(first_contact_date, str):
                            from datetime import datetime
                            first_contact_dt = datetime.fromisoformat(first_contact_date.replace('Z', '+00:00'))
                            first_contact_str = first_contact_dt.strftime("%Y年%m月%d日")
                        else:
                            first_contact_str = first_contact_date.strftime("%Y年%m月%d日")
                    except:
                        first_contact_str = ""
                
                # 格式化当前回复时间
                current_time_str = reply_time.strftime("%Y年%m月%d日 %H:%M")
                
                # 构建包含联系人信息的instruction
                instruction_parts = [
                    f"你是{self.get_my_name()}，正在和{other_name}聊天。"
                ]
                
                # 添加关系信息
                if relationship and relationship != '朋友':
                    instruction_parts.append(f"对方是你的{relationship}。")
                elif relationship == '朋友':
                    instruction_parts.append(f"对方是你的朋友。")
                
                # 添加关系详细描述
                if relationship_detail and relationship_detail.strip():
                    instruction_parts.append(f"{relationship_detail}。")
                
                # 添加首次联系时间
                if first_contact_str:
                    instruction_parts.append(f"你们从{first_contact_str}开始联系。")
                
                # 添加当前时间
                instruction_parts.append(f"现在是{current_time_str}。")
                
                # 添加回复指引
                instruction_parts.append("根据聊天记录和对方的最新消息，用你的风格回复。")
                
                instruction = " ".join(instruction_parts)
                
                training_sample = {
                    "instruction": instruction,
                    "input": f"历史记录:\n{history_text}\n\n对方最新消息:\n{other_content}",
                    "output": my_reply
                }
                
                training_data.append(training_sample)
                
            except Exception as e:
                continue
        
        return training_data
    
    def evaluate_training_data(self, training_data: List[Dict]) -> Tuple[List[Dict], Dict]:
        """使用大模型评估训练数据质量"""
        if not self.use_llm_evaluation:
            return training_data, {}
        
        filtered_data = []
        evaluation_results = {
            'total_samples': len(training_data),
            'evaluated_samples': 0,
            'passed_samples': 0,
            'failed_samples': 0,
            'average_score': 0.0,
            'score_distribution': {},
            'evaluation_details': []
        }
        
        print(f"正在使用大模型评估 {len(training_data)} 条训练数据...")
        
        def evaluate_single_sample(sample_with_index):
            index, sample = sample_with_index
            try:
                score, response = self.llm_evaluator.evaluate_sample(sample)
                return index, sample, score, response, None
            except Exception as e:
                return index, sample, 0.0, "", str(e)
        
        scores = []
        
        with ThreadPoolExecutor(max_workers=self.llm_max_workers) as executor:
            futures = [executor.submit(evaluate_single_sample, (i, sample)) 
                      for i, sample in enumerate(training_data)]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="评估进度"):
                index, sample, score, response, error = future.result()
                
                evaluation_results['evaluated_samples'] += 1
                scores.append(score)
                
                evaluation_detail = {
                    'index': index,
                    'score': score,
                    'response': response,
                    'error': error,
                    'passed': score >= self.min_score
                }
                evaluation_results['evaluation_details'].append(evaluation_detail)
                
                if score >= self.min_score:
                    filtered_data.append(sample)
                    evaluation_results['passed_samples'] += 1
                else:
                    evaluation_results['failed_samples'] += 1
        
        # 计算统计信息
        if scores:
            evaluation_results['average_score'] = sum(scores) / len(scores)
            
            # 分数分布
            score_ranges = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
            for min_score, max_score in score_ranges:
                range_name = f"{min_score}-{max_score}"
                count = sum(1 for s in scores if min_score <= s < max_score)
                evaluation_results['score_distribution'][range_name] = count
        
        print(f"评估完成！通过 {evaluation_results['passed_samples']}/{evaluation_results['total_samples']} 条数据")
        print(f"平均分数: {evaluation_results['average_score']:.2f}")
        
        return filtered_data, evaluation_results
    
    def build_history_text(self, context_messages: List[Dict], current_other_block: List[Dict]) -> str:
        """构建历史对话文本"""
        history_lines = []
        
        for msg in context_messages:
            time_str = msg['create_time'].strftime("%H:%M:%S")
            sender = "我" if msg['is_sender'] == 1 else self.get_other_name(msg)
            history_lines.append(f"[{time_str}] {sender}: {msg['content']}")
        
        for msg in current_other_block:
            time_str = msg['create_time'].strftime("%H:%M:%S")
            sender = self.get_other_name(msg)
            history_lines.append(f"[{time_str}] {sender}: {msg['content']}")
        
        return "\n".join(history_lines[-20:])
    
    def get_my_name(self) -> str:
        """获取我的名字"""
        if self.my_wxid in self.contact_database:
            contact_info = self.contact_database[self.my_wxid]
            if contact_info.get('remark') and contact_info['remark'].strip():
                return contact_info['remark']
            elif contact_info.get('nickname') and contact_info['nickname'].strip():
                return contact_info['nickname']
        
        return "骆明宇"
    
    def get_other_name(self, msg: Dict) -> str:
        """从数据库中获取对方的昵称"""
        talker = msg['talker']
        
        # 先从联系人管理器获取（这里会包含从users.json提取的信息）
        contact_info = self.contact_manager.load_contact(talker)
        if contact_info.get('remark') and contact_info['remark'].strip():
            return contact_info['remark']
        elif contact_info.get('nickname') and contact_info['nickname'].strip():
            return contact_info['nickname']
        
        # 再从原始数据库获取
        if talker in self.contact_database:
            contact_data = self.contact_database[talker]
            if contact_data.get('remark') and contact_data['remark'].strip():
                return contact_data['remark']
            elif contact_data.get('nickname') and contact_data['nickname'].strip():
                return contact_data['nickname']
        
        return talker[-8:] if len(talker) > 8 else talker
    
    def analyze_frequent_words(self, text: str, top_n: int = 50) -> List[Tuple[str, int]]:
        """动态分析常用词汇"""
        words = jieba.lcut(text)
        
        filtered_words = []
        for word in words:
            word = word.strip()
            if len(word) >= 2 and word.isalpha() or word in ['哈哈', '草', '寄', 'nb', '捏', 'okok', '没事', '笑死', '牛逼']:
                filtered_words.append(word)
        
        word_count = Counter(filtered_words)
        return word_count.most_common(top_n)
    
    def analyze_language_patterns(self, messages: List[Dict]) -> Dict:
        """分析个人语言模式"""
        my_messages = [msg for msg in messages if msg['is_sender'] == 1]
        all_text = " ".join([msg['content'] for msg in my_messages])
        frequent_words = self.analyze_frequent_words(all_text, top_n=20)
        
        patterns = {
            "高频词汇": [{"词汇": word, "出现次数": count} for word, count in frequent_words],
            "常用表情": [],
            "消息特点": {
                "总消息数": len(my_messages),
                "平均消息长度": sum(len(msg['content']) for msg in my_messages) / len(my_messages) if my_messages else 0,
                "短消息比例": len([msg for msg in my_messages if len(msg['content']) <= 10]) / len(my_messages) if my_messages else 0
            },
            "联系人统计": self.analyze_contact_statistics(messages)
        }
        
        emoji_pattern = r'\[([^\]]+)\]'
        emojis = re.findall(emoji_pattern, all_text)
        emoji_count = {}
        for emoji in emojis:
            emoji_count[emoji] = emoji_count.get(emoji, 0) + 1
        
        patterns["常用表情"] = [{"表情": emoji, "出现次数": count} 
                            for emoji, count in sorted(emoji_count.items(), key=lambda x: x[1], reverse=True)]
        
        return patterns
    
    def analyze_contact_statistics(self, messages: List[Dict]) -> Dict:
        """分析联系人统计信息"""
        contact_stats = {}
        
        for msg in messages:
            talker = msg['talker']
            if talker not in contact_stats:
                contact_stats[talker] = {
                    "name": self.get_other_name(msg),
                    "total_messages": 0,
                    "my_messages": 0,
                    "other_messages": 0
                }
            
            contact_stats[talker]["total_messages"] += 1
            if msg['is_sender'] == 1:
                contact_stats[talker]["my_messages"] += 1
            else:
                contact_stats[talker]["other_messages"] += 1
        
        sorted_contacts = sorted(contact_stats.items(), key=lambda x: x[1]["total_messages"], reverse=True)
        
        return {
            "联系人总数": len(contact_stats),
            "活跃联系人详情": dict(sorted_contacts[:10])
        }
    
    def convert_timestamps_to_strings(self, obj):
        """递归转换对象中的Timestamp为字符串"""
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self.convert_timestamps_to_strings(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_timestamps_to_strings(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_timestamps_to_strings(item) for item in obj)
        else:
            return obj
    
    def save_stage1_results(self, training_data: List[Dict], language_patterns: Dict, 
                           cleaned_messages: List[Dict], conversation_rounds: List[Tuple]) -> None:
        """保存阶段1的结果"""
        # 转换Timestamp对象为字符串
        cleaned_messages_serializable = self.convert_timestamps_to_strings(cleaned_messages)
        conversation_rounds_serializable = self.convert_timestamps_to_strings([
            (
                [msg for msg in other_block],
                [msg for msg in my_block]
            ) for other_block, my_block in conversation_rounds
        ])
        
        stage1_results = {
            "training_data": training_data,
            "language_patterns": language_patterns,
            "cleaned_messages": cleaned_messages_serializable,
            "conversation_rounds": conversation_rounds_serializable,
            "metadata": {
                "stage": 1,
                "total_samples": len(training_data),
                "processing_time": datetime.now().isoformat(),
                "parameters": {
                    "MAX_INTER_MESSAGE_GAP": self.MAX_INTER_MESSAGE_GAP,
                    "MAX_REPLY_DELAY": self.MAX_REPLY_DELAY,
                    "use_multiprocessing": self.use_multiprocessing
                }
            }
        }
        
        stage1_file = os.path.join(self.output_dir, 'stage1_results.json')
        with open(stage1_file, 'w', encoding='utf-8') as f:
            json.dump(stage1_results, f, ensure_ascii=False, indent=2)
        
        # 也保存一份纯训练数据供查看
        training_only_file = os.path.join(self.output_dir, 'stage1_training_data.json')
        with open(training_only_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"阶段1结果已保存到: {stage1_file}")
        print(f"训练数据已保存到: {training_only_file}")
    
    def process_stage1_extract_and_contacts(self) -> None:
        """阶段1: 数据提取和联系人信息建立"""
        print("=== 阶段1: 数据提取和联系人信息建立 ===")
        print(f"使用多进程: {self.use_multiprocessing}")
        
        # 0. 加载联系人信息数据库
        print("\n=== 步骤0: 加载联系人信息数据库 ===")
        self.load_contact_database()
        
        # 1. 加载所有CSV文件
        print("\n=== 步骤1: 加载数据文件 ===")
        raw_messages = self.load_all_csv_files()
        
        # 2. 数据清洗与预处理
        print("\n=== 步骤2: 数据清洗与预处理 ===")
        cleaned_messages = self.clean_and_preprocess(raw_messages)
        
        # 3. 提取对话回合
        print("\n=== 步骤3: 提取对话回合 ===")
        conversation_rounds = self.extract_message_blocks(cleaned_messages)
        
        # 4. 生成训练数据（不进行评估）
        print("\n=== 步骤4: 生成初始训练数据 ===")
        training_data = self.format_training_data(conversation_rounds, cleaned_messages)
        
        # 5. 分析语言模式
        print("\n=== 步骤5: 分析语言模式 ===")
        language_patterns = self.analyze_language_patterns(cleaned_messages)
        
        # 6. 保存阶段1结果
        print("\n=== 步骤6: 保存阶段1结果 ===")
        self.save_stage1_results(training_data, language_patterns, cleaned_messages, conversation_rounds)
        
        print(f"\n阶段1完成！生成了 {len(training_data)} 条初始训练数据")
        print(f"联系人信息已保存到: {self.contact_manager.contacts_file}")
        print("现在您可以编辑联系人关系，然后运行阶段2进行质量评估")
    
    def process_stage2_evaluation_and_final(self) -> None:
        """阶段2: 大模型评估和最终数据集生成"""
        print("=== 阶段2: 大模型评估和最终数据集生成 ===")
        print(f"使用大模型评估: {self.use_llm_evaluation}")
        
        # 加载阶段1的结果
        print("\n=== 步骤1: 加载阶段1结果 ===")
        stage1_data = self.load_stage1_results()
        if not stage1_data:
            print("错误: 未找到阶段1的结果，请先运行阶段1")
            return
        
        training_data = stage1_data['training_data']
        language_patterns = stage1_data['language_patterns']
        
        print(f"加载了 {len(training_data)} 条训练数据")
        
        # 重新加载联系人信息（可能已被编辑）
        print("\n=== 步骤2: 重新加载联系人信息 ===")
        self.contact_manager = ContactManager(self.output_dir)
        print(f"加载了 {len(self.contact_manager.list_all_contacts())} 个联系人")
        
        # 更新训练数据中的联系人信息
        print("\n=== 步骤3: 更新训练数据中的联系人信息 ===")
        updated_training_data = self.update_training_data_with_contacts(training_data)
        
        evaluation_results = None
        final_training_data = updated_training_data
        
        # 大模型评估（如果启用）
        if self.use_llm_evaluation:
            print("\n=== 步骤4: 大模型质量评估 ===")
            final_training_data, evaluation_results = self.evaluate_training_data(updated_training_data)
        else:
            print("\n=== 步骤4: 跳过大模型评估（已禁用） ===")
        
        # 保存最终结果
        print("\n=== 步骤5: 保存最终结果 ===")
        self.save_final_results(final_training_data, language_patterns, evaluation_results)
        
        print(f"\n阶段2完成！最终生成了 {len(final_training_data)} 条训练数据")
        if evaluation_results:
            print(f"原始数据: {evaluation_results['total_samples']} 条")
            print(f"通过筛选: {evaluation_results['passed_samples']} 条")
            print(f"平均分数: {evaluation_results['average_score']:.2f}")
    
    def load_stage1_results(self) -> Optional[Dict]:
        """加载阶段1的结果"""
        stage1_file = os.path.join(self.output_dir, 'stage1_results.json')
        if not os.path.exists(stage1_file):
            return None
        
        try:
            with open(stage1_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载阶段1结果失败: {e}")
            return None
    
    def update_training_data_with_contacts(self, training_data: List[Dict]) -> List[Dict]:
        """使用更新后的联系人信息重新生成训练数据"""
        updated_data = []
        
        print("正在更新训练数据中的联系人信息...")
        for sample in tqdm(training_data, desc="更新训练样本"):
            try:
                # 从instruction中提取对方姓名信息，然后更新
                instruction = sample['instruction']
                
                # 提取对方ID（这里需要从原始数据中获取，暂时保持原样）
                # TODO: 可以在阶段1保存更多元数据来支持这个功能
                
                updated_data.append(sample)
            except Exception as e:
                continue
        
        return updated_data
    
    def save_final_results(self, training_data: List[Dict], language_patterns: Dict, 
                          evaluation_results: Dict = None) -> None:
        """保存最终结果"""
        # 获取联系人统计摘要
        contacts_summary = self.contact_manager.get_contacts_summary()
        
        results = {
            "training_data": training_data,
            "language_patterns": language_patterns,
            "contacts_summary": contacts_summary,
            "metadata": {
                "stage": 2,
                "total_samples": len(training_data),
                "processing_time": datetime.now().isoformat(),
                "parameters": {
                    "MAX_INTER_MESSAGE_GAP": self.MAX_INTER_MESSAGE_GAP,
                    "MAX_REPLY_DELAY": self.MAX_REPLY_DELAY,
                    "use_multiprocessing": self.use_multiprocessing,
                    "use_llm_evaluation": self.use_llm_evaluation
                }
            }
        }
        
        if evaluation_results:
            results["evaluation_results"] = evaluation_results
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存纯训练数据
        training_only_file = os.path.join(self.output_dir, 'training_data_training_only.json')
        with open(training_only_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        # 如果使用了大模型评估，保存筛选后的数据
        if self.use_llm_evaluation and evaluation_results:
            filtered_file = os.path.join(self.output_dir, 'training_data_filtered.json')
            with open(filtered_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            # 保存评估报告
            evaluation_report_file = os.path.join(self.output_dir, 'evaluation_report.json')
            with open(evaluation_report_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print(f"最终训练数据已保存到: {training_only_file}")
        print(f"完整结果已保存到: {self.output_file}")
        print(f"联系人信息已保存到: {self.contact_manager.contacts_file}")
        if self.use_llm_evaluation:
            print(f"筛选后数据已保存到: {filtered_file}")
            print(f"评估报告已保存到: {evaluation_report_file}")

    def process(self) -> None:
        """主处理流程（兼容模式，执行完整流程）"""
        print("开始处理微信聊天数据...")
        print("注意：建议使用分阶段处理模式")
        print("  stage1: 数据提取和联系人信息")
        print("  stage2: 大模型评估和最终数据集")
        print(f"使用多进程: {self.use_multiprocessing}")
        print(f"使用大模型评估: {self.use_llm_evaluation}")
        
        # 执行阶段1
        self.process_stage1_extract_and_contacts()
        
        # 执行阶段2
        print("\n" + "="*50)
        self.process_stage2_evaluation_and_final()

    def manage_contacts_interactive(self) -> None:
        """交互式联系人管理"""
        while True:
            print("\n=== 联系人关系管理 ===")
            print("1. 查看所有联系人")
            print("2. 搜索联系人")
            print("3. 更新联系人关系")
            print("4. 导出联系人列表")
            print("5. 返回主菜单")
            
            choice = input("请选择操作 (1-5): ").strip()
            
            if choice == '1':
                self.show_all_contacts()
            elif choice == '2':
                self.search_contacts()
            elif choice == '3':
                self.update_contact_relationship()
            elif choice == '4':
                self.export_contacts()
            elif choice == '5':
                break
            else:
                print("无效选择，请重新输入")
    
    def show_all_contacts(self) -> None:
        """显示所有联系人"""
        contacts = self.contact_manager.list_all_contacts()
        if not contacts:
            print("暂无联系人记录")
            return
        
        # 按消息数量排序
        contacts.sort(key=lambda x: x['message_count'], reverse=True)
        
        print(f"\n共有 {len(contacts)} 个联系人:")
        print("-" * 100)
        print(f"{'序号':<4} {'昵称':<20} {'备注':<20} {'关系':<10} {'消息数':<8} {'最后联系':<20}")
        print("-" * 100)
        
        for i, contact in enumerate(contacts[:20], 1):  # 只显示前20个
            nickname = contact.get('nickname', '')[:19] or contact['contact_id'][-8:]
            remark = contact.get('remark', '')[:19]
            relationship = contact.get('relationship', '')[:9]
            msg_count = contact.get('message_count', 0)
            last_contact = contact.get('last_contact_date', '')[:19] if contact.get('last_contact_date') else ''
            
            print(f"{i:<4} {nickname:<20} {remark:<20} {relationship:<10} {msg_count:<8} {last_contact:<20}")
        
        if len(contacts) > 20:
            print(f"... 还有 {len(contacts) - 20} 个联系人")
    
    def search_contacts(self) -> None:
        """搜索联系人"""
        keyword = input("请输入搜索关键词（昵称/备注/ID）: ").strip()
        if not keyword:
            print("关键词不能为空")
            return
        
        contacts = self.contact_manager.list_all_contacts()
        matches = []
        
        for contact in contacts:
            if (keyword.lower() in contact.get('nickname', '').lower() or
                keyword.lower() in contact.get('remark', '').lower() or
                keyword.lower() in contact['contact_id'].lower()):
                matches.append(contact)
        
        if not matches:
            print("未找到匹配的联系人")
            return
        
        print(f"\n找到 {len(matches)} 个匹配的联系人:")
        for i, contact in enumerate(matches, 1):
            display_name = contact.get('remark') or contact.get('nickname') or contact['contact_id'][-8:]
            relationship = contact.get('relationship', '朋友')
            print(f"{i}. {display_name} (ID: {contact['contact_id'][-8:]}) - {relationship}")
    
    def update_contact_relationship(self) -> None:
        """更新联系人关系"""
        keyword = input("请输入要更新的联系人（昵称/备注/ID）: ").strip()
        if not keyword:
            print("输入不能为空")
            return
        
        # 搜索联系人
        contacts = self.contact_manager.list_all_contacts()
        matches = []
        
        for contact in contacts:
            if (keyword.lower() in contact.get('nickname', '').lower() or
                keyword.lower() in contact.get('remark', '').lower() or
                keyword.lower() in contact['contact_id'].lower()):
                matches.append(contact)
        
        if not matches:
            print("未找到匹配的联系人")
            return
        
        if len(matches) > 1:
            print("找到多个匹配的联系人:")
            for i, contact in enumerate(matches, 1):
                print(f"{i}. {contact.get('nickname', contact['contact_id'])} "
                      f"({contact.get('remark', '无备注')})")
            
            try:
                choice = int(input("请选择序号: ")) - 1
                if 0 <= choice < len(matches):
                    selected_contact = matches[choice]
                else:
                    print("无效选择")
                    return
            except ValueError:
                print("请输入有效数字")
                return
        else:
            selected_contact = matches[0]
        
        print(f"\n当前联系人信息:")
        print(f"ID: {selected_contact['contact_id']}")
        print(f"昵称: {selected_contact.get('nickname', '无')}")
        print(f"备注: {selected_contact.get('remark', '无')}")
        print(f"当前关系: {selected_contact.get('relationship', '朋友')}")
        print(f"详细备注: {selected_contact.get('relationship_detail', '无')}")
        
        print("\n请选择关系类型:")
        relationships = ["朋友", "同事", "同学", "家人", "恋人", "师长", "学生", "客户", "陌生人", "其他"]
        for i, rel in enumerate(relationships, 1):
            print(f"{i}. {rel}")
        
        try:
            rel_choice = int(input("请选择关系类型序号 (直接回车保持不变): ").strip())
            if 1 <= rel_choice <= len(relationships):
                new_relationship = relationships[rel_choice - 1]
            else:
                print("无效选择，保持原关系")
                new_relationship = selected_contact.get('relationship', '朋友')
        except ValueError:
            new_relationship = selected_contact.get('relationship', '朋友')
        
        new_detail = input("请输入详细关系备注 (直接回车保持不变): ").strip()
        if not new_detail:
            new_detail = selected_contact.get('relationship_detail', '')
        
        # 更新关系
        self.contact_manager.update_relationship(
            selected_contact['contact_id'], 
            new_relationship, 
            new_detail
        )
        
        print("联系人关系更新成功！")
    
    def export_contacts(self) -> None:
        """导出联系人列表"""
        contacts = self.contact_manager.list_all_contacts()
        if not contacts:
            print("暂无联系人记录")
            return
        
        # 按消息数量排序
        contacts.sort(key=lambda x: x['message_count'], reverse=True)
        
        export_file = os.path.join(self.output_dir, f"contacts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        contacts_summary = self.contact_manager.get_contacts_summary()
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "summary": contacts_summary,
            "contacts": contacts
        }
        
        try:
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"联系人列表已导出到: {export_file}")
            print(f"总计 {contacts_summary['total_contacts']} 个联系人，{contacts_summary['total_messages']} 条消息")
        except Exception as e:
            print(f"导出失败: {e}")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'stage1':
            # 阶段1: 数据提取和联系人信息建立
            print("执行阶段1: 数据提取和联系人信息建立")
            processor = WeChatDataProcessor()
            processor.process_stage1_extract_and_contacts()
            return
            
        elif command == 'stage2':
            # 阶段2: 大模型评估和最终数据集生成
            print("执行阶段2: 大模型评估和最终数据集生成")
            processor = WeChatDataProcessor()
            processor.process_stage2_evaluation_and_final()
            return
            
        elif command == 'contacts':
            # 联系人管理模式
            processor = WeChatDataProcessor()
            processor.manage_contacts_interactive()
            return
            
        elif command == 'filter':
            # 从评估缓存筛选数据
            processor = WeChatDataProcessor()
            processor.filter_from_cache()
            return
            
        elif command == 'help':
            print("微信聊天记录处理工具")
            print("用法:")
            print("  python process_wechat_data.py           # 完整处理流程（兼容模式）")
            print("  python process_wechat_data.py stage1    # 阶段1: 数据提取和联系人信息")
            print("  python process_wechat_data.py stage2    # 阶段2: 大模型评估和最终数据集")
            print("  python process_wechat_data.py contacts  # 管理联系人关系")
            print("  python process_wechat_data.py filter    # 从评估缓存筛选数据")
            print("  python process_wechat_data.py help      # 显示帮助")
            print("\n推荐流程:")
            print("  1. 运行 stage1 提取数据和建立联系人信息")
            print("  2. 使用 contacts 管理和编辑联系人关系")
            print("  3. 运行 stage2 进行质量评估和生成最终数据集")
            print("  4. 使用 filter 从已有评估缓存中重新筛选数据")
            return
    
    # 默认处理模式（兼容模式）
    processor = WeChatDataProcessor()
    processor.process()

if __name__ == "__main__":
    main() 