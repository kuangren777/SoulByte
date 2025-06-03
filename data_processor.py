#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
import os
import glob
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count
from tqdm import tqdm
import jieba

from config_manager import ConfigManager
from contact_manager import ContactManager
from evaluation_cache import EvaluationCache
from llm_evaluator import LLMEvaluator
from history_utils import HistoryManager, build_context_for_process, build_history_text_for_process, format_message_block_content


# 多进程工作函数（必须在全局作用域，避免序列化问题）
def extract_message_blocks_process_worker(args):
    """多进程提取对话回合的工作函数"""
    messages, start_idx, end_idx, process_id, MAX_INTER_MESSAGE_GAP, MAX_REPLY_DELAY, progress_dict = args
    conversation_rounds = []
    i = start_idx
    
    # 进程局部的联系人更新缓存
    local_contact_updates = {}
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
            if processed % 100 == 0:
                progress_dict[process_id] = processed
            continue
        
        i = other_block[-1]['index'] + 1
        processed = i - start_idx
        
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
        if processed % 100 == 0:
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
    
    for idx, (other_block, my_block) in enumerate(conversation_rounds_chunk):
        try:
            reply_time = my_block[0]['create_time']
            current_talker_id = other_block[0]['talker']
            
            # 使用新的历史记录构建方法（前72小时）
            context_messages = build_context_for_process(all_messages, reply_time, current_talker_id)
            
            other_content = format_message_block_content(other_block)
            my_reply = format_message_block_content(my_block)
            history_text = build_history_text_for_process(context_messages, other_block, contact_data, my_name)
            
            # 获取对方的详细联系人信息
            talker_id = other_block[0]['talker']
            contact_info = contact_data.get(talker_id, {})
            
            # 获取对方名称
            if contact_info.get('remark') and contact_info['remark'].strip():
                other_name = contact_info['remark']
            elif contact_info.get('nickname') and contact_info['nickname'].strip():
                other_name = contact_info['nickname']
            else:
                other_name = talker_id[-8:] if len(talker_id) > 8 else talker_id
            
            # 构建详细的联系人信息
            relationship = contact_info.get('relationship', '朋友')
            relationship_detail = contact_info.get('relationship_detail', '')
            first_contact_date = contact_info.get('first_contact_date', '')
            
            # 格式化首次联系时间
            first_contact_str = ""
            if first_contact_date:
                try:
                    if isinstance(first_contact_date, str):
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
        if processed % 10 == 0 or processed == total_to_process:
            progress_dict[process_id] = processed
    
    return training_data


class WeChatDataProcessor:
    """微信聊天数据处理器"""
    
    def __init__(self, config_file: str = "config.json"):
        # 加载配置
        if not os.path.exists(config_file):
            print(f"配置文件 {config_file} 不存在，使用默认配置")
            self.config = None
            self.use_llm_evaluation = False
        else:
            self.config = ConfigManager(config_file)
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
        
        # 设置参数
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
        
        # 初始化管理器
        self.contact_manager = ContactManager(self.output_dir)
        self.history_manager = HistoryManager(self.contact_manager)
        self.evaluation_cache = EvaluationCache(self.output_dir)
        
        # 联系人信息数据库
        self.contact_database = {}
        
        # 增量处理状态文件
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
        
        # 只有在启用评估时才初始化评估器
        if self.use_llm_evaluation and self.config:
            self.llm_evaluator = LLMEvaluator(self.config, self.output_dir)
    
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
        """从所有users.json文件中加载联系人信息"""
        contact_dirs = glob.glob(os.path.join(self.data_dir, "*/"))
        
        for contact_dir in contact_dirs:
            users_file = os.path.join(contact_dir, "users.json")
            if os.path.exists(users_file):
                try:
                    with open(users_file, 'r', encoding='utf-8') as f:
                        users_data = json.load(f)
                        self.contact_database.update(users_data)
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
                    
                    if csv_file in self.processed_files and self.processed_files[csv_file] == file_hash:
                        continue
                    
                    df = pd.read_csv(csv_file)
                    messages = df.to_dict('records')
                    all_messages.extend(messages)
                    
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
        """数据清洗与预处理"""
        cleaned_messages = []
        contact_updates = {}
        
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
                
                # 缓存联系人信息更新
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
                contact_info['message_count'] = updates['message_count']
                
                first_date_str = updates['first_contact_date'].isoformat()
                if (not contact_info.get('first_contact_date') or 
                    first_date_str < contact_info.get('first_contact_date', '')):
                    contact_info['first_contact_date'] = first_date_str
                
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
    
    def get_my_name(self) -> str:
        """获取我的名字"""
        if self.my_wxid and self.my_wxid in self.contact_database:
            contact_info = self.contact_database[self.my_wxid]
            if contact_info.get('remark') and contact_info['remark'].strip():
                return contact_info['remark']
            elif contact_info.get('nickname') and contact_info['nickname'].strip():
                return contact_info['nickname']
        
        return "骆明宇" 