#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import time
from tqdm import tqdm

from .data_processor import WeChatDataProcessor, extract_message_blocks_process_worker, format_training_data_process_worker
from managers.contact_manager import ContactManager
from utils.history_utils import HistoryManager


class WeChatMainProcessor(WeChatDataProcessor):
    """微信聊天数据主处理器 - 继承基础处理器并添加主要流程"""
    
    def extract_message_blocks(self, messages: List[Dict]) -> List[Tuple[List[Dict], List[Dict]]]:
        """提取对话回合：(对方消息块, 我的回复块) 的配对"""
        can_use_multiprocess = len(messages) >= 1000
        
        if not self.use_multiprocessing or not can_use_multiprocess:
            return self._extract_message_blocks_single_process(messages)
        
        # 多进程处理
        chunk_size = max(len(messages) // self.max_workers, 1000)
        chunks = []
        
        manager = Manager()
        progress_dict = manager.dict()
        
        for i in range(0, len(messages), chunk_size):
            end_idx = min(i + chunk_size, len(messages))
            process_id = len(chunks)
            progress_dict[process_id] = 0
            chunks.append((messages, i, end_idx, process_id, self.MAX_INTER_MESSAGE_GAP, self.MAX_REPLY_DELAY, progress_dict))
        
        conversation_rounds = []
        all_contact_updates = {}
        
        print(f"使用多进程处理，分为 {len(chunks)} 个块...")
        print(f"每个进程将处理约 {chunk_size} 条消息")
        
        total_messages = len(messages)
        with tqdm(total=total_messages, desc="多进程提取对话", smoothing=0.1) as pbar:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(extract_message_blocks_process_worker, chunk) for chunk in chunks]
                
                last_total = 0
                while not all(future.done() for future in futures):
                    current_total = sum(progress_dict.values())
                    if current_total > last_total:
                        pbar.update(current_total - last_total)
                        last_total = current_total
                    time.sleep(0.1)
                
                for future in as_completed(futures):
                    try:
                        chunk_rounds, contact_updates = future.result()
                        conversation_rounds.extend(chunk_rounds)
                        
                        for talker, updates in contact_updates.items():
                            if talker not in all_contact_updates:
                                all_contact_updates[talker] = {'message_count': 0, 'last_contact_date': None}
                            all_contact_updates[talker]['message_count'] += updates['message_count']
                            if (all_contact_updates[talker]['last_contact_date'] is None or 
                                updates['last_contact_date'] > all_contact_updates[talker]['last_contact_date']):
                                all_contact_updates[talker]['last_contact_date'] = updates['last_contact_date']
                    except Exception as e:
                        print(f"处理进程块时出错: {e}")
                
                pbar.update(total_messages - pbar.n)
        
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
    
    def find_other_message_block(self, messages: List[Dict], start_idx: int):
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
    
    def find_my_reply_block(self, messages: List[Dict], start_idx: int, last_other_msg: Dict):
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
    
    def format_training_data(self, conversation_rounds: List[Tuple], all_messages: List[Dict]) -> List[Dict]:
        """格式化为训练数据"""
        if len(conversation_rounds) < 100 or not self.use_multiprocessing:
            return self._format_training_data_single_thread(conversation_rounds, all_messages)
        
        # 预先提取所有联系人信息
        print("正在预提取联系人信息...")
        contact_data = {}
        all_talkers = set()
        for other_block, my_block in conversation_rounds:
            for msg in other_block:
                all_talkers.add(msg['talker'])
        
        for talker in tqdm(all_talkers, desc="提取联系人信息"):
            contact_info = self.contact_manager.load_contact(talker)
            contact_data[talker] = contact_info
        
        my_name = self.get_my_name()
        
        # 多进程处理
        chunk_size = max(len(conversation_rounds) // self.max_workers, 10)
        chunks = []
        
        manager = Manager()
        progress_dict = manager.dict()
        
        for i in range(0, len(conversation_rounds), chunk_size):
            end_idx = min(i + chunk_size, len(conversation_rounds))
            chunk_rounds = conversation_rounds[i:end_idx]
            process_id = len(chunks)
            progress_dict[process_id] = 0
            chunks.append((chunk_rounds, all_messages, contact_data, my_name, process_id, progress_dict))
        
        training_data = []
        
        print(f"使用多进程格式化训练数据，分为 {len(chunks)} 个块...")
        
        total_rounds = len(conversation_rounds)
        with tqdm(total=total_rounds, desc="多进程格式化", smoothing=0.1) as pbar:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(format_training_data_process_worker, chunk) for chunk in chunks]
                
                last_total = 0
                while not all(future.done() for future in futures):
                    current_total = sum(progress_dict.values())
                    if current_total > last_total:
                        pbar.update(current_total - last_total)
                        last_total = current_total
                    time.sleep(0.1)
                
                for future in as_completed(futures):
                    try:
                        chunk_training_data = future.result()
                        training_data.extend(chunk_training_data)
                    except Exception as e:
                        print(f"格式化进程块时出错: {e}")
                
                pbar.update(total_rounds - pbar.n)
        
        print(f"多进程格式化完成！生成了 {len(training_data)} 条训练数据")
        return training_data
    
    def _format_training_data_single_thread(self, conversation_rounds: List[Tuple], all_messages: List[Dict]) -> List[Dict]:
        """单线程格式化训练数据"""
        training_data = []
        my_name = self.get_my_name()
        
        print("正在格式化训练数据...")
        for other_block, my_block in tqdm(conversation_rounds, desc="格式化对话"):
            try:
                reply_time = my_block[0]['create_time']
                current_talker_id = other_block[0]['talker']
                
                # 使用新的历史记录构建方法（前72小时）
                context_messages = self.history_manager.build_context(all_messages, reply_time, current_talker_id)
                
                other_content = self.history_manager.format_message_block_content(other_block)
                my_reply = self.history_manager.format_message_block_content(my_block)
                history_text = self.history_manager.build_history_text(context_messages, other_block, my_name)
                
                # 获取对方的详细联系人信息
                talker_id = other_block[0]['talker']
                contact_info = self.contact_manager.load_contact(talker_id)
                other_name = self.contact_manager.get_display_name(talker_id)
                
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
            
            score_ranges = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
            for min_score, max_score in score_ranges:
                range_name = f"{min_score}-{max_score}"
                count = sum(1 for s in scores if min_score <= s < max_score)
                evaluation_results['score_distribution'][range_name] = count
        
        print(f"评估完成！通过 {evaluation_results['passed_samples']}/{evaluation_results['total_samples']} 条数据")
        print(f"平均分数: {evaluation_results['average_score']:.2f}")
        
        return filtered_data, evaluation_results 