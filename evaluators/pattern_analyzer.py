#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SoulByte - 数字人生成系列项目 · 回复模式分析组件
SoulByte - Digital Human Generation Series · Reply Pattern Analysis Component

此模块负责分析用户的聊天记录，总结用户的回复逻辑、语气、语调和表达风格，
通过分段分析和多层次合并的方式，生成全面的用户回复策略概述。
"""

import json
import os
import time
import hashlib
from typing import Dict, List, Any, Tuple
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from managers.config_manager import ConfigManager
from managers.contact_manager import ContactManager
import pandas as pd


class PatternAnalysisCache:
    """回复模式分析缓存管理器"""
    
    def __init__(self, output_dir: str):
        self.cache_dir = os.path.join(output_dir, "pattern_analysis_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "analysis_cache.json")
        self.api_log_file = os.path.join(self.cache_dir, "api_interactions.json")
        self.cache_data = self._load_cache()
        self.api_interactions = self._load_api_log()
    
    def _load_cache(self) -> Dict:
        """加载缓存数据"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载分析缓存失败: {e}")
        return {}
    
    def _load_api_log(self) -> List:
        """加载API交互日志"""
        if os.path.exists(self.api_log_file):
            try:
                with open(self.api_log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载API日志失败: {e}")
        return []
    
    def _save_cache(self):
        """保存缓存数据"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存分析缓存失败: {e}")
    
    def _save_api_log(self):
        """保存API交互日志"""
        try:
            with open(self.api_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.api_interactions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存API日志失败: {e}")
    
    def get_cache_key(self, data: Any) -> str:
        """生成缓存键"""
        data_str = json.dumps(data, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Any:
        """获取缓存结果"""
        return self.cache_data.get(cache_key)
    
    def save_result(self, cache_key: str, result: Any):
        """保存分析结果到缓存"""
        self.cache_data[cache_key] = {
            "result": result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cache_key": cache_key
        }
        self._save_cache()
    
    def log_api_interaction(self, prompt: str, response: str, analysis_type: str, 
                           contact_name: str = "", chunk_idx: int = -1):
        """记录API交互"""
        interaction = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_type": analysis_type,
            "contact_name": contact_name,
            "chunk_idx": chunk_idx,
            "prompt": prompt,
            "response": response,
            "prompt_length": len(prompt),
            "response_length": len(response)
        }
        self.api_interactions.append(interaction)
        self._save_api_log()
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            "total_cached_results": len(self.cache_data),
            "total_api_interactions": len(self.api_interactions),
            "cache_file": self.cache_file,
            "api_log_file": self.api_log_file
        }


class PatternAnalyzer:
    """回复模式分析器 - 使用大模型分析用户的回复模式和风格"""
    
    def __init__(self, config: ConfigManager, output_dir: str = "output"):
        """初始化回复模式分析器"""
        self.config = config
        self.output_dir = output_dir
        
        # API配置
        self.api_url = config.get('pattern_analysis.api_url')
        self.api_key = config.get('pattern_analysis.api_key')
        self.model = config.get('pattern_analysis.model')
        self.timeout = config.get('pattern_analysis.timeout', 60)
        self.retry_attempts = config.get('pattern_analysis.retry_attempts', 3)
        
        # 分析配置
        self.chunk_size = config.get('pattern_analysis.chunk_size', 1000)
        self.max_workers = config.get('pattern_analysis.max_workers', 3)
        self.debug_mode = config.get('pattern_analysis.debug_mode', False)
        
        # 提示词模板
        self.first_level_prompt = config.get('pattern_analysis.first_level_prompt')
        self.second_level_prompt = config.get('pattern_analysis.second_level_prompt')
        self.final_level_prompt = config.get('pattern_analysis.final_level_prompt')
        
        # 联系人管理器和缓存管理器
        self.contact_manager = ContactManager(output_dir)
        self.cache = PatternAnalysisCache(output_dir)
        
        # 获取用户自己的微信ID
        self.my_wxid = config.get('data_processing.my_wxid')
    
    def analyze_chat_patterns(self, messages: List[Dict]) -> Dict:
        """分析聊天记录中的回复模式
        
        Args:
            messages: 所有聊天记录
        
        Returns:
            包含回复模式分析结果的字典
        """
        print("\n🧠 === 开始分析回复模式 ===")
        
        # 显示缓存统计信息
        cache_stats = self.cache.get_cache_stats()
        print(f"📊 缓存统计: {cache_stats['total_cached_results']} 个缓存结果, {cache_stats['total_api_interactions']} 次API调用")
        
        # 1. 按联系人分组消息
        print("👥 按联系人分组消息...")
        contact_messages = self._group_messages_by_contact(messages)
        
        # 2. 第一层分析：按联系人和块分析
        print("🔍 第一层分析：按联系人和块分析...")
        first_level_results = self._perform_first_level_analysis(contact_messages)
        
        # 3. 第二层分析：合并每个联系人的结果
        print("🔄 第二层分析：合并每个联系人的结果...")
        second_level_results = self._perform_second_level_analysis(first_level_results)
        
        # 4. 最终分析：整合所有联系人的结果
        print("✨ 最终分析：整合所有联系人的结果...")
        final_result = self._perform_final_analysis(second_level_results)
        
        # 5. 保存结果
        self._save_analysis_results(final_result, first_level_results, second_level_results)
        
        # 显示最终缓存统计
        final_cache_stats = self.cache.get_cache_stats()
        print(f"📊 最终缓存统计: {final_cache_stats['total_cached_results']} 个缓存结果, {final_cache_stats['total_api_interactions']} 次API调用")
        
        return final_result
    
    def _group_messages_by_contact(self, messages: List[Dict]) -> Dict[str, List[Dict]]:
        """按联系人分组消息，包含双向对话"""
        print("👥 按联系人分组消息...")
        
        # 统计每个联系人的消息数量，找出用户自己的ID
        talker_counts = {}
        for msg in messages:
            talker = msg['talker']
            talker_counts[talker] = talker_counts.get(talker, 0) + 1
        
        # 消息数量最多的通常是用户自己
        if talker_counts:
            my_contact_id = max(talker_counts.items(), key=lambda x: x[1])[0]
            print(f"🔍 识别用户自己的联系人ID: {my_contact_id} (消息数: {talker_counts[my_contact_id]})")
        
        # 优化：按时间排序所有消息，并建立时间索引
        print("⏰ 按时间排序消息并建立索引...")
        sorted_messages = sorted(messages, key=lambda x: pd.to_datetime(x['create_time']) if isinstance(x['create_time'], str) else x['create_time'])
        
        # 建立联系人消息的时间索引（除了用户自己）
        contact_time_index = {}
        user_messages = []
        
        for i, msg in enumerate(sorted_messages):
            msg_time = pd.to_datetime(msg['create_time']) if isinstance(msg['create_time'], str) else msg['create_time']
            talker = msg['talker']
            
            if talker == my_contact_id:
                user_messages.append((i, msg, msg_time))
            else:
                if talker not in contact_time_index:
                    contact_time_index[talker] = []
                contact_time_index[talker].append((i, msg, msg_time))
        
        print(f"📊 用户消息: {len(user_messages)} 条")
        print(f"👥 其他联系人: {len(contact_time_index)} 个")
        
        # 优化的对话分组算法
        contact_conversations = {}
        time_window_seconds = 3600  # 1小时窗口
        
        print("🔍 开始智能对话分组...")
        for contact_id, contact_msgs in contact_time_index.items():
            if len(contact_msgs) < 5:  # 跳过消息太少的联系人
                continue
                
            conversation = []
            contact_name = self.contact_manager.get_display_name(contact_id)
            
            # 添加该联系人的所有消息
            for _, msg, _ in contact_msgs:
                conversation.append(msg)
            
            # 使用二分搜索找到相关的用户消息
            contact_times = [msg_time for _, _, msg_time in contact_msgs]
            min_contact_time = min(contact_times)
            max_contact_time = max(contact_times)
            
            # 扩展时间窗口
            search_start = min_contact_time - pd.Timedelta(seconds=time_window_seconds)
            search_end = max_contact_time + pd.Timedelta(seconds=time_window_seconds)
            
            # 找到时间窗口内的用户消息
            related_user_msgs = []
            for _, user_msg, user_time in user_messages:
                if search_start <= user_time <= search_end:
                    # 检查是否在任何联系人消息的时间窗口内
                    for _, _, contact_time in contact_msgs:
                        if abs((user_time - contact_time).total_seconds()) <= time_window_seconds:
                            related_user_msgs.append(user_msg)
                            break
            
            # 合并并按时间排序
            conversation.extend(related_user_msgs)
            conversation.sort(key=lambda x: pd.to_datetime(x['create_time']) if isinstance(x['create_time'], str) else x['create_time'])
            
            if len(conversation) >= 20:  # 只保留有足够对话的联系人
                contact_conversations[contact_id] = conversation
                user_msg_count = sum(1 for msg in conversation if msg['talker'] == my_contact_id)
                other_msg_count = len(conversation) - user_msg_count
                print(f"👥 联系人 '{contact_name}': 总计 {len(conversation)} 条消息 (用户: {user_msg_count}, 对方: {other_msg_count})")
        
        print(f"✅ 找到 {len(contact_conversations)} 个有效联系人（消息数 >= 20，包含双向对话）")
        return contact_conversations
    
    def _perform_first_level_analysis(self, contact_messages: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """第一层分析：按联系人和块分析"""
        first_level_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for contact_id, messages in contact_messages.items():
                contact_name = self.contact_manager.get_display_name(contact_id)
                relationship = self.contact_manager.get_relationship(contact_id)
                
                # 按块分割消息
                message_chunks = [
                    messages[i:i+self.chunk_size] 
                    for i in range(0, len(messages), self.chunk_size)
                ]
                
                print(f"📊 联系人 '{contact_name}' ({relationship}) 共有 {len(messages)} 条消息，分为 {len(message_chunks)} 个块")
                
                for chunk_idx, chunk in enumerate(message_chunks):
                    futures.append(
                        executor.submit(
                            self._analyze_message_chunk, 
                            contact_id, 
                            contact_name,
                            relationship,
                            chunk, 
                            chunk_idx
                        )
                    )
            
            # 收集结果
            for future in tqdm(futures, desc="分析消息块"):
                result = future.result()
                if result:
                    contact_id = result["contact_id"]
                    if contact_id not in first_level_results:
                        first_level_results[contact_id] = []
                    first_level_results[contact_id].append(result)
        
        return first_level_results
    
    def _analyze_message_chunk(self, contact_id: str, contact_name: str, 
                              relationship: str, messages: List[Dict], 
                              chunk_idx: int) -> Dict:
        """分析单个消息块"""
        # 生成缓存键
        cache_data = {
            "contact_id": contact_id,
            "contact_name": contact_name,
            "relationship": relationship,
            "chunk_idx": chunk_idx,
            "messages": [{"is_sender": msg['is_sender'], "content": msg['content']} for msg in messages]
        }
        cache_key = self.cache.get_cache_key(cache_data)
        
        # 检查缓存
        cached_result = self.cache.get_cached_result(cache_key)
        if cached_result:
            if self.debug_mode:
                print(f"💾 使用缓存结果: {contact_name} 块 {chunk_idx}")
            return cached_result["result"]
        
        # 提取对话
        conversations = []
        for msg in messages:
            sender = "我" if msg['is_sender'] == 1 else contact_name
            conversations.append({
                "sender": sender,
                "content": msg['content'],
                "create_time": msg['create_time']
            })
        
        # 构建提示词
        prompt = self.first_level_prompt.format(
            contact_name=contact_name,
            relationship=relationship,
            conversation_count=len(messages),
            conversations=json.dumps(conversations, ensure_ascii=False)
        )
        
        # 调用API
        try:
            if self.debug_mode:
                print(f"🔍 分析消息块: {contact_name} 块 {chunk_idx}")
                print(f"📝 提示词长度: {len(prompt)} 字符")
            
            analysis = self._call_api(prompt)
            
            # 记录API交互
            self.cache.log_api_interaction(
                prompt=prompt,
                response=analysis,
                analysis_type="first_level",
                contact_name=contact_name,
                chunk_idx=chunk_idx
            )
            
            if self.debug_mode:
                print(f"✅ 分析完成: {contact_name} 块 {chunk_idx}")
                print(f"📄 回复长度: {len(analysis)} 字符")
                print(f"🔍 回复内容预览: {analysis[:200]}...")
            
            result = {
                "contact_id": contact_id,
                "contact_name": contact_name,
                "relationship": relationship,
                "chunk_idx": chunk_idx,
                "message_count": len(messages),
                "analysis": analysis
            }
            
            # 保存到缓存
            self.cache.save_result(cache_key, result)
            
            return result
        except Exception as e:
            print(f"❌ 分析块失败 (联系人: {contact_name}, 块: {chunk_idx}): {e}")
            return None
    
    def _perform_second_level_analysis(self, first_level_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """第二层分析：合并每个联系人的结果"""
        second_level_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for contact_id, analyses in first_level_results.items():
                if not analyses:
                    continue
                
                contact_name = analyses[0]["contact_name"]
                relationship = analyses[0]["relationship"]
                
                futures.append(
                    executor.submit(
                        self._merge_contact_analyses,
                        contact_id,
                        contact_name,
                        relationship,
                        analyses
                    )
                )
            
            # 收集结果
            for future in tqdm(futures, desc="合并联系人分析"):
                result = future.result()
                if result:
                    second_level_results[result["contact_id"]] = result
        
        return second_level_results
    
    def _merge_contact_analyses(self, contact_id: str, contact_name: str, 
                               relationship: str, analyses: List[Dict]) -> Dict:
        """合并单个联系人的所有分析结果"""
        # 生成缓存键
        cache_data = {
            "contact_id": contact_id,
            "contact_name": contact_name,
            "relationship": relationship,
            "analyses": [analysis["analysis"] for analysis in analyses]
        }
        cache_key = self.cache.get_cache_key(cache_data)
        
        # 检查缓存
        cached_result = self.cache.get_cached_result(cache_key)
        if cached_result:
            if self.debug_mode:
                print(f"💾 使用缓存结果: {contact_name} 二级分析")
            return cached_result["result"]
        
        # 提取所有分析结果
        all_analyses = [analysis["analysis"] for analysis in analyses]
        total_messages = sum(analysis["message_count"] for analysis in analyses)
        
        # 构建提示词
        prompt = self.second_level_prompt.format(
            contact_name=contact_name,
            relationship=relationship,
            analysis_count=len(analyses),
            total_messages=total_messages,
            analyses=json.dumps(all_analyses, ensure_ascii=False)
        )
        
        # 调用API
        try:
            if self.debug_mode:
                print(f"🔄 合并联系人分析: {contact_name}")
                print(f"📝 提示词长度: {len(prompt)} 字符")
            
            merged_analysis = self._call_api(prompt)
            
            # 记录API交互
            self.cache.log_api_interaction(
                prompt=prompt,
                response=merged_analysis,
                analysis_type="second_level",
                contact_name=contact_name
            )
            
            if self.debug_mode:
                print(f"✅ 合并完成: {contact_name}")
                print(f"📄 回复长度: {len(merged_analysis)} 字符")
                print(f"🔍 回复内容预览: {merged_analysis[:200]}...")
            
            result = {
                "contact_id": contact_id,
                "contact_name": contact_name,
                "relationship": relationship,
                "analysis_count": len(analyses),
                "total_messages": total_messages,
                "merged_analysis": merged_analysis
            }
            
            # 保存到缓存
            self.cache.save_result(cache_key, result)
            
            return result
        except Exception as e:
            print(f"❌ 合并联系人分析失败 (联系人: {contact_name}): {e}")
            return None
    
    def _perform_final_analysis(self, second_level_results: Dict[str, Dict]) -> Dict:
        """最终分析：整合所有联系人的结果"""
        # 生成缓存键
        contact_analyses = []
        for contact_id, result in second_level_results.items():
            contact_analyses.append({
                "contact_name": result["contact_name"],
                "relationship": result["relationship"],
                "total_messages": result["total_messages"],
                "analysis": result["merged_analysis"]
            })
        
        cache_key = self.cache.get_cache_key(contact_analyses)
        
        # 检查缓存
        cached_result = self.cache.get_cached_result(cache_key)
        if cached_result:
            if self.debug_mode:
                print("💾 使用缓存结果: 最终分析")
            return cached_result["result"]
        
        total_contacts = len(second_level_results)
        total_messages = sum(result["total_messages"] for result in second_level_results.values())
        
        # 构建提示词
        prompt = self.final_level_prompt.format(
            contact_count=total_contacts,
            total_messages=total_messages,
            contact_analyses=json.dumps(contact_analyses, ensure_ascii=False)
        )
        
        # 调用API
        try:
            if self.debug_mode:
                print("✨ 执行最终分析")
                print(f"📝 提示词长度: {len(prompt)} 字符")
            
            final_analysis = self._call_api(prompt)
            
            # 记录API交互
            self.cache.log_api_interaction(
                prompt=prompt,
                response=final_analysis,
                analysis_type="final_level"
            )
            
            if self.debug_mode:
                print("✅ 最终分析完成")
                print(f"📄 回复长度: {len(final_analysis)} 字符")
                print(f"🔍 回复内容预览: {final_analysis[:500]}...")
            
            result = {
                "total_contacts": total_contacts,
                "total_messages": total_messages,
                "final_analysis": final_analysis,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 保存到缓存
            self.cache.save_result(cache_key, result)
            
            return result
        except Exception as e:
            print(f"❌ 最终分析失败: {e}")
            return {
                "total_contacts": total_contacts,
                "total_messages": total_messages,
                "final_analysis": f"分析失败: {e}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
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
            'temperature': 0.7,
            'top_p': 0.95,
        }
        
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                result = response.json()
                
                return result['choices'][0]['message']['content'].strip()
            except Exception as e:
                if attempt == self.retry_attempts - 1:
                    raise e
                time.sleep(2)  # 重试前等待
        
        return "API调用失败"
    
    def _save_analysis_results(self, final_result: Dict, 
                              first_level_results: Dict[str, List[Dict]],
                              second_level_results: Dict[str, Dict]) -> None:
        """保存分析结果"""
        # 创建结果目录
        analysis_dir = os.path.join(self.output_dir, "pattern_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # 保存最终结果
        final_file = os.path.join(analysis_dir, "final_analysis.json")
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        # 保存第一层结果
        first_level_file = os.path.join(analysis_dir, "first_level_analysis.json")
        with open(first_level_file, 'w', encoding='utf-8') as f:
            json.dump(first_level_results, f, ensure_ascii=False, indent=2)
        
        # 保存第二层结果
        second_level_file = os.path.join(analysis_dir, "second_level_analysis.json")
        with open(second_level_file, 'w', encoding='utf-8') as f:
            json.dump(second_level_results, f, ensure_ascii=False, indent=2)
        
        # 保存缓存统计信息
        cache_stats_file = os.path.join(analysis_dir, "cache_statistics.json")
        cache_stats = self.cache.get_cache_stats()
        with open(cache_stats_file, 'w', encoding='utf-8') as f:
            json.dump(cache_stats, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 分析结果已保存到: {analysis_dir}")
        print(f"📊 最终分析: {final_file}")
        print(f"💾 缓存统计: {cache_stats_file}")
        print(f"📝 API交互日志: {cache_stats['api_log_file']}") 