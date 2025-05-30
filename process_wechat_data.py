#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
import os
import glob
from datetime import datetime, timedelta
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import jieba

class WeChatDataProcessor:
    def __init__(self, data_dir: str = "data", output_file: str = "training_data.json"):
        self.data_dir = data_dir
        self.output_file = output_file
        self.my_wxid = "wxid_twmzyezhlsj022"  # 你的微信ID
        
        # 算法参数
        self.MAX_INTER_MESSAGE_GAP = 90  # 同一发送者连续消息的最大时间间隔（秒）
        self.MAX_REPLY_DELAY = 300  # 回复的最大延迟（秒）
        
        # 联系人信息数据库
        self.contact_database = {}
        
        # 消息类型映射
        self.message_type_mapping = {
            "语音": "[语音消息]",
            "图片": "[图片]", 
            "文件": "[文件]",
            "动画表情": "[动画表情]",
            "(分享)卡片式链接": "[分享链接]",
            "合并转发的聊天记录": "[合并转发的聊天记录]",
            "引用回复": "引用回复"  # 保留原内容
        }
        
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
                        print(f"从 {users_file} 加载了 {len(users_data)} 个联系人信息")
                except Exception as e:
                    print(f"加载联系人文件 {users_file} 时出错: {e}")
        
        print(f"联系人数据库总共包含 {len(self.contact_database)} 个联系人")
        
    def load_all_csv_files(self) -> List[Dict]:
        """加载data文件夹中所有子目录的CSV文件"""
        all_messages = []
        
        # 遍历所有联系人目录
        contact_dirs = glob.glob(os.path.join(self.data_dir, "*/"))
        
        for contact_dir in contact_dirs:
            print(f"正在处理目录: {contact_dir}")
            csv_files = glob.glob(os.path.join(contact_dir, "*.csv"))
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    # 转换为字典列表
                    messages = df.to_dict('records')
                    all_messages.extend(messages)
                    print(f"从 {csv_file} 加载了 {len(messages)} 条消息")
                except Exception as e:
                    print(f"处理文件 {csv_file} 时出错: {e}")
                    
        return all_messages
    
    def clean_and_preprocess(self, messages: List[Dict]) -> List[Dict]:
        """数据清洗与预处理"""
        cleaned_messages = []
        
        for msg in messages:
            try:
                # 解析时间
                create_time = pd.to_datetime(msg['CreateTime'])
                
                # 处理消息内容
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
                
                cleaned_messages.append(cleaned_msg)
                
            except Exception as e:
                print(f"处理消息ID {msg.get('id', 'unknown')} 时出错: {e}")
                continue
        
        # 按时间排序
        cleaned_messages.sort(key=lambda x: x['create_time'])
        print(f"清洗后共有 {len(cleaned_messages)} 条消息")
        return cleaned_messages
    
    def process_message_content(self, msg: Dict) -> str:
        """处理不同类型的消息内容"""
        type_name = msg['type_name']
        msg_content = msg['msg'] if pd.notna(msg['msg']) else ""
        
        # 文本消息直接使用
        if type_name == "文本":
            return msg_content
        
        # 语音消息检查是否有转文字
        if type_name == "语音":
            if pd.notna(msg.get('src')) and msg['src'].strip():
                return f"[语音转文字: {msg['src']}]"
            else:
                return "[语音消息]"
        
        # 其他类型使用映射
        return self.message_type_mapping.get(type_name, f"[{type_name}]")
    
    def extract_message_blocks(self, messages: List[Dict]) -> List[Tuple[List[Dict], List[Dict]]]:
        """提取对话回合：(对方消息块, 我的回复块) 的配对"""
        conversation_rounds = []
        i = 0
        
        while i < len(messages):
            # 寻找对方的消息块
            other_block = self.find_other_message_block(messages, i)
            if not other_block:
                i += 1
                continue
            
            # 更新索引到对方消息块结束后
            i = other_block[-1]['index'] + 1
            
            # 寻找我的回复块
            my_block = self.find_my_reply_block(messages, i, other_block[-1])
            if not my_block:
                continue
            
            # 检查回复延迟是否合理
            if self.is_valid_reply_timing(other_block[-1], my_block[0]):
                conversation_rounds.append((other_block, my_block))
                print(f"找到对话回合: 对方 {len(other_block)} 条 -> 我 {len(my_block)} 条")
            
            # 更新索引到我的回复块结束后
            i = my_block[-1]['index'] + 1
        
        print(f"总共提取到 {len(conversation_rounds)} 个对话回合")
        return conversation_rounds
    
    def find_other_message_block(self, messages: List[Dict], start_idx: int) -> Optional[List[Dict]]:
        """寻找对方的消息块"""
        if start_idx >= len(messages):
            return None
        
        msg = messages[start_idx]
        # 必须是对方发送的消息
        if msg['is_sender'] == 1:
            return None
        
        block = [{'index': start_idx, **msg}]
        current_talker = msg['talker']
        last_time = msg['create_time']
        
        # 继续寻找同一发送者的连续消息
        for i in range(start_idx + 1, len(messages)):
            next_msg = messages[i]
            
            # 如果是我发送的消息，停止
            if next_msg['is_sender'] == 1:
                break
            
            # 如果不是同一个发送者，停止  
            if next_msg['talker'] != current_talker:
                break
            
            # 检查时间间隔
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
        
        # 寻找第一条我的回复
        first_my_msg = None
        first_my_idx = start_idx
        
        # 在合理范围内寻找我的第一条回复
        for i in range(start_idx, min(start_idx + 10, len(messages))):
            if messages[i]['is_sender'] == 1:
                # 检查回复时间是否合理
                reply_delay = (messages[i]['create_time'] - last_other_msg['create_time']).total_seconds()
                if reply_delay <= self.MAX_REPLY_DELAY:
                    first_my_msg = messages[i]
                    first_my_idx = i
                    break
        
        if not first_my_msg:
            return None
        
        # 构建我的回复块
        block = [{'index': first_my_idx, **first_my_msg}]
        last_time = first_my_msg['create_time']
        
        # 继续寻找我的连续消息
        for i in range(first_my_idx + 1, len(messages)):
            next_msg = messages[i]
            
            # 如果不是我发送的消息，停止
            if next_msg['is_sender'] != 1:
                break
            
            # 检查时间间隔
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
            # 同一天且在回复时间之前
            if msg_date == reply_date and msg['create_time'] < reply_time:
                context_messages.append(msg)
        
        return context_messages
    
    def format_message_block_content(self, message_block: List[Dict]) -> str:
        """格式化消息块内容，使用<return>分割连续消息"""
        contents = [msg['content'] for msg in message_block]
        return "<return>".join(contents)
    
    def format_training_data(self, conversation_rounds: List[Tuple], all_messages: List[Dict]) -> List[Dict]:
        """格式化为训练数据"""
        training_data = []
        
        for other_block, my_block in conversation_rounds:
            try:
                # 获取回复时间点
                reply_time = my_block[0]['create_time']
                
                # 构建历史上下文
                context_messages = self.build_context(all_messages, reply_time)
                
                # 构建对方消息内容（使用<return>分割）
                other_content = self.format_message_block_content(other_block)
                
                # 构建我的回复内容（使用<return>分割）
                my_reply = self.format_message_block_content(my_block)
                
                # 构建历史对话记录
                history_text = self.build_history_text(context_messages, other_block)
                
                # 构建训练样本
                training_sample = {
                    "instruction": f"你是{self.get_my_name()}，正在和{self.get_other_name(other_block[0])}聊天。根据聊天记录和对方的最新消息，用你的风格回复。",
                    "input": f"历史记录:\n{history_text}\n\n对方最新消息:\n{other_content}",
                    "output": my_reply
                }
                
                training_data.append(training_sample)
                
            except Exception as e:
                print(f"格式化训练数据时出错: {e}")
                continue
        
        return training_data
    
    def build_history_text(self, context_messages: List[Dict], current_other_block: List[Dict]) -> str:
        """构建历史对话文本"""
        history_lines = []
        
        for msg in context_messages:
            time_str = msg['create_time'].strftime("%H:%M:%S")
            sender = "我" if msg['is_sender'] == 1 else self.get_other_name(msg)
            history_lines.append(f"[{time_str}] {sender}: {msg['content']}")
        
        # 添加当前对方的消息块到历史记录中
        for msg in current_other_block:
            time_str = msg['create_time'].strftime("%H:%M:%S")
            sender = self.get_other_name(msg)
            history_lines.append(f"[{time_str}] {sender}: {msg['content']}")
        
        return "\n".join(history_lines[-20:])  # 限制历史记录长度
    
    def get_my_name(self) -> str:
        """获取我的名字"""
        if self.my_wxid in self.contact_database:
            contact_info = self.contact_database[self.my_wxid]
            # 优先使用备注，然后昵称
            if contact_info.get('remark') and contact_info['remark'].strip():
                return contact_info['remark']
            elif contact_info.get('nickname') and contact_info['nickname'].strip():
                return contact_info['nickname']
        
        # 如果数据库中没有，使用默认名字
        return "骆明宇"
    
    def get_other_name(self, msg: Dict) -> str:
        """从数据库中获取对方的昵称"""
        talker = msg['talker']
        
        if talker in self.contact_database:
            contact_info = self.contact_database[talker]
            # 优先使用备注，然后昵称
            if contact_info.get('remark') and contact_info['remark'].strip():
                return contact_info['remark']
            elif contact_info.get('nickname') and contact_info['nickname'].strip():
                return contact_info['nickname']
        
        # 如果数据库中没有，返回wxid的后8位
        return talker[-8:] if len(talker) > 8 else talker
    
    def analyze_frequent_words(self, text: str, top_n: int = 50) -> List[Tuple[str, int]]:
        """动态分析常用词汇"""
        # 使用jieba分词
        words = jieba.lcut(text)
        
        # 过滤掉单字符词汇和标点符号
        filtered_words = []
        for word in words:
            word = word.strip()
            if len(word) >= 2 and word.isalpha() or word in ['哈哈', '草', '寄', 'nb', '捏', 'okok', '没事', '笑死', '牛逼']:
                filtered_words.append(word)
        
        # 统计词频
        word_count = Counter(filtered_words)
        
        return word_count.most_common(top_n)
    
    def analyze_language_patterns(self, messages: List[Dict]) -> Dict:
        """分析个人语言模式"""
        my_messages = [msg for msg in messages if msg['is_sender'] == 1]
        
        # 统计高频词汇
        all_text = " ".join([msg['content'] for msg in my_messages])
        
        # 动态分析常用词汇
        frequent_words = self.analyze_frequent_words(all_text, top_n=20)
        
        # 常用口头禅和表情
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
        
        # 查找常用表情
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
        
        # 按总消息数排序
        sorted_contacts = sorted(contact_stats.items(), key=lambda x: x[1]["total_messages"], reverse=True)
        
        return {
            "联系人总数": len(contact_stats),
            "活跃联系人详情": dict(sorted_contacts[:10])  # 取前10个最活跃的联系人
        }
    
    def process(self) -> None:
        """主处理流程"""
        print("开始处理微信聊天数据...")
        
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
        
        # 4. 生成训练数据
        print("\n=== 步骤4: 生成训练数据 ===")
        training_data = self.format_training_data(conversation_rounds, cleaned_messages)
        
        # 5. 分析语言模式
        print("\n=== 步骤5: 分析语言模式 ===")
        language_patterns = self.analyze_language_patterns(cleaned_messages)
        
        # 6. 保存结果
        print("\n=== 步骤6: 保存结果 ===")
        self.save_results(training_data, language_patterns)
        
        print(f"\n处理完成！生成了 {len(training_data)} 条训练数据")
        print(f"结果已保存到: {self.output_file}")
    
    def save_results(self, training_data: List[Dict], language_patterns: Dict) -> None:
        """保存处理结果"""
        results = {
            "training_data": training_data,
            "language_patterns": language_patterns,
            "contact_database": self.contact_database,
            "metadata": {
                "total_samples": len(training_data),
                "processing_time": datetime.now().isoformat(),
                "parameters": {
                    "MAX_INTER_MESSAGE_GAP": self.MAX_INTER_MESSAGE_GAP,
                    "MAX_REPLY_DELAY": self.MAX_REPLY_DELAY
                }
            }
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 同时保存纯训练数据（用于直接微调）
        training_only_file = self.output_file.replace('.json', '_training_only.json')
        with open(training_only_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"训练数据已保存到: {training_only_file}")
        print(f"完整结果已保存到: {self.output_file}")

def main():
    """主函数"""
    processor = WeChatDataProcessor()
    processor.process()

if __name__ == "__main__":
    main() 