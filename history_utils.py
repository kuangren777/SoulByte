#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
from typing import Dict, List
from contact_manager import ContactManager


class HistoryManager:
    """历史记录管理器"""
    
    def __init__(self, contact_manager: ContactManager):
        self.contact_manager = contact_manager
    
    def build_context(self, messages: List[Dict], reply_time: datetime, 
                     current_talker_id: str) -> List[Dict]:
        """
        构建历史上下文（前72小时内的消息）
        只包含与当前对话联系人相关的消息
        """
        # 计算72小时前的时间点
        time_72h_ago = reply_time - timedelta(hours=72)
        context_messages = []
        
        for msg in messages:
            msg_time = msg['create_time']
            msg_talker = msg['talker']
            
            # 检查时间范围：在72小时内且在回复时间之前
            if time_72h_ago <= msg_time < reply_time:
                # 只包含与当前联系人相关的消息（对方发的或我发给对方的）
                if (msg['is_sender'] == 0 and msg_talker == current_talker_id) or \
                   (msg['is_sender'] == 1):  # 我发的所有消息（可能是回复给这个人的）
                    context_messages.append(msg)
        
        return context_messages
    
    def build_history_text(self, context_messages: List[Dict], 
                          current_other_block: List[Dict], 
                          my_name: str = "我") -> str:
        """
        构建历史对话文本，包含日期和时间信息
        当日期发生变化时会显示日期分隔线
        """
        history_lines = []
        last_date = None
        
        # 合并历史消息和当前对方消息块
        all_messages = context_messages + current_other_block
        all_messages.sort(key=lambda x: x['create_time'])
        
        for msg in all_messages:
            msg_time = msg['create_time']
            msg_date = msg_time.date()
            
            # 如果日期发生变化，添加日期分隔线
            if last_date is None or msg_date != last_date:
                date_str = msg_date.strftime("%Y年%m月%d日")
                history_lines.append(f"————— {date_str} —————")
                last_date = msg_date
            
            # 格式化时间和发送者
            time_str = msg_time.strftime("%H:%M:%S")
            
            if msg['is_sender'] == 1:
                sender = my_name
            else:
                sender = self.contact_manager.get_display_name(msg['talker'])
            
            # 添加消息内容
            history_lines.append(f"[{time_str}] {sender}: {msg['content']}")
        
        # 返回最近的20条记录（包括日期分隔线）
        return "\n".join(history_lines[-30:])  # 增加到30条以包含更多上下文
    
    def format_message_block_content(self, message_block: List[Dict]) -> str:
        """格式化消息块内容，使用<return>分割连续消息"""
        contents = [msg['content'] for msg in message_block]
        return "<return>".join(contents)
    
    def is_same_conversation_context(self, msg1: Dict, msg2: Dict, 
                                   max_time_gap_hours: float = 24) -> bool:
        """
        判断两条消息是否属于同一个对话上下文
        考虑时间间隔和对话连续性
        """
        time_gap = abs((msg2['create_time'] - msg1['create_time']).total_seconds() / 3600)
        
        # 如果时间间隔超过设定值，认为不是同一个对话上下文
        if time_gap > max_time_gap_hours:
            return False
        
        # 如果是同一个联系人，认为是同一个对话上下文
        if msg1['talker'] == msg2['talker']:
            return True
        
        # 如果其中一个是我发的消息，也认为可能是同一个对话上下文
        if msg1['is_sender'] == 1 or msg2['is_sender'] == 1:
            return True
        
        return False
    
    def get_conversation_summary(self, messages: List[Dict], 
                               reply_time: datetime, 
                               current_talker_id: str) -> Dict:
        """获取对话摘要信息"""
        context_messages = self.build_context(messages, reply_time, current_talker_id)
        
        # 统计信息
        my_messages = [msg for msg in context_messages if msg['is_sender'] == 1]
        other_messages = [msg for msg in context_messages if msg['is_sender'] == 0]
        
        # 时间范围
        if context_messages:
            earliest_time = min(msg['create_time'] for msg in context_messages)
            latest_time = max(msg['create_time'] for msg in context_messages)
            time_span_hours = (latest_time - earliest_time).total_seconds() / 3600
        else:
            time_span_hours = 0
            earliest_time = latest_time = reply_time
        
        return {
            "total_messages": len(context_messages),
            "my_messages_count": len(my_messages),
            "other_messages_count": len(other_messages),
            "time_span_hours": round(time_span_hours, 2),
            "earliest_time": earliest_time.isoformat() if context_messages else None,
            "latest_time": latest_time.isoformat() if context_messages else None,
            "current_talker": self.contact_manager.get_display_name(current_talker_id)
        }


# 为了兼容多进程处理，提供一些独立的函数
def build_context_for_process(messages: List[Dict], reply_time: datetime, 
                             current_talker_id: str) -> List[Dict]:
    """多进程版本的历史上下文构建"""
    time_72h_ago = reply_time - timedelta(hours=72)
    context_messages = []
    
    for msg in messages:
        msg_time = msg['create_time']
        msg_talker = msg['talker']
        
        if time_72h_ago <= msg_time < reply_time:
            if (msg['is_sender'] == 0 and msg_talker == current_talker_id) or \
               (msg['is_sender'] == 1):
                context_messages.append(msg)
    
    return context_messages


def build_history_text_for_process(context_messages: List[Dict], 
                                 current_other_block: List[Dict], 
                                 contact_data: Dict, 
                                 my_name: str = "我") -> str:
    """多进程版本的历史对话文本构建"""
    
    def get_other_name_for_process(msg: Dict, contact_data: Dict) -> str:
        """从预传递的数据中获取对方的昵称"""
        talker = msg['talker']
        contact_info = contact_data.get(talker, {})
        
        if contact_info.get('remark') and contact_info['remark'].strip():
            return contact_info['remark']
        elif contact_info.get('nickname') and contact_info['nickname'].strip():
            return contact_info['nickname']
        
        return talker[-8:] if len(talker) > 8 else talker
    
    history_lines = []
    last_date = None
    
    # 合并历史消息和当前对方消息块
    all_messages = context_messages + current_other_block
    all_messages.sort(key=lambda x: x['create_time'])
    
    for msg in all_messages:
        msg_time = msg['create_time']
        msg_date = msg_time.date()
        
        # 如果日期发生变化，添加日期分隔线
        if last_date is None or msg_date != last_date:
            date_str = msg_date.strftime("%Y年%m月%d日")
            history_lines.append(f"————— {date_str} —————")
            last_date = msg_date
        
        # 格式化时间和发送者
        time_str = msg_time.strftime("%H:%M:%S")
        
        if msg['is_sender'] == 1:
            sender = my_name
        else:
            sender = get_other_name_for_process(msg, contact_data)
        
        # 添加消息内容
        history_lines.append(f"[{time_str}] {sender}: {msg['content']}")
    
    return "\n".join(history_lines[-30:])  # 返回最近30条记录


def format_message_block_content(message_block: List[Dict]) -> str:
    """格式化消息块内容，使用<return>分割连续消息"""
    contents = [msg['content'] for msg in message_block]
    return "<return>".join(contents) 