#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm


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
    
    def get_display_name(self, contact_id: str) -> str:
        """获取联系人显示名称"""
        contact_info = self.load_contact(contact_id)
        
        # 优先使用备注，其次昵称，最后使用ID
        if contact_info.get('remark') and contact_info['remark'].strip():
            return contact_info['remark']
        elif contact_info.get('nickname') and contact_info['nickname'].strip():
            return contact_info['nickname']
        
        return contact_id[-8:] if len(contact_id) > 8 else contact_id
    
    def search_contacts(self, keyword: str) -> List[Dict]:
        """搜索联系人"""
        keyword_lower = keyword.lower()
        matches = []
        
        for contact in self.list_all_contacts():
            if (keyword_lower in contact.get('nickname', '').lower() or
                keyword_lower in contact.get('remark', '').lower() or
                keyword_lower in contact['contact_id'].lower()):
                matches.append(contact)
        
        return matches
    
    def get_relationship(self, contact_id: str) -> str:
        """获取联系人关系类型"""
        contact_info = self.load_contact(contact_id)
        return contact_info.get('relationship', '朋友') 