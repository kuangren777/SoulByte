#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SoulByte - 数字人生成系列项目 · 数据处理组件
SoulByte - Digital Human Generation Series · Data Processing Component

SoulByte 是数字人生成系列项目的核心数据处理组件，专注于从微信聊天记录中
提取"数字灵魂"，为个性化数字人的训练和知识库构建提供高质量数据基础。

主要功能：
1. 智能聊天数据提取和清洗
2. 联系人关系网络管理
3. 历史对话上下文构建（72小时窗口）
4. 大模型质量评估和筛选
5. 个人知识库数据准备

未来规划：
- 模型微调流程集成
- 可视化分析界面
- RAG知识检索系统
- 个性化Embedding优化
- 完整数字人生成生态

作者: Kuangren777
版本: 2.0 (数据处理基础版)
更新时间: 2025年6月
"""

import sys
import os
import json
from typing import Dict, List
from datetime import datetime
from collections import Counter
import jieba
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# 导入各个模块
from core import WeChatMainProcessor
from managers import ContactManager, EvaluationCache
from utils import HistoryManager
from evaluators import LLMEvaluator
from evaluators.pattern_analyzer import PatternAnalyzer


class WeChatProcessorApp:
    """SoulByte 智能聊天数据处理应用程序"""
    
    def __init__(self):
        """初始化应用程序"""
        self.processor = WeChatMainProcessor()
        print("SoulByte · 数字人生成系列项目 - 数据处理组件")
        print("=" * 60)
        print("🎯 当前阶段: 智能聊天数据处理和训练数据生成")
        print("🚀 项目愿景: 构建完整的数字人生成生态系统")
        print("💡 核心目标: 从聊天数据中提取'数字灵魂'")
        print("=" * 60)
        print("🔧 核心改进:")
        print("  • 历史记录扩展到前72小时")
        print("  • 智能联系人过滤")
        print("  • 日期时间显示优化")
        print("  • 模块化代码结构")
        print("=" * 60)
    
    def analyze_frequent_words(self, text: str, top_n: int = 50) -> List[tuple]:
        """动态分析常用词汇"""
        words = jieba.lcut(text)
        
        filtered_words = []
        for word in words:
            word = word.strip()
            if len(word) >= 2 and (word.isalpha() or word in ['哈哈', '草', '寄', 'nb', '捏', 'okok', '没事', '笑死', '牛逼']):
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
        
        # 分析表情使用
        import re
        emoji_pattern = r'\[([^\]]+)\]'
        emojis = re.findall(emoji_pattern, all_text)
        emoji_count = {}
        for emoji in emojis:
            emoji_count[emoji] = emoji_count.get(emoji, 0) + 1
        
        patterns["常用表情"] = [{"表情": emoji, "出现次数": count} 
                            for emoji, count in sorted(emoji_count.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        return patterns
    
    def analyze_contact_statistics(self, messages: List[Dict]) -> Dict:
        """分析联系人统计信息"""
        contact_stats = {}
        
        for msg in messages:
            talker = msg['talker']
            if talker not in contact_stats:
                contact_stats[talker] = {
                    "name": self.processor.contact_manager.get_display_name(talker),
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
                           cleaned_messages: List[Dict], conversation_rounds: List[tuple]) -> None:
        """保存阶段1的结果"""
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
                    "MAX_INTER_MESSAGE_GAP": self.processor.MAX_INTER_MESSAGE_GAP,
                    "MAX_REPLY_DELAY": self.processor.MAX_REPLY_DELAY,
                    "use_multiprocessing": self.processor.use_multiprocessing
                }
            }
        }
        
        stage1_file = os.path.join(self.processor.output_dir, 'stage1_results.json')
        with open(stage1_file, 'w', encoding='utf-8') as f:
            json.dump(stage1_results, f, ensure_ascii=False, indent=2)
        
        training_only_file = os.path.join(self.processor.output_dir, 'stage1_training_data.json')
        with open(training_only_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 阶段1结果已保存到: {stage1_file}")
        print(f"✅ 训练数据已保存到: {training_only_file}")
    
    def process_stage1_extract_and_contacts(self) -> None:
        """阶段1: 数据提取和联系人信息建立"""
        print("\n🚀 === 阶段1: 数据提取和联系人信息建立 ===")
        print(f"📊 使用多进程: {self.processor.use_multiprocessing}")
        
        # 0. 加载联系人信息数据库
        print("\n📋 === 步骤0: 加载联系人信息数据库 ===")
        self.processor.load_contact_database()
        
        # 1. 加载所有CSV文件
        print("\n📁 === 步骤1: 加载数据文件 ===")
        raw_messages = self.processor.load_all_csv_files()
        
        # 2. 数据清洗与预处理
        print("\n🧹 === 步骤2: 数据清洗与预处理 ===")
        cleaned_messages = self.processor.clean_and_preprocess(raw_messages)
        
        # 3. 提取对话回合
        print("\n💬 === 步骤3: 提取对话回合 ===")
        conversation_rounds = self.processor.extract_message_blocks(cleaned_messages)
        
        # 4. 生成训练数据（不进行评估）
        print("\n🔄 === 步骤4: 生成初始训练数据 ===")
        training_data = self.processor.format_training_data(conversation_rounds, cleaned_messages)
        
        # 5. 分析语言模式
        print("\n📈 === 步骤5: 分析语言模式 ===")
        language_patterns = self.analyze_language_patterns(cleaned_messages)
        
        # 6. 保存阶段1结果
        print("\n💾 === 步骤6: 保存阶段1结果 ===")
        self.save_stage1_results(training_data, language_patterns, cleaned_messages, conversation_rounds)
        
        print(f"\n🎉 阶段1完成！生成了 {len(training_data)} 条初始训练数据")
        print(f"📋 联系人信息已保存到: {self.processor.contact_manager.contacts_file}")
        print("💡 现在您可以编辑联系人关系，然后运行阶段2进行质量评估")
    
    def load_stage1_results(self) -> dict:
        """加载阶段1的结果"""
        stage1_file = os.path.join(self.processor.output_dir, 'stage1_results.json')
        if not os.path.exists(stage1_file):
            return None
        
        try:
            with open(stage1_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 加载阶段1结果失败: {e}")
            return None
    
    def process_stage2_evaluation_and_final(self) -> None:
        """阶段2: 大模型评估和最终数据集生成"""
        print("\n🔍 === 阶段2: 大模型评估和最终数据集生成 ===")
        print(f"🤖 使用大模型评估: {self.processor.use_llm_evaluation}")
        
        # 加载阶段1的结果
        print("\n📥 === 步骤1: 加载阶段1结果 ===")
        stage1_data = self.load_stage1_results()
        if not stage1_data:
            print("❌ 错误: 未找到阶段1的结果，请先运行阶段1")
            return
        
        training_data = stage1_data['training_data']
        language_patterns = stage1_data['language_patterns']
        
        print(f"✅ 加载了 {len(training_data)} 条训练数据")
        
        # 重新加载联系人信息（可能已被编辑）
        print("\n👥 === 步骤2: 重新加载联系人信息 ===")
        self.processor.contact_manager = ContactManager(self.processor.output_dir)
        print(f"✅ 加载了 {len(self.processor.contact_manager.list_all_contacts())} 个联系人")
        
        evaluation_results = None
        final_training_data = training_data
        
        # 大模型评估（如果启用）
        if self.processor.use_llm_evaluation:
            print("\n🤖 === 步骤3: 大模型质量评估 ===")
            final_training_data, evaluation_results = self.processor.evaluate_training_data(training_data)
        else:
            print("\n⏭️ === 步骤3: 跳过大模型评估（已禁用） ===")
        
        # 保存最终结果
        print("\n💾 === 步骤4: 保存最终结果 ===")
        self.save_final_results(final_training_data, language_patterns, evaluation_results)
        
        print(f"\n🎉 阶段2完成！最终生成了 {len(final_training_data)} 条训练数据")
        if evaluation_results:
            print(f"📊 原始数据: {evaluation_results['total_samples']} 条")
            print(f"✅ 通过筛选: {evaluation_results['passed_samples']} 条")
            print(f"⭐ 平均分数: {evaluation_results['average_score']:.2f}")
    
    def save_final_results(self, training_data: List[Dict], language_patterns: Dict, 
                          evaluation_results: Dict = None) -> None:
        """保存最终结果"""
        contacts_summary = self.processor.contact_manager.get_contacts_summary()
        
        results = {
            "training_data": training_data,
            "language_patterns": language_patterns,
            "contacts_summary": contacts_summary,
            "metadata": {
                "stage": 2,
                "total_samples": len(training_data),
                "processing_time": datetime.now().isoformat(),
                "parameters": {
                    "MAX_INTER_MESSAGE_GAP": self.processor.MAX_INTER_MESSAGE_GAP,
                    "MAX_REPLY_DELAY": self.processor.MAX_REPLY_DELAY,
                    "use_multiprocessing": self.processor.use_multiprocessing,
                    "use_llm_evaluation": self.processor.use_llm_evaluation
                }
            }
        }
        
        if evaluation_results:
            results["evaluation_results"] = evaluation_results
        
        with open(self.processor.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存纯训练数据
        training_only_file = os.path.join(self.processor.output_dir, 'training_data_training_only.json')
        with open(training_only_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        # 如果使用了大模型评估，保存筛选后的数据
        if self.processor.use_llm_evaluation and evaluation_results:
            filtered_file = os.path.join(self.processor.output_dir, 'training_data_filtered.json')
            with open(filtered_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            # 保存评估报告
            evaluation_report_file = os.path.join(self.processor.output_dir, 'evaluation_report.json')
            with open(evaluation_report_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 最终训练数据已保存到: {training_only_file}")
        print(f"✅ 完整结果已保存到: {self.processor.output_file}")
        print(f"✅ 联系人信息已保存到: {self.processor.contact_manager.contacts_file}")
        if self.processor.use_llm_evaluation:
            print(f"✅ 筛选后数据已保存到: {filtered_file}")
            print(f"✅ 评估报告已保存到: {evaluation_report_file}")
    
    def filter_from_cache(self) -> None:
        """从评估缓存中筛选训练数据"""
        print("\n🔍 === 从评估缓存筛选训练数据 ===")
        
        # 检查缓存文件是否存在
        if not os.path.exists(self.processor.evaluation_cache.cache_file):
            print(f"❌ 错误: 评估缓存文件 {self.processor.evaluation_cache.cache_file} 不存在")
            return
            
        # 获取配置的最低分数
        min_score = self.processor.min_score
        print(f"📊 使用最低分数阈值: {min_score}")
        
        # 从缓存中筛选数据
        filtered_data = self.processor.evaluation_cache.filter_by_score(min_score)
        
        if not filtered_data:
            print("⚠️ 警告: 没有符合条件的数据")
            return
        
        # 获取联系人统计摘要
        contacts_summary = self.processor.contact_manager.get_contacts_summary()
        
        # 分析筛选后数据的语言模式
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
        filtered_file = os.path.join(self.processor.output_dir, f'training_data_filtered_{min_score}.json')
        with open(filtered_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存纯训练数据
        training_only_file = os.path.join(self.processor.output_dir, f'training_only_filtered_{min_score}.json')
        with open(training_only_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"🎉 筛选完成！从缓存中筛选出 {len(filtered_data)} 条训练数据")
        print(f"✅ 完整结果已保存到: {filtered_file}")
        print(f"✅ 纯训练数据已保存到: {training_only_file}")
    
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
        
        # 分析表情使用
        import re
        emoji_pattern = r'\[([^\]]+)\]'
        emojis = re.findall(emoji_pattern, all_text)
        emoji_count = {}
        for emoji in emojis:
            emoji_count[emoji] = emoji_count.get(emoji, 0) + 1
        
        patterns["常用表情"] = [{"表情": emoji, "出现次数": count} 
                            for emoji, count in sorted(emoji_count.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        return patterns
    
    def manage_contacts_interactive(self) -> None:
        """交互式联系人管理"""
        contact_manager = ContactManager(self.processor.output_dir)
        
        while True:
            print("\n👥 === 联系人关系管理 ===")
            print("1. 查看所有联系人")
            print("2. 搜索联系人")
            print("3. 更新联系人关系")
            print("4. 导出联系人列表")
            print("5. 返回主菜单")
            
            choice = input("请选择操作 (1-5): ").strip()
            
            if choice == '1':
                self.show_all_contacts(contact_manager)
            elif choice == '2':
                self.search_contacts(contact_manager)
            elif choice == '3':
                self.update_contact_relationship(contact_manager)
            elif choice == '4':
                self.export_contacts(contact_manager)
            elif choice == '5':
                break
            else:
                print("❌ 无效选择，请重新输入")
    
    def show_all_contacts(self, contact_manager: ContactManager) -> None:
        """显示所有联系人"""
        contacts = contact_manager.list_all_contacts()
        if not contacts:
            print("📭 暂无联系人记录")
            return
        
        # 按消息数量排序
        contacts.sort(key=lambda x: x['message_count'], reverse=True)
        
        print(f"\n📋 共有 {len(contacts)} 个联系人:")
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
    
    def search_contacts(self, contact_manager: ContactManager) -> None:
        """搜索联系人"""
        keyword = input("🔍 请输入搜索关键词（昵称/备注/ID）: ").strip()
        if not keyword:
            print("❌ 关键词不能为空")
            return
        
        matches = contact_manager.search_contacts(keyword)
        
        if not matches:
            print("📭 未找到匹配的联系人")
            return
        
        print(f"\n✅ 找到 {len(matches)} 个匹配的联系人:")
        for i, contact in enumerate(matches, 1):
            display_name = contact.get('remark') or contact.get('nickname') or contact['contact_id'][-8:]
            relationship = contact.get('relationship', '朋友')
            print(f"{i}. {display_name} (ID: {contact['contact_id'][-8:]}) - {relationship}")
    
    def update_contact_relationship(self, contact_manager: ContactManager) -> None:
        """更新联系人关系"""
        keyword = input("👤 请输入要更新的联系人（昵称/备注/ID）: ").strip()
        if not keyword:
            print("❌ 输入不能为空")
            return
        
        matches = contact_manager.search_contacts(keyword)
        
        if not matches:
            print("📭 未找到匹配的联系人")
            return
        
        if len(matches) > 1:
            print("🔍 找到多个匹配的联系人:")
            for i, contact in enumerate(matches, 1):
                print(f"{i}. {contact.get('nickname', contact['contact_id'])} "
                      f"({contact.get('remark', '无备注')})")
            
            try:
                choice = int(input("请选择序号: ")) - 1
                if 0 <= choice < len(matches):
                    selected_contact = matches[choice]
                else:
                    print("❌ 无效选择")
                    return
            except ValueError:
                print("❌ 请输入有效数字")
                return
        else:
            selected_contact = matches[0]
        
        print(f"\n📋 当前联系人信息:")
        print(f"ID: {selected_contact['contact_id']}")
        print(f"昵称: {selected_contact.get('nickname', '无')}")
        print(f"备注: {selected_contact.get('remark', '无')}")
        print(f"当前关系: {selected_contact.get('relationship', '朋友')}")
        print(f"详细备注: {selected_contact.get('relationship_detail', '无')}")
        
        print("\n👥 请选择关系类型:")
        relationships = ["朋友", "同事", "同学", "家人", "恋人", "师长", "学生", "客户", "陌生人", "其他"]
        for i, rel in enumerate(relationships, 1):
            print(f"{i}. {rel}")
        
        try:
            rel_choice = int(input("请选择关系类型序号 (直接回车保持不变): ").strip())
            if 1 <= rel_choice <= len(relationships):
                new_relationship = relationships[rel_choice - 1]
            else:
                print("⚠️ 无效选择，保持原关系")
                new_relationship = selected_contact.get('relationship', '朋友')
        except ValueError:
            new_relationship = selected_contact.get('relationship', '朋友')
        
        new_detail = input("📝 请输入详细关系备注 (直接回车保持不变): ").strip()
        if not new_detail:
            new_detail = selected_contact.get('relationship_detail', '')
        
        # 更新关系
        contact_manager.update_relationship(
            selected_contact['contact_id'], 
            new_relationship, 
            new_detail
        )
        
        print("✅ 联系人关系更新成功！")
    
    def export_contacts(self, contact_manager: ContactManager) -> None:
        """导出联系人列表"""
        contacts = contact_manager.list_all_contacts()
        if not contacts:
            print("📭 暂无联系人记录")
            return
        
        # 按消息数量排序
        contacts.sort(key=lambda x: x['message_count'], reverse=True)
        
        export_file = os.path.join(self.processor.output_dir, f"contacts_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        contacts_summary = contact_manager.get_contacts_summary()
        
        export_data = {
            "export_time": datetime.now().isoformat(),
            "summary": contacts_summary,
            "contacts": contacts
        }
        
        try:
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 联系人列表已导出到: {export_file}")
            print(f"📊 总计 {contacts_summary['total_contacts']} 个联系人，{contacts_summary['total_messages']} 条消息")
        except Exception as e:
            print(f"❌ 导出失败: {e}")
    
    def process_full(self) -> None:
        """完整处理流程（兼容模式）"""
        print("🚀 开始完整处理流程...")
        print("💡 注意：建议使用分阶段处理模式以获得更好的控制")
        print("  stage1: 数据提取和联系人信息")
        print("  stage2: 大模型评估和最终数据集")
        print(f"📊 使用多进程: {self.processor.use_multiprocessing}")
        print(f"🤖 使用大模型评估: {self.processor.use_llm_evaluation}")
        
        # 执行阶段1
        self.process_stage1_extract_and_contacts()
        
        # 执行阶段2
        print("\n" + "="*60)
        self.process_stage2_evaluation_and_final()

    def list_available_contacts(self, data_dir: str = "data") -> List[str]:
        """列出所有可用的联系人文件夹"""
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"❌ 数据目录不存在: {data_path}")
            return []
        
        contact_folders = []
        for item in data_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # 检查文件夹中是否有CSV文件
                csv_files = list(item.glob("*.csv"))
                if csv_files:
                    contact_folders.append(item.name)
        
        return sorted(contact_folders)

    def load_contact_messages(self, contact_folder: str, data_dir: str = "data") -> List[Dict]:
        """加载指定联系人的所有消息
        
        Args:
            contact_folder: 联系人文件夹名称
            data_dir: 数据目录
            
        Returns:
            消息列表
        """
        contact_path = Path(data_dir) / contact_folder
        if not contact_path.exists():
            print(f"❌ 联系人文件夹不存在: {contact_path}")
            return []
        
        messages = []
        csv_files = list(contact_path.glob("*.csv"))
        
        if not csv_files:
            print(f"⚠️ 联系人文件夹中没有CSV文件: {contact_path}")
            return []
        
        print(f"📁 加载联系人 {contact_folder} 的数据...")
        print(f"   找到 {len(csv_files)} 个CSV文件")
        
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                
                # 标准化列名（不同导出可能有不同的列名）
                column_mapping = {
                    'localId': 'local_id',
                    'talkerId': 'talker', 
                    'type': 'type',
                    'content': 'content',
                    'msg': 'content',  # 新的映射
                    'createTime': 'timestamp',
                    'CreateTime': 'timestamp',  # 新的映射
                    'isSender': 'is_sender',
                    'is_sender': 'is_sender',  # 保持不变
                    'talker': 'talker'
                }
                
                # 重命名列
                df = df.rename(columns=column_mapping)
                
                # 确保必要的列存在
                required_cols = ['talker', 'content', 'timestamp', 'is_sender']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"⚠️ 文件 {csv_file.name} 缺少必要列: {missing_cols}")
                    continue
                
                # 过滤文本消息
                if 'type' in df.columns:
                    text_df = df[df['type'] == 1]
                elif 'type_name' in df.columns:
                    text_df = df[df['type_name'] == '文本']
                else:
                    text_df = df
                
                # 转换为字典列表
                for _, row in text_df.iterrows():
                    timestamp = pd.to_datetime(row['timestamp'])
                    message = {
                        'talker': str(row['talker']),
                        'content': str(row['content']),
                        'timestamp': timestamp,
                        'create_time': timestamp.isoformat(),  # 转换为字符串格式
                        'is_sender': int(row['is_sender']),
                        'contact_folder': contact_folder
                    }
                    messages.append(message)
                    
                print(f"   ✅ {csv_file.name}: {len(text_df)} 条文本消息")
                
            except Exception as e:
                print(f"❌ 加载文件失败 {csv_file.name}: {e}")
                continue
        
        # 按时间排序
        messages.sort(key=lambda x: x['timestamp'])
        
        print(f"📊 联系人 {contact_folder} 总计: {len(messages)} 条消息")
        return messages

    def load_all_messages_for_analysis(self, specific_contact: str = None, data_dir: str = "data") -> List[Dict]:
        """加载所有消息或特定联系人的消息进行分析
        
        Args:
            specific_contact: 特定联系人文件夹名称，None表示加载所有
            data_dir: 数据目录
            
        Returns:
            所有消息列表
        """
        all_messages = []
        
        if specific_contact:
            contact_folders = [specific_contact] if specific_contact in self.list_available_contacts(data_dir) else []
            if not contact_folders:
                print(f"❌ 未找到联系人: {specific_contact}")
                return []
        else:
            contact_folders = self.list_available_contacts(data_dir)
        
        if not contact_folders:
            print("❌ 没有找到任何联系人文件夹")
            return []
        
        print(f"📂 准备分析 {len(contact_folders)} 个联系人的数据")
        
        for contact_folder in contact_folders:
            messages = self.load_contact_messages(contact_folder, data_dir)
            all_messages.extend(messages)
        
        # 按时间排序所有消息
        all_messages.sort(key=lambda x: x['timestamp'])
        
        print(f"📊 总计加载: {len(all_messages)} 条消息")
        return all_messages

    def analyze_messages_patterns(self, messages: List[Dict]) -> Dict:
        """分析消息并生成回复模式报告
        
        Args:
            messages: 消息列表
            
        Returns:
            分析结果
        """
        if not messages:
            print("❌ 没有消息可供分析")
            return {}
        
        print("\n🔍 开始分析聊天模式...")
        
        # 初始化模式分析器
        from managers import ConfigManager
        config = ConfigManager('config.json')
        pattern_analyzer = PatternAnalyzer(config, self.processor.output_dir)
        
        # 使用模式分析器进行分析
        analysis_result = pattern_analyzer.analyze_chat_patterns(messages)
        
        return analysis_result

    def save_analysis_report(self, analysis_result: Dict, contact_name: str = None):
        """保存分析报告
        
        Args:
            analysis_result: 分析结果
            contact_name: 联系人名称（如果是单个联系人分析）
        """
        if not analysis_result:
            print("❌ 没有分析结果可保存")
            return
        
        # 确定输出文件名
        if contact_name:
            filename = f"analysis_report_{contact_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            filename = f"analysis_report_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file = os.path.join(self.processor.output_dir, filename)
        
        # 添加元数据
        report = {
            "analysis_time": datetime.now().isoformat(),
            "analyzed_contact": contact_name,
            "analysis_type": "single_contact" if contact_name else "all_contacts",
            "results": analysis_result
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"✅ 分析报告已保存: {output_file}")
            
        except Exception as e:
            print(f"❌ 保存分析报告失败: {e}")

    def print_analysis_summary(self, analysis_result: Dict):
        """打印分析结果摘要"""
        if not analysis_result:
            return
        
        print("\n📊 === 分析结果摘要 ===")
        
        if 'total_contacts' in analysis_result:
            print(f"👥 分析联系人数: {analysis_result['total_contacts']}")
        
        if 'total_messages' in analysis_result:
            print(f"💬 分析消息总数: {analysis_result['total_messages']}")
        
        if 'processing_time' in analysis_result:
            print(f"⏱️ 处理时间: {analysis_result['processing_time']}")
        
        print("\n💡 详细分析结果请查看输出的JSON文件")

    def process_analyze_independent(self, contact_name: str = None, data_dir: str = "data"):
        """独立聊天分析流程，不依赖stage1结果
        
        Args:
            contact_name: 特定联系人名称，None表示分析所有
            data_dir: 数据目录
        """
        print("\n🧠 === 独立聊天分析流程 ===")
        print("📝 此流程直接从分组数据文件夹读取，无需stage1支持")
        print("=" * 50)
        
        # 如果没有指定联系人，先列出所有可用联系人
        if not contact_name:
            contacts = self.list_available_contacts(data_dir)
            if contacts:
                print(f"\n📋 找到 {len(contacts)} 个联系人文件夹:")
                for i, contact in enumerate(contacts, 1):
                    print(f"  {i:2d}. {contact}")
                print("\n💡 可以使用 'python main.py analyze --contact 联系人名称' 分析特定联系人")
            else:
                print("\n📭 没有找到任何联系人文件夹")
                return
        
        # 加载消息
        try:
            messages = self.load_all_messages_for_analysis(contact_name, data_dir)
        except Exception as e:
            print(f"❌ 加载消息失败: {e}")
            return
        
        if not messages:
            print("❌ 没有找到可分析的消息")
            return
        
        # 执行分析
        try:
            analysis_result = self.analyze_messages_patterns(messages)
        except Exception as e:
            print(f"❌ 分析失败: {e}")
            return
        
        # 保存报告
        try:
            self.save_analysis_report(analysis_result, contact_name)
            self.print_analysis_summary(analysis_result)
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")
            return
        
        print("\n🎉 分析完成！")


def show_help():
    """显示帮助信息"""
    help_text = """
🔧 SoulByte - 数字人生成系列项目 · 数据处理组件 v2.0

📖 功能概述:
  SoulByte 是数字人生成系列项目的核心数据处理组件，专注于从微信聊天记录中
  提取"数字灵魂"，为个性化数字人训练提供高质量数据基础。

🎯 项目定位:
  • 数字人生成系列项目的数据基石
  • 个人聊天数据的智能处理和知识萃取
  • 为后续模型训练和知识库构建奠定基础

💡 核心改进:
  • 历史记录扩展到前72小时（原来只有当天）
  • 智能联系人过滤（只显示相关对话历史）
  • 完整的日期时间显示和日期分隔线
  • 模块化代码结构，易于维护和扩展

🛣️ 发展规划:
  v2.x - 微调流程、可视化界面、前后端联通
  v3.x - 个人知识库、RAG检索、Rerank排序、Embedding优化
  v4.x+ - 数字人生成、智能对话引擎、生态系统集成

🚀 使用方法:
  python main.py                    # 完整处理流程（兼容模式）
  python main.py stage1             # 阶段1: 数据提取和联系人信息
  python main.py stage2             # 阶段2: 大模型评估和最终数据集
  python main.py contacts           # 管理联系人关系
  python main.py filter             # 从评估缓存筛选数据
  python main.py analyze            # 独立聊天分析（不依赖stage1）
  python main.py help               # 显示帮助

🧠 独立分析使用方法:
  python main.py analyze                      # 分析所有联系人
  python main.py analyze --contact 张三       # 只分析特定联系人
  python main.py analyze --list               # 列出所有可用联系人
  python main.py analyze --data-dir ./chats   # 指定数据目录

📋 推荐工作流:
  1. 运行 stage1 提取数据和建立联系人信息
  2. 使用 contacts 管理和编辑联系人关系
  3. 运行 stage2 进行质量评估和生成最终数据集
  4. 使用 filter 从已有评估缓存中重新筛选数据

📁 项目结构:
  main.py              # 主入口文件
  core/                 # 核心处理模块
  ├── data_processor.py    # 数据处理器
  └── main_processor.py    # 主要处理流程
  managers/             # 管理器模块
  ├── config_manager.py    # 配置管理
  ├── contact_manager.py   # 联系人管理
  └── evaluation_cache.py  # 评估缓存
  utils/                # 工具模块
  └── history_utils.py     # 历史记录处理
  evaluators/           # 评估模块
  └── llm_evaluator.py     # 大模型评估器

⚙️ 配置文件:
  确保您有 config.json 配置文件，参考项目中的示例配置。

📊 输出文件:
  • output/stage1_results.json      - 阶段1完整结果
  • output/training_data.json       - 最终训练数据
  • output/contacts.json            - 联系人信息数据库
  • output/evaluation_cache.json    - 评估结果缓存

🆘 获取支持:
  如有问题，请查看 README.md 文件或检查配置是否正确。
"""
    print(help_text)


def main():
    """主函数"""
    print("SoulByte v2.0")
    print("SoulByte - Intelligent WeChat Data Processing Tool v2.0")
    print("=" * 60)
    
    app = WeChatProcessorApp()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'stage1':
            print("🚀 执行阶段1: 数据提取和联系人信息建立")
            app.process_stage1_extract_and_contacts()
            return
            
        elif command == 'stage2':
            print("🔍 执行阶段2: 大模型评估和最终数据集生成")
            app.process_stage2_evaluation_and_final()
            return
            
        elif command == 'contacts':
            print("👥 进入联系人管理模式")
            app.manage_contacts_interactive()
            return
            
        elif command == 'filter':
            print("🔍 从评估缓存筛选数据")
            app.filter_from_cache()
            return
            
        elif command == 'analyze':
            print("🧠 独立聊天分析")
            # 解析额外参数
            contact_name = None
            data_dir = "data"
            
            # 简单的参数解析
            for i, arg in enumerate(sys.argv[2:], start=2):
                if arg == '--contact' and i + 1 < len(sys.argv):
                    contact_name = sys.argv[i + 1]
                elif arg == '--data-dir' and i + 1 < len(sys.argv):
                    data_dir = sys.argv[i + 1]
                elif arg == '--list':
                    contacts = app.list_available_contacts(data_dir)
                    if contacts:
                        print(f"\n📋 找到 {len(contacts)} 个联系人文件夹:")
                        for i, contact in enumerate(contacts, 1):
                            print(f"  {i:2d}. {contact}")
                    else:
                        print("\n📭 没有找到任何联系人文件夹")
                    return
            
            app.process_analyze_independent(contact_name, data_dir)
            return
            
        elif command in ['help', '--help', '-h']:
            show_help()
            return
            
        else:
            print(f"❌ 未知命令: {command}")
            show_help()
            return
    
    # 默认处理模式（兼容模式）
    print("🔄 使用完整处理流程（兼容模式）")
    app.process_full()


if __name__ == "__main__":
    main()