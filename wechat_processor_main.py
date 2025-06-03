#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from datetime import datetime
from collections import Counter
from typing import List, Dict
import jieba
import pandas as pd
from tqdm import tqdm

from main_processor import WeChatMainProcessor
from contact_manager import ContactManager
from evaluation_cache import EvaluationCache


class WeChatProcessorApp:
    """微信聊天数据处理应用程序"""
    
    def __init__(self):
        self.processor = WeChatMainProcessor()
    
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
        
        print(f"阶段1结果已保存到: {stage1_file}")
        print(f"训练数据已保存到: {training_only_file}")
    
    def process_stage1_extract_and_contacts(self) -> None:
        """阶段1: 数据提取和联系人信息建立"""
        print("=== 阶段1: 数据提取和联系人信息建立 ===")
        print(f"使用多进程: {self.processor.use_multiprocessing}")
        
        # 0. 加载联系人信息数据库
        print("\n=== 步骤0: 加载联系人信息数据库 ===")
        self.processor.load_contact_database()
        
        # 1. 加载所有CSV文件
        print("\n=== 步骤1: 加载数据文件 ===")
        raw_messages = self.processor.load_all_csv_files()
        
        # 2. 数据清洗与预处理
        print("\n=== 步骤2: 数据清洗与预处理 ===")
        cleaned_messages = self.processor.clean_and_preprocess(raw_messages)
        
        # 3. 提取对话回合
        print("\n=== 步骤3: 提取对话回合 ===")
        conversation_rounds = self.processor.extract_message_blocks(cleaned_messages)
        
        # 4. 生成训练数据（不进行评估）
        print("\n=== 步骤4: 生成初始训练数据 ===")
        training_data = self.processor.format_training_data(conversation_rounds, cleaned_messages)
        
        # 5. 分析语言模式
        print("\n=== 步骤5: 分析语言模式 ===")
        language_patterns = self.analyze_language_patterns(cleaned_messages)
        
        # 6. 保存阶段1结果
        print("\n=== 步骤6: 保存阶段1结果 ===")
        self.save_stage1_results(training_data, language_patterns, cleaned_messages, conversation_rounds)
        
        print(f"\n阶段1完成！生成了 {len(training_data)} 条初始训练数据")
        print(f"联系人信息已保存到: {self.processor.contact_manager.contacts_file}")
        print("现在您可以编辑联系人关系，然后运行阶段2进行质量评估")
    
    def load_stage1_results(self) -> dict:
        """加载阶段1的结果"""
        stage1_file = os.path.join(self.processor.output_dir, 'stage1_results.json')
        if not os.path.exists(stage1_file):
            return None
        
        try:
            with open(stage1_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载阶段1结果失败: {e}")
            return None
    
    def process_stage2_evaluation_and_final(self) -> None:
        """阶段2: 大模型评估和最终数据集生成"""
        print("=== 阶段2: 大模型评估和最终数据集生成 ===")
        print(f"使用大模型评估: {self.processor.use_llm_evaluation}")
        
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
        self.processor.contact_manager = ContactManager(self.processor.output_dir)
        print(f"加载了 {len(self.processor.contact_manager.list_all_contacts())} 个联系人")
        
        evaluation_results = None
        final_training_data = training_data
        
        # 大模型评估（如果启用）
        if self.processor.use_llm_evaluation:
            print("\n=== 步骤3: 大模型质量评估 ===")
            final_training_data, evaluation_results = self.processor.evaluate_training_data(training_data)
        else:
            print("\n=== 步骤3: 跳过大模型评估（已禁用） ===")
        
        # 保存最终结果
        print("\n=== 步骤4: 保存最终结果 ===")
        self.save_final_results(final_training_data, language_patterns, evaluation_results)
        
        print(f"\n阶段2完成！最终生成了 {len(final_training_data)} 条训练数据")
        if evaluation_results:
            print(f"原始数据: {evaluation_results['total_samples']} 条")
            print(f"通过筛选: {evaluation_results['passed_samples']} 条")
            print(f"平均分数: {evaluation_results['average_score']:.2f}")
    
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
        
        print(f"最终训练数据已保存到: {training_only_file}")
        print(f"完整结果已保存到: {self.processor.output_file}")
        print(f"联系人信息已保存到: {self.processor.contact_manager.contacts_file}")
        if self.processor.use_llm_evaluation:
            print(f"筛选后数据已保存到: {filtered_file}")
            print(f"评估报告已保存到: {evaluation_report_file}")
    
    def filter_from_cache(self) -> None:
        """从评估缓存中筛选训练数据"""
        print("=== 从评估缓存筛选训练数据 ===")
        
        # 检查缓存文件是否存在
        if not os.path.exists(self.processor.evaluation_cache.cache_file):
            print(f"错误: 评估缓存文件 {self.processor.evaluation_cache.cache_file} 不存在")
            return
            
        # 获取配置的最低分数
        min_score = self.processor.min_score
        print(f"使用最低分数阈值: {min_score}")
        
        # 从缓存中筛选数据
        filtered_data = self.processor.evaluation_cache.filter_by_score(min_score)
        
        if not filtered_data:
            print("警告: 没有符合条件的数据")
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
            print("\n=== 联系人关系管理 ===")
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
                print("无效选择，请重新输入")
    
    def show_all_contacts(self, contact_manager: ContactManager) -> None:
        """显示所有联系人"""
        contacts = contact_manager.list_all_contacts()
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
    
    def search_contacts(self, contact_manager: ContactManager) -> None:
        """搜索联系人"""
        keyword = input("请输入搜索关键词（昵称/备注/ID）: ").strip()
        if not keyword:
            print("关键词不能为空")
            return
        
        matches = contact_manager.search_contacts(keyword)
        
        if not matches:
            print("未找到匹配的联系人")
            return
        
        print(f"\n找到 {len(matches)} 个匹配的联系人:")
        for i, contact in enumerate(matches, 1):
            display_name = contact.get('remark') or contact.get('nickname') or contact['contact_id'][-8:]
            relationship = contact.get('relationship', '朋友')
            print(f"{i}. {display_name} (ID: {contact['contact_id'][-8:]}) - {relationship}")
    
    def update_contact_relationship(self, contact_manager: ContactManager) -> None:
        """更新联系人关系"""
        keyword = input("请输入要更新的联系人（昵称/备注/ID）: ").strip()
        if not keyword:
            print("输入不能为空")
            return
        
        matches = contact_manager.search_contacts(keyword)
        
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
        contact_manager.update_relationship(
            selected_contact['contact_id'], 
            new_relationship, 
            new_detail
        )
        
        print("联系人关系更新成功！")
    
    def export_contacts(self, contact_manager: ContactManager) -> None:
        """导出联系人列表"""
        contacts = contact_manager.list_all_contacts()
        if not contacts:
            print("暂无联系人记录")
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
            print(f"联系人列表已导出到: {export_file}")
            print(f"总计 {contacts_summary['total_contacts']} 个联系人，{contacts_summary['total_messages']} 条消息")
        except Exception as e:
            print(f"导出失败: {e}")
    
    def process_full(self) -> None:
        """完整处理流程（兼容模式）"""
        print("开始处理微信聊天数据...")
        print("注意：建议使用分阶段处理模式")
        print("  stage1: 数据提取和联系人信息")
        print("  stage2: 大模型评估和最终数据集")
        print(f"使用多进程: {self.processor.use_multiprocessing}")
        print(f"使用大模型评估: {self.processor.use_llm_evaluation}")
        
        # 执行阶段1
        self.process_stage1_extract_and_contacts()
        
        # 执行阶段2
        print("\n" + "="*50)
        self.process_stage2_evaluation_and_final()


def main():
    """主函数"""
    app = WeChatProcessorApp()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'stage1':
            print("执行阶段1: 数据提取和联系人信息建立")
            app.process_stage1_extract_and_contacts()
            return
            
        elif command == 'stage2':
            print("执行阶段2: 大模型评估和最终数据集生成")
            app.process_stage2_evaluation_and_final()
            return
            
        elif command == 'contacts':
            app.manage_contacts_interactive()
            return
            
        elif command == 'filter':
            app.filter_from_cache()
            return
            
        elif command == 'help':
            print("微信聊天记录处理工具")
            print("用法:")
            print("  python wechat_processor_main.py           # 完整处理流程（兼容模式）")
            print("  python wechat_processor_main.py stage1    # 阶段1: 数据提取和联系人信息")
            print("  python wechat_processor_main.py stage2    # 阶段2: 大模型评估和最终数据集")
            print("  python wechat_processor_main.py contacts  # 管理联系人关系")
            print("  python wechat_processor_main.py filter    # 从评估缓存筛选数据")
            print("  python wechat_processor_main.py help      # 显示帮助")
            print("\n推荐流程:")
            print("  1. 运行 stage1 提取数据和建立联系人信息")
            print("  2. 使用 contacts 管理和编辑联系人关系")
            print("  3. 运行 stage2 进行质量评估和生成最终数据集")
            print("  4. 使用 filter 从已有评估缓存中重新筛选数据")
            print("\n重要更新：")
            print("  - 历史记录现在包含前72小时的相关对话")
            print("  - 聊天记录显示完整的日期和时间信息") 
            print("  - 只显示与当前联系人相关的历史对话")
            return
    
    # 默认处理模式（兼容模式）
    app.process_full()


if __name__ == "__main__":
    main() 