#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
微信聊天记录处理工具迁移脚本
用于验证新版本与旧版本的兼容性和功能对比
"""

import os
import sys
import json
from datetime import datetime


def check_file_structure():
    """检查新版本的文件结构"""
    print("=== 检查新版本文件结构 ===")
    
    required_files = [
        'config_manager.py',
        'contact_manager.py', 
        'evaluation_cache.py',
        'llm_evaluator.py',
        'history_utils.py',
        'data_processor.py',
        'main_processor.py',
        'wechat_processor_main.py'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} (缺失)")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n警告：缺失 {len(missing_files)} 个文件")
        return False
    else:
        print("\n✅ 所有文件都存在")
        return True


def test_imports():
    """测试模块导入"""
    print("\n=== 测试模块导入 ===")
    
    modules_to_test = [
        ('config_manager', 'ConfigManager'),
        ('contact_manager', 'ContactManager'),
        ('evaluation_cache', 'EvaluationCache'), 
        ('llm_evaluator', 'LLMEvaluator'),
        ('history_utils', 'HistoryManager'),
        ('data_processor', 'WeChatDataProcessor'),
        ('main_processor', 'WeChatMainProcessor'),
        ('wechat_processor_main', 'WeChatProcessorApp')
    ]
    
    success_count = 0
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
    
    print(f"\n导入测试结果: {success_count}/{len(modules_to_test)} 成功")
    return success_count == len(modules_to_test)


def check_config_compatibility():
    """检查配置文件兼容性"""
    print("\n=== 检查配置文件兼容性 ===")
    
    if not os.path.exists('config.json'):
        print("❌ 未找到 config.json 文件")
        print("建议创建配置文件，参考 README_重构版本.md 中的配置示例")
        return False
    
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查必要的配置节
        required_sections = ['data_processing']
        optional_sections = ['llm_evaluation']
        
        for section in required_sections:
            if section in config:
                print(f"✅ 配置节 '{section}' 存在")
            else:
                print(f"❌ 配置节 '{section}' 缺失")
                return False
        
        for section in optional_sections:
            if section in config:
                print(f"✅ 配置节 '{section}' 存在 (可选)")
            else:
                print(f"⚠️  配置节 '{section}' 不存在 (可选，但推荐添加)")
        
        print("✅ 配置文件兼容")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件解析失败: {e}")
        return False


def check_data_directory():
    """检查数据目录结构"""
    print("\n=== 检查数据目录结构 ===")
    
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录 '{data_dir}' 不存在")
        return False
    
    print(f"✅ 数据目录 '{data_dir}' 存在")
    
    # 检查是否有子目录和CSV文件
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not subdirs:
        print("⚠️  数据目录中没有子目录")
        return False
    
    csv_count = 0
    users_json_count = 0
    
    for subdir in subdirs[:3]:  # 只检查前3个子目录
        subdir_path = os.path.join(data_dir, subdir)
        csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
        csv_count += len(csv_files)
        
        if os.path.exists(os.path.join(subdir_path, 'users.json')):
            users_json_count += 1
    
    print(f"✅ 找到 {len(subdirs)} 个联系人目录")
    print(f"✅ 找到 {csv_count} 个CSV文件 (前3个目录)")
    print(f"✅ 找到 {users_json_count} 个users.json文件 (前3个目录)")
    
    return True


def check_output_directory():
    """检查输出目录"""
    print("\n=== 检查输出目录 ===")
    
    output_dir = 'output'
    if not os.path.exists(output_dir):
        print(f"ℹ️  输出目录 '{output_dir}' 不存在，将在首次运行时创建")
        return True
    
    print(f"✅ 输出目录 '{output_dir}' 存在")
    
    # 检查是否有之前的处理结果
    existing_files = os.listdir(output_dir)
    if existing_files:
        print(f"ℹ️  输出目录包含 {len(existing_files)} 个文件（之前的处理结果）")
        
        important_files = [
            'stage1_results.json',
            'contacts.json', 
            'evaluation_cache.json',
            'training_data.json'
        ]
        
        for file in important_files:
            if file in existing_files:
                print(f"  📁 {file}")
    
    return True


def run_compatibility_test():
    """运行兼容性测试"""
    print("\n=== 运行功能兼容性测试 ===")
    
    try:
        # 测试新版本主入口
        from wechat_processor_main import WeChatProcessorApp
        app = WeChatProcessorApp()
        print("✅ 新版本主应用初始化成功")
        
        # 测试历史管理器
        from history_utils import HistoryManager
        from contact_manager import ContactManager
        
        contact_manager = ContactManager('output')
        history_manager = HistoryManager(contact_manager)
        print("✅ 历史记录管理器初始化成功")
        
        # 测试配置管理器
        from config_manager import ConfigManager
        if os.path.exists('config.json'):
            config_manager = ConfigManager('config.json')
            print("✅ 配置管理器加载成功")
        else:
            config_manager = ConfigManager('nonexistent.json')  # 测试默认配置
            print("✅ 配置管理器默认初始化成功")
        
        print("✅ 所有功能模块测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 兼容性测试失败: {e}")
        return False


def create_sample_config():
    """创建示例配置文件"""
    print("\n=== 创建示例配置文件 ===")
    
    if os.path.exists('config.json'):
        backup_name = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.rename('config.json', backup_name)
        print(f"已备份现有配置文件为: {backup_name}")
    
    sample_config = {
        "data_processing": {
            "data_dir": "data",
            "output_file": "training_data.json",
            "my_wxid": "your_wechat_id_here",
            "max_inter_message_gap": 90,
            "max_reply_delay": 300,
            "use_multiprocessing": True,
            "max_workers": 4
        },
        "llm_evaluation": {
            "enabled": False,
            "api_url": "your_api_url_here",
            "api_key": "your_api_key_here", 
            "model": "your_model_name_here",
            "max_workers": 3,
            "min_score": 5.0,
            "timeout": 30,
            "retry_attempts": 3,
            "evaluation_prompt": "请评估这个微信聊天训练样本的质量。考虑以下几个方面：\n1. 对话的自然性和流畅性\n2. 回复的相关性和准确性\n3. 语言风格的一致性\n4. 是否包含足够的上下文信息\n\n请给出1-10分的评分，其中10分为最高质量。\n\n指令: {instruction}\n输入: {input}\n输出: {output}\n\n评分："
        }
    }
    
    with open('config.json', 'w', encoding='utf-8') as f:
        json.dump(sample_config, f, ensure_ascii=False, indent=2)
    
    print("✅ 已创建示例配置文件 config.json")
    print("请编辑配置文件，设置正确的参数后再运行处理程序")


def main():
    """主函数"""
    print("微信聊天记录处理工具 - 迁移检查脚本")
    print("=" * 50)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"当前目录: {os.getcwd()}")
    print("=" * 50)
    
    # 运行各项检查
    checks = [
        ("文件结构", check_file_structure),
        ("模块导入", test_imports),
        ("配置兼容性", check_config_compatibility),
        ("数据目录", check_data_directory),
        ("输出目录", check_output_directory),
        ("功能兼容性", run_compatibility_test)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed_checks += 1
        except Exception as e:
            print(f"❌ {check_name} 检查时出错: {e}")
    
    print("\n" + "=" * 50)
    print(f"检查完成: {passed_checks}/{total_checks} 项通过")
    
    if passed_checks == total_checks:
        print("🎉 恭喜！所有检查都通过了，可以使用新版本")
        print("\n推荐使用方式:")
        print("python wechat_processor_main.py stage1")
        print("python wechat_processor_main.py contacts")
        print("python wechat_processor_main.py stage2")
    elif passed_checks >= total_checks - 2:
        print("⚠️  大部分检查通过，但有一些小问题需要解决")
        print("可以尝试运行新版本，如遇问题可切回旧版本")
    else:
        print("❌ 检查未通过，建议先解决问题再使用新版本")
        print("可以继续使用旧版本: python process_wechat_data.py")
    
    # 询问是否创建配置文件
    if not os.path.exists('config.json'):
        response = input("\n是否创建示例配置文件? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            create_sample_config()
    
    print("\n查看详细说明请参考: README_重构版本.md")


if __name__ == "__main__":
    main() 