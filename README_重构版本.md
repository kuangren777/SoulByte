# 微信聊天记录处理工具 - 重构版本

## 重要更新

### 🔧 代码重构
原本的单一大文件 `process_wechat_data.py` 已被重构为多个专门的模块：

### 📁 新的文件结构

```
.
├── config_manager.py          # 配置管理器
├── contact_manager.py         # 联系人管理器  
├── evaluation_cache.py        # 评估缓存管理器
├── llm_evaluator.py          # 大模型评估器
├── history_utils.py          # 历史记录处理工具
├── data_processor.py         # 核心数据处理器
├── main_processor.py         # 主要处理流程
├── wechat_processor_main.py  # 新的主入口文件
└── process_wechat_data.py    # 原文件（保留兼容性）
```

### ✨ 功能改进

#### 1. 历史记录大幅改进
- **扩展时间范围**：从当日扩展到前72小时的历史记录
- **联系人过滤**：只显示与当前对话联系人相关的历史消息
- **日期显示**：聊天记录现在显示完整的日期和时间信息
- **日期分隔线**：当日期发生变化时会自动添加日期分隔线

#### 2. 历史记录示例
```
————— 2024年03月15日 —————
[14:32:15] 小明: 今天的会议准备得怎么样？
[14:35:22] 我: 差不多了，PPT已经做完了
[16:20:45] 小明: 那就好，明天见

————— 2024年03月16日 —————  
[09:15:30] 小明: 早上好！
[09:16:12] 我: 早上好，准备出发了吗？
[18:45:23] 小明: 今天开会的事情我想再讨论一下
```

#### 3. 更智能的对话上下文
- **72小时窗口**：获取回复时间前72小时内的相关对话
- **关系化过滤**：只包含我和当前联系人之间的对话历史
- **时间连续性**：保持对话的时间连续性和逻辑性

## 使用方法

### 新的主入口（推荐）
```bash
# 使用新的重构版本
python wechat_processor_main.py stage1    # 阶段1: 数据提取
python wechat_processor_main.py contacts  # 管理联系人关系
python wechat_processor_main.py stage2    # 阶段2: 评估和生成
python wechat_processor_main.py filter    # 从缓存筛选
python wechat_processor_main.py help      # 查看帮助
```

### 兼容旧版本
```bash
# 原有的入口仍然可用
python process_wechat_data.py stage1
python process_wechat_data.py contacts
python process_wechat_data.py stage2
```

## 技术改进

### 模块化设计
- **ConfigManager**: 统一的配置管理
- **ContactManager**: 联系人信息管理和关系维护
- **EvaluationCache**: 评估结果缓存，支持增量评估
- **LLMEvaluator**: 大模型评估接口
- **HistoryManager**: 智能历史记录构建
- **DataProcessor**: 核心数据处理逻辑

### 性能优化
- **多进程支持**: 大数据集的多进程处理
- **增量处理**: 避免重复处理已处理过的文件
- **缓存机制**: 评估结果缓存，避免重复评估
- **内存优化**: 更好的内存使用和垃圾回收

### 历史记录处理逻辑

#### 旧版本问题
```python
# 旧版本只获取当日消息
def build_context(messages, reply_time):
    reply_date = reply_time.date()
    context_messages = []
    for msg in messages:
        msg_date = msg['create_time'].date()
        if msg_date == reply_date and msg['create_time'] < reply_time:
            context_messages.append(msg)  # 混合所有联系人的消息
    return context_messages
```

#### 新版本改进
```python
# 新版本获取前72小时相关对话
def build_context(messages, reply_time, current_talker_id):
    time_72h_ago = reply_time - timedelta(hours=72)
    context_messages = []
    for msg in messages:
        if time_72h_ago <= msg['create_time'] < reply_time:
            # 只包含与当前联系人相关的消息
            if (msg['is_sender'] == 0 and msg['talker'] == current_talker_id) or \
               (msg['is_sender'] == 1):  # 我的所有消息
                context_messages.append(msg)
    return context_messages
```

## 配置文件

确保您的 `config.json` 包含所有必要的配置：

```json
{
  "data_processing": {
    "data_dir": "data",
    "output_file": "training_data.json",
    "my_wxid": "your_wechat_id",
    "max_inter_message_gap": 90,
    "max_reply_delay": 300,
    "use_multiprocessing": true,
    "max_workers": 4
  },
  "llm_evaluation": {
    "enabled": true,
    "api_url": "your_api_url",
    "api_key": "your_api_key",
    "model": "your_model_name",
    "max_workers": 3,
    "min_score": 5.0,
    "timeout": 30,
    "retry_attempts": 3,
    "evaluation_prompt": "请评估这个训练样本的质量..."
  }
}
```

## 输出文件说明

### 阶段1输出
- `output/stage1_results.json`: 完整的阶段1结果
- `output/stage1_training_data.json`: 纯训练数据
- `output/contacts.json`: 联系人信息数据库

### 阶段2输出  
- `output/training_data.json`: 最终完整结果
- `output/training_data_training_only.json`: 纯训练数据
- `output/evaluation_cache.json`: 评估缓存
- `output/evaluation_report.json`: 评估报告（如果启用评估）

### 缓存筛选输出
- `output/training_data_filtered_{score}.json`: 按分数筛选的完整结果
- `output/training_only_filtered_{score}.json`: 按分数筛选的纯训练数据

## 兼容性说明

- 原有的 `process_wechat_data.py` 文件保留，确保向后兼容
- 所有原有的配置和数据格式保持不变
- 可以无缝从旧版本迁移到新版本

## 推荐工作流

1. **数据提取**: `python wechat_processor_main.py stage1`
2. **编辑联系人关系**: `python wechat_processor_main.py contacts`
3. **质量评估**: `python wechat_processor_main.py stage2`
4. **调整筛选阈值**: `python wechat_processor_main.py filter`

## 注意事项

- 新版本的历史记录包含更多上下文信息，生成的训练数据质量更高
- 72小时窗口确保了足够的对话背景，同时避免了过于久远的无关信息
- 联系人过滤确保历史记录的相关性和一致性
- 建议在处理大量数据时启用多进程处理以提高效率 