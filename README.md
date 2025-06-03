# SoulByte - 微信聊天数据处理工具 v2.0

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0-brightgreen.svg)](CHANGELOG.md)
[![Series](https://img.shields.io/badge/series-数字人生成-purple.svg)](ROADMAP.md)

SoulByte 是**数字人生成系列项目**的核心数据处理组件，专注于将微信聊天记录转换为高质量的AI训练数据集和个人知识库。该工具采用模块化设计，支持智能历史记录构建、联系人关系管理和大模型质量评估，为数字人的个性化训练和知识库构建奠定基础。

## 🎯 项目定位

### 📍 系列项目概览
- **当前阶段**: 数据预处理和训练数据生成
- **项目愿景**: 构建完整的数字人生成生态系统
- **核心目标**: 从个人聊天数据中提取"数字灵魂"，生成个性化数字人

### 🚀 SoulByte 在系列中的角色
- **数据基石**: 为数字人提供高质量的训练数据
- **知识萃取**: 从聊天记录中提取个人知识和语言模式
- **个性化**: 保持个人独特的表达风格和思维习惯

## 🌟 核心特性

### 🔧 当前功能（v2.0）
- **智能数据处理**: 72小时时间窗口的历史记录构建
- **联系人关系管理**: 智能识别和管理人际关系网络
- **质量评估**: 大模型驱动的训练数据质量评估
- **模块化架构**: 易于扩展和集成的组件化设计

### 🚀 智能历史记录构建
- **72小时时间窗口**：扩展历史记录范围，提供更丰富的对话上下文
- **智能联系人过滤**：只显示与当前对话者相关的历史消息
- **日期时间优化**：完整的日期显示和智能日期分隔线
- **上下文连贯性**：保持对话逻辑性和时间连续性

### 👥 联系人关系管理  
- **自动信息提取**：从users.json自动加载联系人基本信息
- **关系分类管理**：支持朋友、同事、家人等多种关系类型
- **交互式编辑**：友好的命令行界面管理联系人关系
- **统计分析**：提供详细的联系人活跃度统计

### 🤖 大模型质量评估
- **智能评分**：使用大模型对训练样本进行质量评估
- **缓存机制**：避免重复评估，提高处理效率
- **灵活筛选**：支持多种分数阈值的数据筛选
- **评估报告**：生成详细的质量评估报告

## 🛣️ 发展路线图

### 🔮 即将推出（v2.x）
- [ ] **微调流程集成**
  - 自动化模型微调管道
  - 个性化模型训练流程
  - 训练效果监控和评估

- [ ] **可视化界面**
  - Web端数据分析界面
  - 聊天记录可视化展示
  - 个人语言模式分析图表

- [ ] **前后端联通**
  - RESTful API接口
  - 实时数据处理服务
  - 多用户支持和权限管理

### 🌟 核心规划（v3.x）
- [ ] **个人知识库构建**
  - 智能知识抽取和组织
  - 语义化知识图谱构建
  - 个人记忆和经验归档

- [ ] **RAG增强检索**
  - 向量数据库集成
  - 智能检索和匹配
  - 上下文感知的知识检索

- [ ] **Rerank智能排序**
  - 多模态相关性排序
  - 个性化排序模型
  - 动态权重调整

- [ ] **Embedding优化**
  - 个性化向量表示
  - 多维度语义编码
  - 实时向量更新

### 🎯 终极愿景（v4.x+）
- [ ] **数字人生成**
  - 完整的数字人格模型
  - 多模态交互能力
  - 情感和个性表达

- [ ] **智能对话引擎**
  - 上下文感知对话
  - 长期记忆管理
  - 个性化回复生成

- [ ] **生态系统集成**
  - 多平台数据源支持
  - 第三方应用集成
  - 开放API生态

## 📁 项目结构

```
soulbyte/
├── main.py                     # 🚀 主入口文件
├── README.md                   # 📖 项目文档
├── ROADMAP.md                  # 🛣️ 发展路线图（规划中）
├── config.json                 # ⚙️ 配置文件
├── requirements.txt            # 📦 依赖列表
│
├── core/                       # 🔧 核心处理模块
│   ├── __init__.py
│   ├── data_processor.py       # 数据处理器
│   └── main_processor.py       # 主要处理流程
│
├── managers/                   # 📋 管理器模块
│   ├── __init__.py
│   ├── config_manager.py       # 配置管理
│   ├── contact_manager.py      # 联系人管理
│   └── evaluation_cache.py     # 评估缓存管理
│
├── utils/                      # 🛠️ 工具模块
│   ├── __init__.py
│   └── history_utils.py        # 历史记录处理工具
│
├── evaluators/                 # 🤖 评估模块
│   ├── __init__.py
│   └── llm_evaluator.py        # 大模型评估器
│
├── data/                       # 📁 数据目录
│   ├── contact1/
│   │   ├── messages.csv
│   │   └── users.json
│   └── contact2/
│       ├── messages.csv
│       └── users.json
│
├── output/                     # 📤 输出目录
│   ├── stage1_results.json     # 阶段1结果
│   ├── training_data.json      # 最终训练数据
│   ├── contacts.json           # 联系人数据库
│   └── evaluation_cache.json   # 评估缓存
│
└── future_modules/             # 🚧 未来模块（规划中）
    ├── fine_tuning/            # 微调模块
    ├── web_interface/          # Web界面
    ├── knowledge_base/         # 知识库模块
    ├── rag_engine/             # RAG引擎
    └── digital_human/          # 数字人生成
```

## 🚀 快速开始

### 环境准备

1. **Python环境要求**
   ```bash
   # Python 3.8 或更高版本
   python --version
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **数据准备**
   
   将微信聊天记录CSV文件放置在以下结构中：
   ```
   data/
   ├── 联系人1/
   │   ├── messages.csv    # 聊天记录
   │   └── users.json      # 联系人信息
   ├── 联系人2/
   │   ├── messages.csv
   │   └── users.json
   └── ...
   ```

4. **配置文件**
   
   复制并编辑配置文件：
   ```bash
   cp config.example.json config.json
   # 编辑 config.json，设置你的参数
   ```

### 基本使用

#### 🎯 推荐工作流（分阶段处理）

```bash
# 1. 阶段1: 数据提取和联系人信息建立
python main.py stage1

# 2. 管理联系人关系（可选）
python main.py contacts

# 3. 阶段2: 大模型评估和最终数据集生成
python main.py stage2

# 4. 从缓存筛选不同分数的数据（可选）
python main.py filter
```

#### 🔄 完整处理流程（兼容模式）

```bash
# 一次性完成所有处理步骤
python main.py
```

#### 📋 查看帮助

```bash
python main.py help
```

## ⚙️ 配置说明

### 基本配置 (config.json)

```json
{
  "data_processing": {
    "data_dir": "data",                    // 数据目录路径
    "output_file": "training_data.json",   // 输出文件名
    "my_wxid": "your_wechat_id",          // 你的微信ID
    "max_inter_message_gap": 90,          // 消息块内最大间隔（秒）
    "max_reply_delay": 300,               // 最大回复延迟（秒）
    "use_multiprocessing": true,          // 是否使用多进程
    "max_workers": 4                      // 最大进程数
  },
  "llm_evaluation": {
    "enabled": true,                      // 是否启用大模型评估
    "api_url": "https://api.example.com/v1/chat/completions",
    "api_key": "your_api_key",
    "model": "your_model_name",
    "max_workers": 3,                     // 评估并发数
    "min_score": 5.0,                     // 最低通过分数
    "timeout": 30,                        // 请求超时时间
    "retry_attempts": 3,                  // 重试次数
    "evaluation_prompt": "请评估这个训练样本的质量..."
  }
}
```

### 高级配置选项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `max_inter_message_gap` | 90 | 同一消息块内消息间最大时间间隔（秒） |
| `max_reply_delay` | 300 | 认定为有效回复的最大延迟时间（秒） |
| `use_multiprocessing` | true | 大数据集时是否启用多进程加速 |
| `max_workers` | 4 | 多进程处理时的最大进程数 |
| `min_score` | 5.0 | 大模型评估的最低通过分数（1-10分） |

## 💡 核心功能详解

### 🔄 历史记录构建

#### 传统方案 vs 新方案对比

**传统方案问题：**
```python
# 旧版本只获取当日消息，且混合所有联系人
def build_context_old(messages, reply_time):
    reply_date = reply_time.date()
    context_messages = []
    for msg in messages:
        if msg['create_time'].date() == reply_date:
            context_messages.append(msg)  # 包含所有人的消息
    return context_messages
```

**新方案改进：**
```python
# 新版本：72小时窗口 + 智能联系人过滤
def build_context_new(messages, reply_time, current_talker_id):
    time_72h_ago = reply_time - timedelta(hours=72)
    context_messages = []
    for msg in messages:
        if time_72h_ago <= msg['create_time'] < reply_time:
            # 只包含相关对话：对方发的消息 + 我的所有消息
            if (msg['is_sender'] == 0 and msg['talker'] == current_talker_id) or \
               (msg['is_sender'] == 1):
                context_messages.append(msg)
    return context_messages
```

#### 历史记录展示效果

**新版本显示效果：**
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

### 👥 联系人关系管理

#### 支持的关系类型
- 👫 **朋友** - 一般朋友关系
- 💼 **同事** - 工作相关联系人  
- 🎓 **同学** - 学校关系
- 👨‍👩‍👧‍👦 **家人** - 家庭成员
- 💕 **恋人** - 恋爱关系
- 👨‍🏫 **师长** - 老师、导师等
- 👨‍🎓 **学生** - 学生关系
- 🤝 **客户** - 商务关系
- ❓ **陌生人** - 不熟悉的人
- 📋 **其他** - 自定义关系

#### 联系人管理操作

```bash
# 进入联系人管理界面
python main.py contacts

# 支持的操作：
# 1. 查看所有联系人 - 显示联系人列表和统计信息
# 2. 搜索联系人 - 通过昵称/备注/ID搜索
# 3. 更新联系人关系 - 修改关系类型和详细备注
# 4. 导出联系人列表 - 生成完整的联系人报告
```

### 🤖 大模型质量评估

#### 评估标准
- **对话自然性** (25%) - 回复是否自然流畅
- **内容相关性** (25%) - 回复是否与上下文相关
- **语言风格** (25%) - 是否保持一致的语言风格
- **信息完整性** (25%) - 是否包含足够的上下文信息

#### 评估流程
1. **智能缓存** - 避免重复评估相同样本
2. **并发处理** - 支持多线程并发评估
3. **错误重试** - 网络错误时自动重试
4. **分数解析** - 智能提取评分结果
5. **结果筛选** - 根据分数阈值筛选高质量数据

#### 评估结果分析
```json
{
  "total_samples": 1000,
  "evaluated_samples": 1000,
  "passed_samples": 750,
  "average_score": 6.8,
  "score_distribution": {
    "0-2": 50,
    "2-4": 100,
    "4-6": 100,
    "6-8": 500,
    "8-10": 250
  }
}
```

## 📊 输出文件说明

### 阶段1输出文件

| 文件名 | 描述 | 内容 |
|--------|------|------|
| `stage1_results.json` | 完整的阶段1结果 | 训练数据、语言模式、原始消息、对话回合 |
| `stage1_training_data.json` | 纯训练数据 | 仅包含instruction、input、output格式的训练样本 |
| `contacts.json` | 联系人数据库 | 所有联系人的详细信息和关系数据 |

### 阶段2输出文件

| 文件名 | 描述 | 内容 |
|--------|------|------|
| `training_data.json` | 最终完整结果 | 经过评估的完整训练数据和分析报告 |
| `training_data_training_only.json` | 最终纯训练数据 | 仅包含高质量的训练样本 |
| `evaluation_cache.json` | 评估缓存 | 所有样本的评估结果缓存 |
| `evaluation_report.json` | 评估报告 | 详细的质量评估统计和分析 |

### 缓存筛选输出

| 文件名 | 描述 |
|--------|------|
| `training_data_filtered_{score}.json` | 按分数筛选的完整结果 |
| `training_only_filtered_{score}.json` | 按分数筛选的纯训练数据 |

## 🔧 高级用法

### 批量处理多个数据集

```bash
# 处理不同数据集
python main.py stage1 --data-dir dataset1
python main.py stage1 --data-dir dataset2

# 合并结果
python tools/merge_datasets.py dataset1/output dataset2/output
```

### 自定义评估标准

```python
# 在 evaluators/llm_evaluator.py 中修改评估提示
evaluation_prompt = """
自定义评估标准：
1. 专业术语使用准确性 (30%)
2. 情感表达恰当性 (30%)  
3. 对话逻辑连贯性 (25%)
4. 文化背景适应性 (15%)

请对以下样本评分(1-10分)：
指令: {instruction}
输入: {input}
输出: {output}
"""
```

### 性能优化建议

#### 大数据集处理优化

```json
{
  "data_processing": {
    "use_multiprocessing": true,
    "max_workers": 8,              // 根据CPU核心数调整
    "chunk_size": 5000             // 增大块大小
  },
  "llm_evaluation": {
    "max_workers": 5,              // 根据API限制调整
    "batch_size": 20,              // 批量评估
    "timeout": 60                  // 增加超时时间
  }
}
```

#### 内存使用优化

```python
# 对于超大数据集，启用流式处理
ENABLE_STREAMING = True
STREAM_CHUNK_SIZE = 1000

# 定期清理内存
import gc
gc.collect()
```

## 🐛 故障排除

### 常见问题解决

#### 1. 导入错误
```bash
# 错误: ModuleNotFoundError: No module named 'core'
# 解决: 确保在项目根目录运行
cd /path/to/soulbyte
python main.py
```

#### 2. 数据格式错误  
```bash
# 错误: CSV文件格式不正确
# 解决: 检查CSV文件是否包含必要字段
required_columns = ['id', 'CreateTime', 'is_sender', 'talker', 'msg', 'type_name']
```

#### 3. 大模型API错误
```bash
# 错误: API调用失败
# 解决: 检查API配置和网络连接
# 1. 验证API密钥是否正确
# 2. 检查API URL是否可访问
# 3. 确认API请求格式是否符合要求
```

#### 4. 内存不足
```bash
# 错误: MemoryError
# 解决: 调整处理参数
{
  "data_processing": {
    "use_multiprocessing": false,  // 禁用多进程
    "max_workers": 2,              // 减少进程数
    "chunk_size": 1000            // 减小块大小
  }
}
```

### 日志和调试

#### 启用详细日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 性能监控
```bash
# 监控内存使用
python -m memory_profiler main.py stage1

# 监控CPU使用
python -m cProfile -o profile.prof main.py stage1
```

## 🤝 参与贡献

### 🔧 开发贡献
我们欢迎各种形式的贡献：
- **代码开发**: 新功能开发、Bug修复、性能优化
- **文档完善**: 使用说明、API文档、教程编写
- **测试验证**: 功能测试、性能测试、兼容性测试
- **创意建议**: 新功能建议、用户体验改进

### 📊 数据贡献
- **测试数据**: 提供匿名化的测试数据集
- **使用反馈**: 分享使用经验和改进建议
- **效果评估**: 帮助评估数据质量和处理效果

### 🌟 社区建设
- **经验分享**: 分享数字人生成的心得体会
- **技术交流**: 参与技术讨论和问题解答
- **生态建设**: 推广项目，扩大影响力

## 📄 许可证与声明

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

### ⚠️ 重要声明
- **隐私保护**: 请确保处理的聊天数据已获得相关人员同意
- **数据安全**: 建议在本地环境处理敏感个人数据
- **合规使用**: 请遵守相关法律法规和隐私政策
- **研究目的**: 本工具主要用于学术研究和个人学习

## 🆘 获取支持

- 📧 **邮件支持**: support@soulbyte.dev
- 💬 **社区讨论**: [GitHub Discussions](https://github.com/your-repo/discussions)
- 🐛 **问题报告**: [GitHub Issues](https://github.com/your-repo/issues)
- 📖 **详细文档**: [Wiki页面](https://github.com/your-repo/wiki)
- 🎯 **发展规划**: [项目路线图](ROADMAP.md)

## 🎯 数字人生成愿景

### 🧠 个人数字化
通过SoulByte处理的聊天数据将成为数字人的"记忆基石"：
- **语言习惯**: 提取个人独特的表达方式和语言风格
- **思维模式**: 分析个人的逻辑思维和决策模式
- **情感特征**: 识别个人的情感表达和反应模式
- **知识结构**: 构建个人的知识体系和专业领域

### 🔮 未来应用场景
- **智能助手**: 个性化的AI助手，完全模拟个人风格
- **虚拟陪伴**: 基于真实记忆的虚拟伴侣或朋友
- **知识传承**: 永久保存个人的智慧和经验
- **教育培训**: 个性化的学习和指导系统

### 🌐 技术生态
SoulByte将与以下技术深度集成：
- **大语言模型**: GPT、Claude、LLaMA等主流模型
- **向量数据库**: Pinecone、Weaviate、Chroma等
- **机器学习框架**: PyTorch、TensorFlow、HuggingFace
- **可视化工具**: Streamlit、Gradio、Plotly等

---

**🌟 如果这个项目对您有帮助，请给我们一个星标！**

**📢 欢迎分享您的使用经验和改进建议！**

**🚀 期待与您一起构建数字人生成的未来！**