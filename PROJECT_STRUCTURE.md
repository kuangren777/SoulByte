# 项目结构说明

## 📁 目录结构

```
wechat_message_datasets/
├── main.py                     # 🚀 主入口文件
├── README.md                   # 📖 项目文档
├── config.json                 # ⚙️ 配置文件
├── config.example.json         # 📋 示例配置文件
├── requirements.txt            # 📦 依赖列表
├── PROJECT_STRUCTURE.md        # 📁 项目结构说明
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
└── backup_files/               # 🗂️ 备份文件
    ├── process_wechat_data.py  # 原始单体文件
    └── migrate_to_new_version.py # 迁移脚本
```

## 🔧 模块说明

### 主入口 (main.py)
- **功能**: 统一的命令行入口，支持分阶段处理
- **命令**: 
  - `python main.py stage1` - 数据提取和联系人信息
  - `python main.py stage2` - 大模型评估和最终数据集
  - `python main.py contacts` - 联系人管理
  - `python main.py filter` - 缓存筛选
  - `python main.py help` - 帮助信息

### 核心模块 (core/)
- **data_processor.py**: 基础数据处理器
  - 数据加载和清洗
  - 消息类型处理
  - 联系人信息管理
  - 多进程支持

- **main_processor.py**: 主要处理流程
  - 对话回合提取
  - 训练数据格式化
  - 大模型评估集成

### 管理器模块 (managers/)
- **config_manager.py**: 配置管理
  - JSON配置文件解析
  - 配置项验证
  - 默认值处理

- **contact_manager.py**: 联系人管理
  - 联系人信息存储
  - 关系类型管理
  - 搜索和更新功能

- **evaluation_cache.py**: 评估缓存
  - 评估结果缓存
  - 避免重复评估
  - 分数筛选功能

### 工具模块 (utils/)
- **history_utils.py**: 历史记录处理
  - 72小时时间窗口
  - 智能联系人过滤
  - 日期时间格式化
  - 多进程兼容函数

### 评估模块 (evaluators/)
- **llm_evaluator.py**: 大模型评估器
  - API调用管理
  - 评分解析
  - 错误重试机制
  - 缓存集成

## 🔄 数据流程

```
1. 数据加载 (data_processor.py)
   ↓
2. 数据清洗 (data_processor.py)
   ↓
3. 对话提取 (main_processor.py)
   ↓
4. 历史构建 (history_utils.py)
   ↓
5. 训练格式化 (main_processor.py)
   ↓
6. 质量评估 (llm_evaluator.py)
   ↓
7. 结果输出 (main.py)
```

## 🎯 设计原则

### 1. 模块化设计
- 每个模块职责单一
- 接口清晰明确
- 易于测试和维护

### 2. 可扩展性
- 支持新的消息类型
- 支持新的评估器
- 支持新的输出格式

### 3. 性能优化
- 多进程处理支持
- 智能缓存机制
- 内存使用优化

### 4. 用户友好
- 分阶段处理流程
- 详细的进度显示
- 完善的错误处理

## 🔧 扩展指南

### 添加新的消息类型处理
1. 在 `data_processor.py` 的 `process_message_content()` 方法中添加新类型
2. 更新 `message_type_mapping` 字典

### 添加新的评估器
1. 在 `evaluators/` 目录下创建新的评估器文件
2. 继承基础评估器接口
3. 在 `main_processor.py` 中集成

### 添加新的输出格式
1. 在 `main.py` 中添加新的保存方法
2. 支持不同的数据格式（JSON、CSV、XML等）

### 添加新的管理器
1. 在 `managers/` 目录下创建新的管理器
2. 实现标准的管理器接口
3. 在 `__init__.py` 中导出

## 🚀 部署建议

### 开发环境
```bash
# 克隆项目
git clone <repository>
cd wechat_message_datasets

# 安装依赖
pip install -r requirements.txt

# 配置文件
cp config.example.json config.json
# 编辑 config.json
```

### 生产环境
```bash
# 使用虚拟环境
python -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置优化
# 调整 max_workers 根据服务器配置
# 启用多进程处理
# 配置大模型API
```

### Docker部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

## 📊 监控和日志

### 性能监控
- 处理时间统计
- 内存使用监控
- API调用统计
- 缓存命中率

### 日志记录
- 处理进度日志
- 错误异常日志
- 性能指标日志
- 用户操作日志

### 质量指标
- 数据质量评分
- 处理成功率
- 评估准确性
- 用户满意度