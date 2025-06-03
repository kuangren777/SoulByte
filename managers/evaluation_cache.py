#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import hashlib
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class EvaluationCache:
    """评估结果缓存管理器"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.cache_file = os.path.join(output_dir, "evaluation_cache.json")
        self.cache = self._load_cache()
        # 添加线程锁保护缓存访问
        self._lock = threading.Lock()
    
    def _load_cache(self) -> Dict:
        """加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载评估缓存失败: {e}")
        return {}
    
    def _generate_key(self, sample: Dict) -> str:
        """生成样本的唯一键"""
        content = f"{sample['instruction']}|{sample['input']}|{sample['output']}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_score(self, sample: Dict) -> Optional[Tuple[float, str]]:
        """获取缓存的评分"""
        key = self._generate_key(sample)
        with self._lock:
            if key in self.cache:
                cached_data = self.cache[key]
                return cached_data['score'], cached_data['response']
        return None
    
    def save_score(self, sample: Dict, score: float, response: str) -> None:
        """保存评分到缓存"""
        key = self._generate_key(sample)
        cache_entry = {
            'score': score,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'sample': sample
        }
        
        with self._lock:
            self.cache[key] = cache_entry
            # 立即保存到文件
            self._save_cache_unsafe()
    
    def _save_cache_unsafe(self) -> None:
        """保存缓存到文件（不加锁，内部使用）"""
        try:
            # 创建缓存的副本以避免在序列化过程中被修改
            cache_copy = self.cache.copy()
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_copy, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存评估缓存失败: {e}")
    
    def _save_cache(self) -> None:
        """保存缓存到文件（公共接口）"""
        with self._lock:
            self._save_cache_unsafe()
    
    def filter_by_score(self, min_score: float) -> List[Dict]:
        """根据最小分数筛选样本"""
        filtered_samples = []
        stats = {
            'total': len(self.cache),
            'passed': 0,
            'failed': 0,
            'score_distribution': {}
        }
        
        # 分数分布统计初始化
        for score_range in [(0,2), (2,4), (4,6), (6,8), (8,10)]:
            stats['score_distribution'][f"{score_range[0]}-{score_range[1]}"] = 0
        
        with self._lock:
            for key, entry in self.cache.items():
                score = entry.get('score', 0)
                sample = entry.get('sample')
                
                # 统计分数分布
                for min_r, max_r in [(0,2), (2,4), (4,6), (6,8), (8,10)]:
                    if min_r <= score < max_r:
                        stats['score_distribution'][f"{min_r}-{max_r}"] += 1
                
                if score >= min_score and sample:
                    filtered_samples.append(sample)
                    stats['passed'] += 1
                else:
                    stats['failed'] += 1
        
        print(f"从缓存中筛选：总计 {stats['total']} 条数据，通过 {stats['passed']} 条，未通过 {stats['failed']} 条")
        print("分数分布:")
        for range_name, count in stats['score_distribution'].items():
            percent = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {range_name}: {count} ({percent:.1f}%)")
        
        return filtered_samples
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        with self._lock:
            return len(self.cache)
    
    def clear_cache(self) -> None:
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self._save_cache_unsafe()
        print("评估缓存已清空") 