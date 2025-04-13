# -*- coding: utf-8 -*-
import os
import json
import uuid
import random
from pathlib import Path
import math
from collections import defaultdict, Counter
import copy # 导入 copy 模块

# --- 配置 ---
BASE_DATA_PATH = Path(r"C:\Users\chy\Desktop\llava_dataset4") # 数据集根目录
OUTPUT_DIR = Path(r"C:\Users\chy\Desktop\llava_dataset4") # 输出JSON文件的目录
RANDOM_SEED = 42 # 随机种子，确保结果可复现
POSITIVE_CATEGORIES = ["dog", "cat", "horse", "porcelain", "ink painting"] # 正样本类别
NUM_CATEGORIES = len(POSITIVE_CATEGORIES) # 类别数量

# --- 文件4 负样本比例配置 ---
CROSS_NEGATIVE_RATIO_F4 = 0.4 # 交叉负样本比例
SIMPLE_NEGATIVE_RATIO_F4 = 0.4 # 简单负样本比例
HARD_NEGATIVE_RATIO_F4 = 0.2  # 难负样本比例

# --- 设置随机种子 ---
random.seed(RANDOM_SEED)

# --- 辅助函数 ---

def get_image_files(folder_path):
    """递归查找指定文件夹下的所有图片文件（常见格式）"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    image_files = []
    if not folder_path.is_dir():
        print(f"警告: 文件夹不存在 {folder_path}")
        return []
    for item in folder_path.rglob('*'):
        if item.is_file() and item.suffix.lower() in image_extensions:
            image_files.append(item)
    return image_files

def format_image_path(image_path, base_path):
    """将绝对路径转换为相对于base_path的、使用正斜杠的相对路径"""
    try:
        relative_path = image_path.relative_to(base_path)
        return relative_path.as_posix()
    except ValueError:
        print(f"错误: 路径 {image_path} 不在基础路径 {base_path} 下。将使用绝对路径。")
        return image_path.as_posix()

def create_sample(image_path_str, question_category, answer, metadata=None):
    """创建单个样本的JSON对象, 可选包含元数据"""
    question_templates = {
        "dog": "Is this an image of a dog?",
        "cat": "Is this an image of a cat?",
        "horse": "Is this an image of a horse?",
        "porcelain": "Is this an image of a porcelain?",
        "ink painting": "Is this an image of an ink painting?",
    }
    if question_category not in question_templates:
        raise ValueError(f"未知的问题类别: {question_category}")
    sample_data = {
        "id": str(uuid.uuid4()),
        "image": image_path_str,
        "conversations": [
            {"from": "human", "value": question_templates[question_category]},
            {"from": "gpt", "value": answer}
        ]
    }
    if metadata:
        sample_data['metadata'] = metadata
    return sample_data

def save_json(data_to_save, filename):
    """
    将数据保存为JSON文件。
    在保存前会清理所有样本中的 'metadata' 键。
    """
    output_path = OUTPUT_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(data_to_save, list):
        print(f"警告：传递给 save_json 的数据不是列表，类型为 {type(data_to_save)}。将尝试转换。")
        try: data_to_save = list(data_to_save)
        except TypeError: print(f"错误：无法将数据转换为列表以保存到 {filename}。"); return

    # --- 在这里执行清理 ---
    print(f"正在清理 {filename} 中样本的 metadata...")
    cleaned_data = []
    metadata_found_count = 0
    for sample in data_to_save:
        sample_copy = sample.copy()
        if 'metadata' in sample_copy:
            del sample_copy['metadata']
            metadata_found_count += 1
        cleaned_data.append(sample_copy)

    if metadata_found_count > 0:
        print(f"  已从 {metadata_found_count} 个样本中移除 metadata 键。")
    else:
        print("  未在样本中找到需要移除的 metadata 键。")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    print(f"成功生成文件: {output_path}，包含 {len(cleaned_data)} 个样本 (已清理 metadata)。")


def get_question_category(sample):
    """从样本中提取问题类别"""
    question = sample["conversations"][0]["value"]
    for category in POSITIVE_CATEGORIES:
        expected_question = create_sample("dummy", category, "Yes")["conversations"][0]["value"]
        if question == expected_question: return category
    for category in POSITIVE_CATEGORIES:
        if category.lower() in question.lower(): return category
    return "unknown"

def get_source_category(sample):
    """优先从样本的元数据获取真实的来源类别"""
    if 'metadata' in sample and 'true_source_category' in sample['metadata']:
        return sample['metadata']['true_source_category']
    return "unknown_source"

def verify_balance(samples, description):
    """验证数据集中的问题类别是否绝对均衡"""
    if not samples: print(f"{description}: 数据集为空。"); return True
    question_counts = Counter(get_question_category(s) for s in samples)
    total_samples = len(samples)
    if total_samples == 0: print(f"{description}: 数据集样本总数为 0。"); return True
    expected_count_float = total_samples / NUM_CATEGORIES if NUM_CATEGORIES > 0 else 0
    is_balanced = True
    print(f"\n--- {description} 问题类别平衡性验证 ---")
    print(f"总样本数: {total_samples}")
    can_be_perfectly_balanced = math.isclose(total_samples % NUM_CATEGORIES, 0) if NUM_CATEGORIES > 0 else True
    if not can_be_perfectly_balanced: print(f"警告: 总样本数 {total_samples} 不能被类别数 {NUM_CATEGORIES} 整除。")
    print(f"期望每个问题类别的样本数: {expected_count_float:.2f}")
    print("实际分布:")
    max_diff = 0
    for category in POSITIVE_CATEGORIES:
        count = question_counts.get(category, 0)
        print(f"  问题类别 '{category}': {count} 个样本")
        diff = abs(count - expected_count_float)
        max_diff = max(max_diff, diff)
        if diff > 1.001: is_balanced = False; print(f"    -> 不平衡!")
    if not can_be_perfectly_balanced and max_diff <= 1.001: print("注意: 因总数无法整除，存在最多1个样本差异，可接受。"); is_balanced = True
    if is_balanced: print("结论: 问题类别分布足够均衡。")
    else: print("结论: 问题类别分布不均衡！")
    print("-" * (len(description) + 20))
    return is_balanced

def generate_balanced_negatives(
        target_total_negatives, negative_type, positive_images_map,
        simple_negative_images_list, hard_negative_images_map, base_data_path
    ):
    """生成均衡负样本（内部逻辑不变，仍然为 cross 类型生成 metadata）"""
    generated_negatives = []
    if target_total_negatives <= 0: return generated_negatives
    base_count_per_question = target_total_negatives // NUM_CATEGORIES if NUM_CATEGORIES > 0 else 0
    remainder_question = target_total_negatives % NUM_CATEGORIES if NUM_CATEGORIES > 0 else 0
    target_counts_per_question_category={c:base_count_per_question for c in POSITIVE_CATEGORIES}
    if remainder_question > 0:
        cats_add_q = random.sample(POSITIVE_CATEGORIES, remainder_question)
        for cat in cats_add_q: target_counts_per_question_category[cat] += 1
    print(f"  为 '{negative_type}' 类型生成负样本，各问题类别目标数: {target_counts_per_question_category}")

    if negative_type == 'cross':
        print("  为 'cross' 类型负样本进行来源类别均衡处理 (确保全局来源图片唯一)...")
        cross_negatives_list = []
        globally_used_cross_source_images = set()
        requests = []
        for q_cat, total_needed in target_counts_per_question_category.items():
            if total_needed == 0: continue
            others=[c for c in POSITIVE_CATEGORIES if c!=q_cat]; num_others=len(others)
            if num_others == 0: continue
            base_s=total_needed//num_others if num_others>0 else 0; rem_s=total_needed%num_others if num_others>0 else 0
            target_s_counts={s:base_s for s in others}
            if rem_s>0: sources_add_s=random.sample(others,rem_s); [target_s_counts.update({s:target_s_counts[s]+1}) for s in sources_add_s]
            for s_cat, needed_s in target_s_counts.items():
                 if needed_s > 0: requests.append({'q_cat': q_cat, 's_cat': s_cat, 'needed': needed_s})
        random.shuffle(requests)
        for req in requests:
            q_cat, s_cat, needed = req['q_cat'], req['s_cat'], req['needed']
            all_s_paths = positive_images_map.get(s_cat, [])
            if not all_s_paths: print(f"警告: 来源 '{s_cat}' 无图片。"); continue
            available_s_paths = [p for p in all_s_paths if p not in globally_used_cross_source_images]
            if not available_s_paths: print(f"警告: 来源 '{s_cat}' 图片已用完 (for Q:{q_cat})。"); continue
            num_take = min(needed, len(available_s_paths))
            if num_take < needed: print(f"警告: 来源 '{s_cat}' 唯一图片({len(available_s_paths)})<需({needed})for Q:{q_cat}。取{num_take}。")
            selected_paths = random.sample(available_s_paths, num_take)
            for img_path in selected_paths:
                try:
                    rel_path=format_image_path(img_path, base_data_path)
                    meta={'true_source_category': s_cat}; sample=create_sample(rel_path, q_cat, "No", metadata=meta)
                    cross_negatives_list.append(sample); globally_used_cross_source_images.add(img_path)
                except Exception as e: print(f"创建交叉样本出错:{e}(Q:{q_cat},S:{s_cat},P:{img_path})")
        generated_negatives = cross_negatives_list
        print(f"  'cross' 类型生成完成，共 {len(generated_negatives)} 个样本 (目标: {target_total_negatives})。")
        print(f"  共使用 {len(globally_used_cross_source_images)} 张独特来源图片。")

    elif negative_type == 'simple':
        print("  为 'simple' 类型负样本进行处理 (确保全局来源图片唯一)...")
        simple_negatives_list = []
        globally_used_simple_negative_images = set()
        requests = []
        for q_cat, needed in target_counts_per_question_category.items():
            if needed > 0: requests.append({'q_cat': q_cat, 'needed': needed})
        random.shuffle(requests)
        all_simple_paths = list(simple_negative_images_list)
        if not all_simple_paths and target_total_negatives > 0: print("警告:'ez_negative'为空?"); return []
        for req in requests:
            q_cat, needed = req['q_cat'], req['needed']
            available_simple_paths=[p for p in all_simple_paths if p not in globally_used_simple_negative_images]
            if not available_simple_paths: print(f"警告:简单负样本已用完(for Q:{q_cat})。"); continue
            num_take = min(needed, len(available_simple_paths))
            if num_take < needed: print(f"警告:简单负样本唯一图片({len(available_simple_paths)})<需({needed})for Q:{q_cat}。取{num_take}。")
            selected_paths = random.sample(available_simple_paths, num_take)
            for img_path in selected_paths:
                try:
                    rel_path = format_image_path(img_path, base_data_path)
                    sample = create_sample(rel_path, q_cat, "No")
                    simple_negatives_list.append(sample); globally_used_simple_negative_images.add(img_path)
                except Exception as e: print(f"创建简单样本出错:{e}(Q:{q_cat},P:{img_path})")
        generated_negatives = simple_negatives_list
        print(f"  'simple' 类型生成完成，共 {len(generated_negatives)} 个样本 (目标: {target_total_negatives})。")
        print(f"  共使用 {len(globally_used_simple_negative_images)} 张独特简单负样本图片。")

    else: # negative_type == 'hard'
        print(f"  为 '{negative_type}' 类型生成负样本 (局部追踪)...")
        used_image_identifiers_local = set()
        for q_cat, target_count in target_counts_per_question_category.items():
            if target_count == 0: continue
            generated_count=0; attempts=0; max_attempts=target_count*10+200
            pool = hard_negative_images_map.get(q_cat, [])
            if not pool: print(f"警告: 问题 '{q_cat}' 的 'hard' 图片池为空。"); continue
            indices = list(range(len(pool))); random.shuffle(indices); ptr = 0
            while generated_count < target_count and attempts < max_attempts:
                attempts += 1
                if ptr >= len(indices): print(f"信息: 问题 '{q_cat}' 的 '{negative_type}' 图片池耗尽。"); break
                idx = indices[ptr]; img_path = pool[idx]; identifier = (str(img_path), q_cat)
                if identifier in used_image_identifiers_local: ptr += 1; continue
                try:
                    rel_path = format_image_path(img_path, base_data_path)
                    sample = create_sample(rel_path, q_cat, "No")
                    generated_negatives.append(sample); used_image_identifiers_local.add(identifier)
                    generated_count += 1; ptr += 1
                except Exception as e: print(f"生成{negative_type}样本出错:{e}(Q:{q_cat},P:{img_path})"); ptr += 1
            if generated_count < target_count: print(f"警告: 类别 '{q_cat}' 的 '{negative_type}' 仅生成 {generated_count}/{target_count} 个。")
        print(f"  实际生成 {len(generated_negatives)} 个 '{negative_type}' 样本。")

    random.shuffle(generated_negatives)
    return generated_negatives

# --- select_balanced_subset 保持不变，依赖 get_source_category 读取 metadata ---
def select_balanced_subset(samples_pool, target_count, sample_type='unknown'):
    """从样本池中选择均衡子集。'cross'类型会尝试保持来源均衡。"""
    if not samples_pool: print("警告: select_balanced_subset 收到空样本池。"); return []
    if target_count <= 0: return []
    if target_count >= len(samples_pool):
        print(f"信息: 目标数量 ({target_count}) >= 样本池大小 ({len(samples_pool)})，返回整个池。")
        shuffled_pool = copy.deepcopy(samples_pool); random.shuffle(shuffled_pool); return shuffled_pool

    print(f"\n  开始为 '{sample_type}' 类型进行均衡子集抽样 (目标: {target_count})...")
    target_counts_per_q_cat = {}
    base_count_q = target_count // NUM_CATEGORIES if NUM_CATEGORIES > 0 else 0
    remainder_q = target_count % NUM_CATEGORIES if NUM_CATEGORIES > 0 else 0
    target_counts_per_q_cat = {cat: base_count_q for cat in POSITIVE_CATEGORIES}
    if remainder_q > 0:
        cats_in_pool=set(get_question_category(s) for s in samples_pool if get_question_category(s)!='unknown')
        if not cats_in_pool: print("警告:抽样池中无有效问题类别"); return []
        cats_add_q=random.sample(list(cats_in_pool), min(remainder_q,len(cats_in_pool)))
        if len(cats_add_q)<remainder_q: cats_add_q.extend(random.choices(list(cats_in_pool),k=remainder_q-len(cats_add_q)))
        for cat in cats_add_q[:remainder_q]: target_counts_per_q_cat[cat]=target_counts_per_q_cat.get(cat,0)+1
    # print(f"    子集目标问题类别分布: {target_counts_per_q_cat}") # 减少打印

    selected_samples = [] # 这里存储的是带 metadata 的样本（如果是 cross）
    grouped_by_q_cat = defaultdict(list)
    for sample in samples_pool:
        q_cat = get_question_category(sample)
        if q_cat != "unknown": grouped_by_q_cat[q_cat].append(sample)

    for q_cat, needed_q_count in target_counts_per_q_cat.items():
        if needed_q_count == 0: continue
        available_samples_for_q = grouped_by_q_cat.get(q_cat, [])
        if not available_samples_for_q: print(f"警告:子集抽样时，问题'{q_cat}'在源池无样本。"); continue
        # print(f"    处理问题 '{q_cat}', 需 {needed_q_count} (源池有 {len(available_samples_for_q)})...") # 减少打印

        if sample_type == 'cross':
            # print(f"      执行 'cross' 类型来源均衡抽样...") # 减少打印
            grouped_by_s_cat = defaultdict(list)
            possible_s_cats = [c for c in POSITIVE_CATEGORIES if c != q_cat]; num_s_cats = len(possible_s_cats)
            if num_s_cats == 0:
                num_take = min(needed_q_count, len(available_samples_for_q))
                selected_samples.extend(random.sample(available_samples_for_q, num_take)); continue

            unknown_s_count = 0
            for sample in available_samples_for_q:
                s_cat = get_source_category(sample) # <-- Still uses metadata here
                if s_cat in possible_s_cats: grouped_by_s_cat[s_cat].append(sample)
                elif s_cat == "unknown_source": unknown_s_count += 1
            # if unknown_s_count > 0: print(f"警告:抽样问题'{q_cat}'时遇到{unknown_s_count}个未知来源样本") # 减少打印

            target_s_counts = {}; base_s = needed_q_count // num_s_cats if num_s_cats > 0 else 0
            rem_s = needed_q_count % num_s_cats if num_s_cats > 0 else 0
            target_s_counts = {s: base_s for s in possible_s_cats}
            if rem_s > 0:
                s_cats_in_group = list(grouped_by_s_cat.keys())
                if s_cats_in_group:
                    s_add_s = random.sample(s_cats_in_group, min(rem_s, len(s_cats_in_group)))
                    if len(s_add_s)<rem_s: s_add_s.extend(random.choices(s_cats_in_group,k=rem_s-len(s_add_s)))
                    for s_cat in s_add_s[:rem_s]: target_s_counts[s_cat]=target_s_counts.get(s_cat,0)+1
            # print(f"        目标来源分布: {target_s_counts}") # 减少打印

            selected_for_q = []
            for s_cat, needed_s_count in target_s_counts.items():
                 if needed_s_count == 0: continue
                 available_s_samples = grouped_by_s_cat.get(s_cat, [])
                 if not available_s_samples: print(f"警告:问题'{q_cat}'来源'{s_cat}'在可用池无样本。"); continue
                 num_take_s = min(needed_s_count, len(available_s_samples))
                 if num_take_s < needed_s_count: print(f"警告:问题'{q_cat}',来源'{s_cat}':可用({len(available_s_samples)})<需({needed_s_count})。取{num_take_s}。")
                 selected_s = random.sample(available_s_samples, num_take_s)
                 selected_for_q.extend(selected_s)
                 # print(f"          从来源 '{s_cat}' 抽取 {len(selected_s)} 个。") # 减少打印
            # print(f"      问题 '{q_cat}' 实际抽取 {len(selected_for_q)} 个 cross 样本。") # 减少打印
            selected_samples.extend(selected_for_q)
        else: # simple, hard, unknown
             num_take = min(needed_q_count, len(available_samples_for_q))
             if num_take < needed_q_count: print(f"警告:问题'{q_cat}'({sample_type})可用({len(available_samples_for_q)})<需({needed_q_count})。取{num_take}。")
             selected_for_q = random.sample(available_samples_for_q, num_take)
             # print(f"      问题 '{q_cat}' ({sample_type}) 随机抽取 {len(selected_for_q)} 个。") # 减少打印
             selected_samples.extend(selected_for_q)

    print(f"  均衡子集抽样完成，实际选择 {len(selected_samples)} 个样本。")
    random.shuffle(selected_samples)
    # 返回的 selected_samples 中，交叉负样本仍然包含 metadata
    return selected_samples

# --- verify_cross_negative_source_balance 保持不变，依赖 get_source_category 读取 metadata ---
def verify_cross_negative_source_balance(cross_negative_samples, description):
    """验证交叉负样本的来源类别是否均衡 (基于元数据)"""
    print(f"\n--- {description} 交叉负样本来源均衡性验证 ---")
    if not cross_negative_samples: print("样本列表为空。"); return

    grouped_by_question = defaultdict(list)
    valid_cross_samples = 0
    cross_samples_with_source = [] # 存储有效的交叉负样本
    for sample in cross_negative_samples:
        source_cat_check = get_source_category(sample)
        if source_cat_check != "unknown_source":
            q_cat = get_question_category(sample)
            if q_cat != "unknown":
                grouped_by_question[q_cat].append(sample)
                valid_cross_samples += 1
                cross_samples_with_source.append(sample)

    if valid_cross_samples == 0: print("未找到有效的交叉负样本进行来源验证。"); return

    print(f"对 {valid_cross_samples} 个有效交叉负样本进行验证...")
    overall_source_balanced = True
    for q_cat, samples_for_q in grouped_by_question.items():
        print(f"  问题类别 '{q_cat}' (共 {len(samples_for_q)} 个样本):")
        source_counts = Counter()
        possible_s_cats=[c for c in POSITIVE_CATEGORIES if c != q_cat]; num_s_cats = len(possible_s_cats)
        if num_s_cats == 0: continue
        expected_s_count = len(samples_for_q) / num_s_cats if num_s_cats > 0 else 0
        unknown_s_count = 0
        for sample in samples_for_q:
            s_cat = get_source_category(sample)
            if s_cat in possible_s_cats: source_counts[s_cat] += 1
            else: unknown_s_count += 1
        print(f"    期望每个来源类别的样本数: {expected_s_count:.2f}")
        print(f"    实际来源分布:")
        is_q_cat_balanced = True; max_diff = 0
        for src_cat in possible_s_cats:
            count = source_counts[src_cat]
            diff = abs(count - expected_s_count)
            max_diff = max(max_diff, diff)
            print(f"      来源 '{src_cat}': {count} 个样本")
            if diff > 1.001: is_q_cat_balanced = False; print(f"        -> 不平衡!")
        if unknown_s_count > 0: print(f"      来源 'unknown_source': {unknown_s_count} 个"); is_q_cat_balanced = False
        can_be_perfectly_s_balanced = math.isclose(len(samples_for_q)%num_s_cats, 0) if num_s_cats>0 else True
        if not can_be_perfectly_s_balanced and max_diff<=1.001 and unknown_s_count==0: print("    注意: 因数无法整除，最多1个差异，可接受。"); is_q_cat_balanced=True
        if not is_q_cat_balanced: overall_source_balanced = False; print(f"    结论: 问题 '{q_cat}' 来源不均衡。")
        else: print(f"    结论: 问题 '{q_cat}' 来源足够均衡。")
    if overall_source_balanced: print("总体结论: 交叉负样本来源分布足够均衡。")
    else: print("总体结论: 交叉负样本来源分布存在不均衡！")
    print("-" * (len(description) + 30))


# --- 主脚本开始 ---

# --- 1. 数据收集 ---
print("开始收集图片文件路径...")
positive_images = {}
for category in POSITIVE_CATEGORIES:
    folder_path = BASE_DATA_PATH / category; files = get_image_files(folder_path)
    if not files: print(f"警告: 正样本 '{category}' 未找到图片！")
    positive_images[category] = files; print(f"找到 {len(files)} 张 '{category}' 正样本图片.")
min_samples = min((len(paths) for paths in positive_images.values() if paths), default=0)
if min_samples == 0: raise ValueError("至少一个正样本类别图片数为0或未找到。")
print(f"\n将使用每个正样本类别 {min_samples} 张图片确保均衡。")
ACTUAL_TOTAL_POSITIVE_SAMPLES = min_samples * NUM_CATEGORIES
TARGET_TOTAL_NEGATIVES_PER_FILE = ACTUAL_TOTAL_POSITIVE_SAMPLES

# --- 准备正样本列表 ---
positive_samples_base = []
for category, images in positive_images.items():
     if len(images) >= min_samples:
          sampled = random.sample(images, min_samples)
          for img in sampled: positive_samples_base.append(create_sample(format_image_path(img, BASE_DATA_PATH), category, "Yes"))
print(f"已准备好 {len(positive_samples_base)} 个均衡正样本。")
random.shuffle(positive_samples_base)

# --- 收集负样本图片路径 ---
ez_folder = BASE_DATA_PATH / "ez_negative"; simple_neg_list = get_image_files(ez_folder)
print(f"找到 {len(simple_neg_list)} 张简单负样本图片.")
hard_neg_map = {}
for cat in POSITIVE_CATEGORIES:
    folder = BASE_DATA_PATH / f"{cat}_negative"; files = get_image_files(folder)
    if not files: print(f"警告: 难负样本 '{cat}_negative' 未找到图片。")
    hard_neg_map[cat] = files; print(f"找到 {len(files)} 张 '{cat}_negative' 难负样本图片.")
print("图片文件路径收集完成。\n")

# --- 2. 生成文件1：纯正样本 ---
print("--- 开始生成 文件1：纯正样本 ---")
# 文件1 不包含 metadata，直接保存
save_json(positive_samples_base, "llava_dataset1.json")
print(f"文件1 正样本数: {len(positive_samples_base)}")
verify_balance(positive_samples_base, "文件1 (纯正样本)")
print("-" * 30 + "\n")

# --- 3. 预生成均衡的负样本池 ---
print("--- 开始预生成均衡的负样本池 ---")
# --- 交叉负样本池 (生成时带 metadata) ---
print("生成交叉负样本池...")
target_cross_pool_size = TARGET_TOTAL_NEGATIVES_PER_FILE
all_balanced_cross_negatives_with_meta = generate_balanced_negatives(
    target_total_negatives=target_cross_pool_size, negative_type='cross',
    positive_images_map=positive_images, simple_negative_images_list=[],
    hard_negative_images_map={}, base_data_path=BASE_DATA_PATH
)
print(f"交叉负样本池生成完毕(带metadata)，包含 {len(all_balanced_cross_negatives_with_meta)} 个样本。")
# 验证时使用带 metadata 的池
verify_balance(all_balanced_cross_negatives_with_meta, "交叉负样本池 (问题均衡性)")
verify_cross_negative_source_balance(all_balanced_cross_negatives_with_meta, "交叉负样本池 (来源均衡性)")

# --- 简单负样本池 (生成时不带 metadata) ---
target_simple_f3 = TARGET_TOTAL_NEGATIVES_PER_FILE // 2
# --- !! 修正：在这里计算文件4的负样本数量 !! ---
num_negatives_file4 = TARGET_TOTAL_NEGATIVES_PER_FILE # 文件4目标负样本总数
num_cross_f4 = math.floor(num_negatives_file4 * CROSS_NEGATIVE_RATIO_F4)
num_simple_f4_calc = math.floor(num_negatives_file4 * SIMPLE_NEGATIVE_RATIO_F4)
num_hard_f4_calc = num_negatives_file4 - num_cross_f4 - num_simple_f4_calc # 剩余给难负样本
if num_hard_f4_calc < 0: num_hard_f4_calc = 0
if (num_cross_f4 + num_simple_f4_calc + num_hard_f4_calc) != num_negatives_file4:
     print(f"警告: 文件4 负样本目标数计算有误! ({num_cross_f4}+{num_simple_f4_calc}+{num_hard_f4_calc} != {num_negatives_file4})")
# ------------------------------------------
target_simple_pool_size = max(target_simple_f3, num_simple_f4_calc) # 使用计算出的F4简单数
print(f"\n生成简单负样本池 (最大量: {target_simple_pool_size})...")
all_balanced_simple_negatives = generate_balanced_negatives(
    target_total_negatives=target_simple_pool_size, negative_type='simple',
    positive_images_map={}, simple_negative_images_list=simple_neg_list,
    hard_negative_images_map={}, base_data_path=BASE_DATA_PATH
)
print(f"简单负样本池生成完毕，包含 {len(all_balanced_simple_negatives)} 个样本。")
verify_balance(all_balanced_simple_negatives, "简单负样本池")

# --- 难负样本池 (生成时不带 metadata) ---
target_hard_pool_size = num_hard_f4_calc # 使用计算出的F4困难数
print(f"\n生成难负样本池 (量: {target_hard_pool_size})...")
all_balanced_hard_negatives = generate_balanced_negatives(
    target_total_negatives=target_hard_pool_size, negative_type='hard',
    positive_images_map={}, simple_negative_images_list=[],
    hard_negative_images_map=hard_neg_map, base_data_path=BASE_DATA_PATH
)
print(f"难负样本池生成完毕，包含 {len(all_balanced_hard_negatives)} 个样本。")
verify_balance(all_balanced_hard_negatives, "难负样本池")
print("-" * 30 + "\n")


# --- 4. 生成文件2：正样本 + 交叉负样本 ---
print("--- 开始生成 文件2：正样本 + 交叉负样本 ---")
# 合并样本列表 (交叉负样本此时仍包含 metadata)
samples_file2_to_save = positive_samples_base + all_balanced_cross_negatives_with_meta
random.shuffle(samples_file2_to_save)
# --- 在保存前调用 save_json 清理 metadata ---
save_json(samples_file2_to_save, "llava_dataset2.json")
print(f"文件2 正样本数: {len(positive_samples_base)}")
print(f"文件2 交叉负样本数: {len(all_balanced_cross_negatives_with_meta)}")
print(f"文件2 总样本数: {len(samples_file2_to_save)}")
# 验证时使用带 metadata 的列表
verify_balance(samples_file2_to_save, "文件2 (问题均衡性)")
verify_cross_negative_source_balance(all_balanced_cross_negatives_with_meta, "文件2 交叉负样本 (来源均衡性)")
print("-" * 30 + "\n")


# --- 5. 生成文件3：正样本 + (子集)交叉负样本 + (子集)简单负样本 ---
print("--- 开始生成 文件3：加入简单负样本 ---")
num_negatives_file3 = TARGET_TOTAL_NEGATIVES_PER_FILE
num_cross_f3 = num_negatives_file3 // 2; num_simple_f3 = num_negatives_file3 - num_cross_f3
print(f"文件3 需要 {num_cross_f3} 交叉负样本 和 {num_simple_f3} 简单负样本。")

print("从交叉负样本池选择子集...")
# 从带 metadata 的池中采样，返回的子集也带 metadata
selected_cross_f3_with_meta = select_balanced_subset(all_balanced_cross_negatives_with_meta, num_cross_f3, sample_type='cross')
print("从简单负样本池选择子集...")
selected_simple_f3 = select_balanced_subset(all_balanced_simple_negatives, num_simple_f3, sample_type='simple')

# 合并样本列表 (交叉负样本子集此时仍包含 metadata)
samples_file3_to_save = positive_samples_base + selected_cross_f3_with_meta + selected_simple_f3
random.shuffle(samples_file3_to_save)
# --- 在保存前调用 save_json 清理 metadata ---
save_json(samples_file3_to_save, "llava_dataset3.json")
print(f"文件3 正样本数: {len(positive_samples_base)}")
print(f"文件3 交叉负样本数: {len(selected_cross_f3_with_meta)}")
print(f"文件3 简单负样本数: {len(selected_simple_f3)}")
print(f"文件3 总样本数: {len(samples_file3_to_save)}")
# 验证时使用带 metadata 的子集
verify_balance(samples_file3_to_save, "文件3 (问题均衡性)")
verify_cross_negative_source_balance(selected_cross_f3_with_meta, "文件3 交叉负样本子集 (来源均衡性)")
print("-" * 30 + "\n")


# --- 6. 生成文件4：正样本 + 按比例混合负样本子集 ---
print("--- 开始生成 文件4：加入难负样本 (使用可调比例) ---")
# --- !! 确保这些变量已在前面计算过 !! ---
# num_negatives_file4, num_cross_f4, num_simple_f4_calc, num_hard_f4_calc
# ------------------------------------------
print(f"文件4 配置比例: Cross={CROSS_NEGATIVE_RATIO_F4:.2f}, Simple={SIMPLE_NEGATIVE_RATIO_F4:.2f}, Hard={HARD_NEGATIVE_RATIO_F4:.2f}")
print(f"文件4 目标总负样本数: {num_negatives_file4}") # 使用已定义的变量
print(f"文件4 计算数量: Cross={num_cross_f4}, Simple={num_simple_f4_calc}, Hard={num_hard_f4_calc}") # 使用已定义的变量

print("从交叉负样本池选择文件4子集...")
# 从带 metadata 的池中采样
selected_cross_f4_with_meta = select_balanced_subset(all_balanced_cross_negatives_with_meta, num_cross_f4, sample_type='cross')
print("从简单负样本池选择文件4子集...")
selected_simple_f4 = select_balanced_subset(all_balanced_simple_negatives, num_simple_f4_calc, sample_type='simple') # 使用 num_simple_f4_calc
print(f"直接使用已生成的难负样本池 ({len(all_balanced_hard_negatives)} 个样本)。")
selected_hard_f4 = all_balanced_hard_negatives # 难负样本池不含 metadata

# 合并样本列表 (交叉负样本子集此时仍包含 metadata)
samples_file4_to_save = positive_samples_base + selected_cross_f4_with_meta + selected_simple_f4 + selected_hard_f4
random.shuffle(samples_file4_to_save)
# --- 在保存前调用 save_json 清理 metadata ---
save_json(samples_file4_to_save, "llava_dataset4.json")

actual_cross_f4=len(selected_cross_f4_with_meta); actual_simple_f4=len(selected_simple_f4); actual_hard_f4=len(selected_hard_f4)
actual_total_neg_f4 = actual_cross_f4 + actual_simple_f4 + actual_hard_f4

print(f"\n--- 文件4 最终构成 ---")
print(f"正样本数: {len(positive_samples_base)}")
# 报告数量使用带 meta 的子集长度或原始池长度
print(f"交叉负样本数: {actual_cross_f4} (目标: {num_cross_f4})")
print(f"简单负样本数: {actual_simple_f4} (目标: {num_simple_f4_calc})")
print(f"难负样本数: {actual_hard_f4} (目标: {num_hard_f4_calc})")
print(f"总负样本数: {actual_total_neg_f4} (目标: {num_negatives_file4})")
print(f"文件4 总样本数: {len(samples_file4_to_save)}")
if actual_total_neg_f4 != num_negatives_file4: print(f"注意：最终负样本总数({actual_total_neg_f4})与目标({num_negatives_file4})可能因源图片不足或抽样警告不符。")

# 验证时使用带 metadata 的子集
verify_balance(samples_file4_to_save, "文件4 (问题均衡性)")
verify_cross_negative_source_balance(selected_cross_f4_with_meta, "文件4 交叉负样本子集 (来源均衡性)")
print("-" * 30 + "\n")

print("所有JSON文件生成完毕！")