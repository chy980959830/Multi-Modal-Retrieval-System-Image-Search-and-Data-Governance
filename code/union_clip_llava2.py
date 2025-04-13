import os
import time

import numpy as np
import torch
import clip
from PIL import Image
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

#from llavarun import eval_model


# 更改huggingface的缓存默认目录
import os
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['TRANSFORMERS_OFFLINE'] = '0' 

# 学术加速
import subprocess
import os

result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

def get_llava_model():
    model_path = "liuhaotian/llava-v1.5-7b"

    llava_tokenizer, llava_model, llava_image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        load_8bit=True
    )
    return llava_tokenizer,llava_model,llava_image_processor, context_len


def get_chinese_clip_model(chinese_prompt, device):
    query_texts = [f"{chinese_prompt}"]
    text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
    text_encoder = BertForSequenceClassification.from_pretrained(
        "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval()
    text = text_tokenizer(query_texts, return_tensors='pt', padding=True)['input_ids']

    # 加载CLIP的image encoder
    chinese_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    chinese_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # image = processor(images=Image.open(requests.get(url, stream=True).raw), return_tensors="pt")

    # 放到GPU
    chinese_model = chinese_model.to(device)
    text_encoder = text_encoder.to(device)
    text = text.to(device)

    return chinese_model, chinese_processor, text_encoder, text


def get_english_clip_model(device):
    english_model, english_preprocess = clip.load("ViT-B/32", device=device)
    return english_model, english_preprocess


def get_chinese_score(url, chinese_processor, chinese_model, text_encoder, text, device):
    image = chinese_processor(images=Image.open(url), return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = chinese_model.get_image_features(**image)
        text_features = text_encoder(text).logits
        # 归一化
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        # 计算余弦相似度 logit_scale是尺度系数
        logit_scale = chinese_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        # 返回相似度的计算值
        return logits_per_image.cpu().numpy()[0]


def get_english_score(url, english_model, english_preprocess, english_prompt, device):
    image = english_preprocess(Image.open(url)).unsqueeze(0).to(device)
    # 只有一种类别名称
    text = clip.tokenize([f"{english_prompt}"]).to(device)

    with torch.no_grad():
        image_features = english_model.encode_image(image)
        text_features = english_model.encode_text(text)

        # 相似度计算
        logits_per_image, logits_per_text = english_model(image, text)
        return logits_per_image.cpu().numpy()[0]


def get_llava_result(image_url,prompt,llava_tokenizer, llava_model, llava_image_processor, context_len):

    # prompt = f"请问这张图片里描述的是{prompt}吗,请回答yes或no,不要包含其它输出"
    prompt = "Is this picture of a chinese porcelain? If yes, answer yes. If no, answer no"

    url = image_url
    model_path = "liuhaotian/llava-v1.5-7b"

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": url,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 500
    })()

    output = eval_model(args)
    if "yes" in output.lower():
        return True
    else:
        return False



def union_clip_by_threshold(image_folder, positive_cate, chinese_prompt, english_prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 获取模型
    chinese_model, chinese_processor, text_encoder, text = get_chinese_clip_model(chinese_prompt, device)
    english_model, english_preprocess = get_english_clip_model(device)
    llava_tokenizer,llava_model,llava_image_processor, context_len = get_llava_model()


    cate_name_list = os.listdir(image_folder)
    pos_res = []
    neg_res = []

    # 判断是不是正样本的类别
    cat_tables = {"T-shirt": "T恤",
                  "guitar": "吉他",
                  "badminton-racket": "羽毛球拍",
                  "baozi": "包子",
                  "lychee": "荔枝",
                  }
    english_threshold = {"T-shirt": 25.61,
                         "guitar": 25.22,
                         "badminton-racket": 27.48,
                         "baozi": 28.37,
                         "lychee": 27.63}
    chinese_threshold = {"T恤": 8.89,
                         "吉他": 11.28,
                         "羽毛球拍": 14.8,
                         "包子": 15.19,
                         "荔枝": 10.38}
    # 时间记录
    english_clip_time = 0. # 英文clip推理时间
    chinese_clip_time = 0. # 中文clip推理时间
    clip_time = 0. # 英文 + 中文 clip推理时间
    llava_time = 0. # llava的推理时间
    clip_llava_time = 0. # 英文 + 中文 clip + llava 推理时间

    llava_image_nums = 0
    # 遍历每一个文件夹
    for cate_name in cate_name_list:
        # 判断这个文件夹是不是正样本
        if cate_name == positive_cate:
            image_list = os.listdir(os.path.join(image_folder, cate_name))
            for item in image_list:
                url = os.path.join(image_folder, cate_name, item)
                # 每一张图片做处理
                # 分别获取中文和英文的得分，并计算时间
                start_chinese = time.time()
                chinese_score = get_chinese_score(url, chinese_processor, chinese_model, text_encoder, text, device)
                end_chinese = time.time()

                start_english = time.time()
                english_score = get_english_score(url, english_model, english_preprocess, english_prompt, device)
                end_english = time.time()

                # 取并集 中文clip得分大于阈值 或者英文clip得分大于阈值 都正确
                if (chinese_score >= chinese_threshold.get(chinese_prompt) or \
                        english_score >= english_threshold.get(english_prompt)) :
                  
                    # 先clip，再llava
                    llava_start = time.time()
                    flag = get_llava_result(image_url=url,
                                            prompt=chinese_prompt,
                                            llava_tokenizer=llava_tokenizer,
                                            llava_model=llava_model,
                                            llava_image_processor=llava_image_processor,
                                            context_len=context_len)
                    llava_end = time.time()
                    end_time = llava_end

                    llava_image_nums += 1
                    # 统计llava时间
                    llava_time += llava_end - llava_start

                    if flag:
                        pos_res.append(1)
                    else:
                        pos_res.append(0)
                else:
                    end_time = time.time()
                    pos_res.append(0)

                chinese_clip_time += end_chinese - start_chinese
                english_clip_time += end_english - start_english
                clip_time += end_english - start_chinese
                clip_llava_time += end_time - start_chinese

        else:
            image_list = os.listdir(os.path.join(image_folder, cate_name))
            for item in image_list:
                url = os.path.join(image_folder, cate_name, item)
                start_chinese = time.time()
                chinese_score = get_chinese_score(url, chinese_processor, chinese_model, text_encoder, text, device)
                end_chinese = time.time()

                start_english = time.time()
                english_score = get_english_score(url, english_model, english_preprocess, english_prompt, device)
                end_english = time.time()

                if (chinese_score >= chinese_threshold.get(chinese_prompt) or \
                        english_score >= english_threshold.get(english_prompt)) :
                    # 如果clip判断为正样本，再输入llava中
                    llava_start = time.time()
                    flag = get_llava_result(image_url=url,
                                            prompt=chinese_prompt,
                                            llava_tokenizer=llava_tokenizer,
                                            llava_model=llava_model,
                                            llava_image_processor=llava_image_processor,
                                            context_len=context_len)
                    llava_end = time.time()
                    # 这里统计一下llava推理的数量
                    llava_image_nums += 1

                    llava_time += llava_end - llava_start
                    end_time = llava_end

                    if flag:
                        neg_res.append(1)
                    else:
                        neg_res.append(0)
                else:
                    end_time = time.time()
                    neg_res.append(0)

                chinese_clip_time += end_chinese - start_chinese
                english_clip_time += end_english - start_english
                clip_time += end_english - start_chinese
                clip_llava_time += end_time - start_chinese


        print(f"英文clip推理所有样本花费的时间为：{english_clip_time:.4f}秒")
        print(f"中文推理所有样本花费的时间为：{chinese_clip_time:.4f}秒")
        print(f"英文+中文clip推理所有样本花费的时间为：{clip_time:.4f}秒")
        print(f"llava推理所有样本花费的时间为：{llava_time:.4f}秒")
        print(f"英文+中文clip+llava推理所有样本花费的时间为：{clip_llava_time:.4f}秒")
        print(f"llava推理的样本数量为：{llava_image_nums}张")
        print(f"{cate_name}类别文件夹处理完成")
        # print("Label probs:", logits_per_image.cpu().numpy())
    return pos_res, neg_res


def eval(res_pos, res_neg):
    res_pos = np.array(res_pos)
    res_neg = np.array(res_neg)
    # 计算tp,fp,fn 这里注意后面的分母是不是会出现nan或者0
    tp = sum(res_pos == 1)
    fp = sum(res_neg == 1)
    fn = sum(res_pos == 0)

    # precision = TP / TP + FP
    precision = tp / (tp + fp)

    # recall = TP / TP + FN
    recall = tp / (tp + fn)

    # f1-score = 2 * precision * recall / (precision + recall )
    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score, precision, recall


def main(image_folder, positive_cate,chinese_prompt,english_prompt):

    # 获取所有样本的clip 相似度分数
    pos_res, neg_res = union_clip_by_threshold(image_folder, positive_cate, chinese_prompt, english_prompt)
    print(len(pos_res))
    print(len(neg_res))

    cat_name = positive_cate

    f1_score, precision, recall = eval(pos_res, neg_res)

    print(f"{cat_name}_best_f1_score", f1_score)
    print(f"{cat_name}_best_precision", precision)
    print(f"{cat_name}_best_recall", recall)

    print('done')


if __name__ == '__main__':
    # 分别计算正样本和负样本的相似度
    image_folder = r"/root/autodl-tmp/Image-Downloader/images/"
    # 中文输入本次要算的类别,从下面中选，对应的是文件夹的名称
    # cat_tables = {"玫瑰": "rose",
    #               "狼": "wolf",
    #               "运动型多功能汽车": "suv", sport utility vehicle
    #               "中文书法": "chinese_calligraphy",
    #               "中国瓷器": "chinese_porcelain",
    #               }
    positive_cate = "guitar"

    # 中英文的prompt
    chinese_prompt = "吉他"
    english_prompt = "guitar"

    main(image_folder=image_folder, positive_cate=positive_cate,chinese_prompt=chinese_prompt,english_prompt=english_prompt)
