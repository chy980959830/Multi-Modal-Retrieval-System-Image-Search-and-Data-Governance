from PIL import Image
import requests
import clip
import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from torchvision.datasets import ImageFolder
from typing import List, Tuple, Dict, Union, Optional, Callable, Any, cast
from pathlib import Path
import os
from tqdm import tqdm
import shutil
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.model.builder import load_pretrained_model

ORIGINAL_IMAGE_DIR = "/root/autodl-tmp/Image-Downloader/images/"
NEW_IMAGE_DIR = "/root/autodl-tmp/images/"

# 更改huggingface的缓存默认目录
import os
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['TRANSFORMERS_OFFLINE'] = '0' 

"""参考DatasetFolder类源码：
https://pytorch.org/vision/stable/generated/torchvision.datasets.DatasetFolder.html#torchvision.datasets.DatasetFolder"""
from torchvision.datasets import VisionDataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))

class CustomDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        loader: Callable[[str], Any] = default_loader,
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        # 目标类别
        target_classes: List[str] = None,
        **kwargs
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        original_classes, original_class_to_idx = self.find_classes(self.root)
        target_class_to_idx = {cls_name: i for i, cls_name in enumerate(target_classes)}
        map_class = {}
        # 5个正类加上其他
        if len(target_classes) == 6:
            for target_class in target_classes:
                if target_class == "其他" or target_class == "others":
                    map_class[target_class] = ["tennis-racket", "cherry", "mantou", "dress-shirt", "violin"]
                elif target_class in ["badminton-racket", "羽毛球拍"]:
                    map_class[target_class] = ["badminton-racket"]
                elif target_class in ["lychee", "荔枝"]:
                    map_class[target_class] = ["lychee"]
                elif target_class in ["baozi", "包子"]:
                    map_class[target_class] = ["baozi"]
                elif target_class in ["T-shirt", "T恤"]:
                    map_class[target_class] = ["T-shirt"]
                elif target_class in ["guitar", "吉他"]:
                    map_class[target_class] = ["guitar"]
                else:
                    raise ValueError(f"Found no valid class {target_class}.")
        # 1个正类加上其他
        elif len(target_classes) == 2:
            for target_class in target_classes:
                if target_class in ["badminton-racket", "羽毛球拍"]:
                    map_class[target_class] = ["badminton-racket"]
                    if target_class == "badminton-racket":
                        map_class["not badminton-racket"] = [cls_name for idx, cls_name in enumerate(original_classes) if idx != original_class_to_idx["badminton-racket"]]
                    else:
                        map_class["不是羽毛球拍"] = [cls_name for idx, cls_name in enumerate(original_classes) if idx != original_class_to_idx["badminton-racket"]]
                elif target_class in ["lychee", "荔枝"]:
                    map_class[target_class] = ["lychee"]
                    if target_class == "lychee":
                        map_class["not lychee"] = [cls_name for idx, cls_name in enumerate(original_classes) if idx != original_class_to_idx["lychee"]]
                    else:
                        map_class["不是荔枝"] = [cls_name for idx, cls_name in enumerate(original_classes) if idx != original_class_to_idx["lychee"]]
                elif target_class in ["baozi", "包子"]:
                    map_class[target_class] = ["baozi"]
                    if target_class == "baozi":
                        map_class["not baozi"] = [cls_name for idx, cls_name in enumerate(original_classes) if idx != original_class_to_idx["baozi"]]
                    else:
                        map_class["不是包子"] = [cls_name for idx, cls_name in enumerate(original_classes) if idx != original_class_to_idx["baozi"]]
                elif target_class in ["T-shirt", "T恤"]:
                    map_class[target_class] = ["T-shirt"]
                    if target_class == "T-shirt":
                        map_class["not T-shirt"] = [cls_name for idx, cls_name in enumerate(original_classes) if idx != original_class_to_idx["T-shirt"]]
                    else:
                        map_class["不是T恤"] = [cls_name for idx, cls_name in enumerate(original_classes) if idx != original_class_to_idx["T-shirt"]]
                elif target_class in ["guitar", "吉他"]:
                    map_class[target_class] = ["guitar"]
                    if target_class == "guitar":
                        map_class["not guitar"] = [cls_name for idx, cls_name in enumerate(original_classes) if idx != original_class_to_idx["guitar"]]
                    else:
                        map_class["不是吉他"] = [cls_name for idx, cls_name in enumerate(original_classes) if idx != original_class_to_idx["guitar"]]
        else:
            raise ValueError("Not match!")
            
        samples = self.make_dataset(
            self.root,
            target_class_to_idx,
            map_class,
            extensions=IMG_EXTENSIONS,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
        )
        
        self.loader = loader
        self.extensions = extensions

        self.classes = target_classes
        self.class_to_idx = target_class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        
    @staticmethod    
    def make_dataset(
        directory: Union[str, Path],
        class_to_idx: Dict[str, int],
        map_class: Optional[Dict[str, str]] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
    ) -> List[Tuple[str, int]]:
        """
        重写 make_dataset，处理合并后的标签逻辑
        """
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in class_to_idx.keys():
            class_index = class_to_idx[target_class]
            for path_name in map_class[target_class]:
                target_dir = os.path.join(directory, path_name)
                if not os.path.isdir(target_dir):
                    continue
                for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        if is_valid_file(path):
                            item = path, class_index
                            instances.append(item)

                            if target_class not in available_classes:
                                available_classes.add(target_class)
        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes and not allow_empty:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances
        
    def find_classes(self, directory: Union[str, Path]) -> List[str]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
            
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

def clip_en_predict(model, test_loader, text_inputs, threshold):
    model.eval()
    # 记录正类和负类的阈值
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            # 计算图像特征
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # 计算文本特征
            text_features = model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logit_scale = model.logit_scale.exp()
        
            # 计算相似度（例如：进行零样本分类）
            similarity = logit_scale * torch.nn.functional.cosine_similarity(image_features, text_features)
            predicted_indices = (similarity < threshold).int()
        
            # 可将predicted_indices与labels对比，计算准确率等
            all_preds.extend(predicted_indices.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_preds, all_labels

def clip_cn_predict(clip_model, text_encoder, test_loader, text_inputs, threshold):

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images['pixel_values'][0].to(device)

            # 计算图像特征
            image_features = clip_model.get_image_features(pixel_values=images)
            # 计算文本特征
            text_features = text_encoder(text_inputs).logits
            # 归一化
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            # 计算余弦相似度 logit_scale是尺度系数
            logit_scale = clip_model.logit_scale.exp()
            similarity = logit_scale * torch.nn.functional.cosine_similarity(image_features, text_features)
        
            # 通过阈值判断
            preds = (similarity < threshold).int()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return all_preds, all_labels

def eval(preds, labels, target_classes):
    assert len(preds) == len(labels)
    preds = np.array(preds)
    labels = np.array(labels)
    print(f"label_idx\tlabel\tprecision\trecall\tF1-score\n")
    for label_idx, target_class in enumerate(target_classes):
         # 计算tp,fp,fn 这里注意后面的分母是不是会出现nan或者0
         tp = sum((preds == label_idx) & (labels == label_idx))
         fp = sum((preds == label_idx) & (labels != label_idx))
         fn = sum((preds != label_idx) & (labels == label_idx))
         # precision = TP / TP + FP
         precision = tp / (tp + fp)
         # recall = TP / TP + FN
         recall = tp / (tp + fn)
         
         # f1-score = 2 * precision * recall / (precision + recall )
         f1_score = 2 * precision * recall / (precision + recall)
         print(f"{label_idx}\t{target_class}\t{precision:.4f}\t{recall:.4f}\t{f1_score:.4f}\n")


def filter_preds(preds, dataset, target_class, model_path):
    # 获取所有正样本的路径
    positive_samples = [
        (idx, sample[0]) 
        for idx, (sample, pred) in enumerate(zip(dataset.samples, preds))
        if pred == 0  # 假设0表示T恤类别
    ]
    filter_nums = 0
    prompt = f"Is this picture includes {target_class}? If yes, answer yes. If no, answer no"
    for idx, src_path in tqdm(positive_samples, desc="llava筛选进度"):
        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(model_path),
            "query": prompt,
            "conv_mode": None,
            "image_file": src_path,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512})()
        # 保留yes的样本，将no的样本筛选出去
        output = eval_model(args)
        assert preds[idx] == 0
        print(output)
        if "yes" in output.lower():
            preds[idx] = 0
        else:
            filter_nums += 1
            preds[idx] = 1
    print("filter:", filter_nums)
    return preds

"""
# 以cn_dataset的labels为准
def union_preds(preds_cn, preds_en, cn_dataset, en_dataset, all_labels):
    positive_samples_cn = [
        (idx, sample[0]) 
        for idx, (sample, pred) in enumerate(zip(cn_dataset.samples, preds_cn))
        if pred == 0  # 假设0表示T恤类别
    ]
    positive_samples_en = [
        (idx, sample[0]) 
        for idx, (sample, pred) in enumerate(zip(en_dataset.samples, preds_en))
        if pred == 0  # 假设0表示T恤类别
    ]
    for idx, src_path in tqdm(positive_samples_en, desc="合并进度"):
"""

if __name__ == "__main__":
     # 加载LLaVA
    model_path = "liuhaotian/llava-v1.5-7b"
    """
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path),
        load_8bit=True
    )   
    """
    target_classes_en = ["lychee"]
    en_thresholds = [27.63269]
    target_classes_cn = ["荔枝"]
    cn_thresholds = [10.37709]

    # 加载英文CLIP模型和对应的预处理方法
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_en_model, clip_en_preprocess = clip.load("ViT-B/32", device=device)
    # 加载中文CLIP模型和对应的预处理方法
    text_tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese")
    text_encoder = BertForSequenceClassification.from_pretrained("IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese").eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14") 
    clip_model = clip_model.to(device)
    text_encoder = text_encoder.to(device)
    preds_en_classes = []
    preds_cn_classes = []
    target_class = target_classes_en[0]
    en_binary_classes = [target_class, "not "+target_class]
    cn_binary_classes = [target_classes_cn[0], "不是"+target_classes_cn[0]]

    en_dataset = CustomDataset(
        root=ORIGINAL_IMAGE_DIR,
        transform=clip_en_preprocess,
        target_classes=en_binary_classes)
    cn_dataset = CustomDataset(
        root=ORIGINAL_IMAGE_DIR,
        transform=processor.image_processor,
        target_classes=cn_binary_classes)
        
    print(f"Load dataset with classes:{en_binary_classes[0]}, {en_binary_classes[1]}")
    en_test_loader = DataLoader(en_dataset, batch_size=32, shuffle=False, num_workers=4)
    cn_test_loader = DataLoader(cn_dataset, batch_size=32, shuffle=False, num_workers=4)

    en_text_inputs = torch.tensor(clip.tokenize(target_class)).to(device)
    cn_text_inputs = text_tokenizer([target_classes_cn[0]], return_tensors='pt', padding=True)['input_ids'].to("cuda")
    clip_en_model.to('cuda')
    preds_en, all_labels_en = clip_en_predict(clip_en_model, en_test_loader, en_text_inputs, en_thresholds[0])
    print(eval(preds_en, all_labels_en,en_binary_classes))
    clip_en_model.to('cpu')
    clip_model.to('cuda')
    preds_cn, all_labels_cn = clip_cn_predict(clip_model, text_encoder, cn_test_loader, cn_text_inputs, cn_thresholds[0])
    assert all_labels_cn == all_labels_en
    clip_model.to('cpu')
    print(eval(preds_cn, all_labels_cn,en_binary_classes))
    #preds = union_preds(preds_cn, preds_en, cn_dataset, en_dataset, all_labels)
    preds = [0 if (cn ==0 or en ==0) else 1 for cn, en in zip(preds_cn, preds_en)]
    print(eval(preds, all_labels_en,en_binary_classes))
    new_preds = filter_preds(preds, cn_dataset, target_class, model_path)
    print(eval(new_preds, all_labels_cn,en_binary_classes))
        
    