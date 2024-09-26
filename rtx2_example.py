import torch
from rtx import RTX2
from PIL import Image
from torchvision import transforms
import torch
import einops
import numpy as np
import requests

def tokenize(text_input):
    # スペースで区切られた単語をトークン化（簡易版）
    words = text_input.split()  # テキストをスペースで分割
    # トークン化: 各単語をユニークな整数にマッピング（例: 辞書を使用）
    token_dict = {word: idx for idx, word in enumerate(set(words))}
    tokens = [token_dict[word] for word in words]  # 各単語を整数に変換
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # (1, seq_length)


def run():
    # usage
    #img = torch.randn(1, 3, 256, 256)
    img_path = '/home/koki/Downloads/Screenshot from 2024-09-26 17-07-24.png'
    #img = Image.open(img_path)
    #text = torch.randint(0, 20000, (1, 1024))
    text = tokenize("Pick up the yorgurt.")
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # (C, H, W)に合うようにリサイズ
        transforms.ToTensor()         # Tensorに変換 (C, H, W)
    ])

    img = transform(img).unsqueeze(0)
    try:
        img= einops.rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=32, p2=32)
        print("Rearranged tensor shape:", img.shape)
    except einops.EinopsError as e:
        print("EinopsError:", e)
    """
    img = np.array(Image.open(img_path).resize((32, 32)))
    if img.shape[2] == 4:
        img = img[:, :, :3]  # RGBに変換

    # numpy配列をtorchテンソルに変換し、バッチ次元を追加
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    model = RTX2()
    output = model(img, text)
    print(output)
    


if __name__ == "__main__":
    run()
