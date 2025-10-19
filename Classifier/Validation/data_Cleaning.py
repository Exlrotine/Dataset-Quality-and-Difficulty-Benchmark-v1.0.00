# dup_scan.py
import os, json, hashlib, argparse
from pathlib import Path
from PIL import Image
import imagehash


def iter_images(root, exts={'.png'}):
    for p in Path(root).rglob('*'):
        if p.suffix.lower() in exts:
            yield p


def perceptual_hash(img_path, mode='phash'):
    img = Image.open(img_path).convert('RGB')
    if mode == 'phash':  return imagehash.phash(img)       # 64-bit
    if mode == 'dhash':  return imagehash.dhash(img)
    raise ValueError


def main(root, mode='phash'):
    hash2paths = {}                   # 存储图片哈希值
    for p in iter_images(root):
        h = str(perceptual_hash(p, mode))   # or sha1_hash(p)
        hash2paths.setdefault(h, []).append(str(p))

    dup = {h: ps for h, ps in hash2paths.items() if len(ps) > 1}
    print(f"Scanned {sum(len(v) for v in hash2paths.values())} images,"
          f" duplicates groups = {len(dup)}")

    out = Path(root)/'duplicate_report.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(dup, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', help='dataset root folder')
    parser.add_argument('--mode', default='phash', choices=['phash', 'dhash', 'sha1'])
    main(**vars(parser.parse_args()))
