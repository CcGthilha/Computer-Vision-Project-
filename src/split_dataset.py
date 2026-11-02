# แบ่ง dataset เป็น train/val/test (70/15/15) แยก fresh/rotten
# วิธีใช้: แตก zip ไว้ข้างไฟล์นี้ แก้ SRC ให้ตรงชื่อโฟลเดอร์ แล้วรัน python split_dataset.py
import random, shutil
from pathlib import Path

SRC, DST = Path('Fruits Dataset for Classification'), Path('dataset')
random.seed(42)  # seed คงที่ ทุกคนรันแล้วได้ผลเหมือนกัน
groups, count = {}, {}
for p in SRC.rglob('*'):
    if p.suffix.lower() not in ('.jpg', '.jpeg', '.png'): continue
    r = str(p.parent.relative_to(SRC)).lower()
    c = 'rotten' if 'rotten' in r or 'bad' in r else 'fresh' if 'fresh' in r or 'good' in r else None
    if c: groups.setdefault((p.parent, c), []).append(p)
for (folder, c), ps in groups.items():  # แบ่งทีละโฟลเดอร์ผลไม้ ให้ทุกชนิดกระจายเท่ากัน
    random.shuffle(ps); a, b = int(len(ps) * .7), int(len(ps) * .85)
    for split, part in (('train', ps[:a]), ('val', ps[a:b]), ('test', ps[b:])):
        d = DST / split / c; d.mkdir(parents=True, exist_ok=True)
        for p in part: shutil.copy(p, d / f'{folder.name}_{p.name}'.replace(' ', '_'))
        count[split, c] = count.get((split, c), 0) + len(part)
print(f"{'split':<6}{'fresh':>7}{'rotten':>8}")
for s in ('train', 'val', 'test'): print(f"{s:<6}{count.get((s,'fresh'),0):>7}{count.get((s,'rotten'),0):>8}")
print('รวม', sum(count.values()), 'ภาพ')
