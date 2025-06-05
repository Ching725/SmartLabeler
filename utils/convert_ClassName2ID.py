from pathlib import Path

label_dir = Path("/Users/ching/Documents/SideProject/data/DAGM_2007_Dataset/Class1/Train/Label/yolo")
for txt in label_dir.glob("*.txt"):
    lines = []
    with open(txt, 'r') as f:
        for line in f:
            if line.startswith("class1"):
                line = line.replace("class1", "1", 1)
            lines.append(line)
    with open(txt, 'w') as f:
        f.writelines(lines)

print("✅ 所有標註已轉為 class ID = 1")