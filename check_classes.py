import os

path = r"furits\fruits-360_100x100\fruits-360\Training"

classes = os.listdir(path)

print("Total Classes:", len(classes))
print("\nFirst 20 Classes:")

for c in classes[:20]:
    print(c)