import os
import json

classes = sorted(
    os.listdir(
        r"furits\fruits-360_100x100\fruits-360\Training"
    )
)

with open("classes.json", "w") as f:
    json.dump(classes, f)

print("✅ classes.json created")