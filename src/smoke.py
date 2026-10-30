import torch, sys
from PIL import Image, ImageDraw
from transformers import AutoProcessor, GroundingDinoForObjectDetection

dev = "mps" if torch.backends.mps.is_available() else "cpu"
name = "IDEA-Research/grounding-dino-tiny"
proc = AutoProcessor.from_pretrained(name)
model = GroundingDinoForObjectDetection.from_pretrained(name).to(dev).eval()

img = Image.open(sys.argv[1]).convert("RGB")
inputs = proc(images=img, text="person.", return_tensors="pt").to(dev)
with torch.no_grad():
    out = model(**inputs)
res = proc.post_process_grounded_object_detection(
    out, inputs.input_ids, threshold=0.3, text_threshold=0.25,
    target_sizes=[img.size[::-1]])[0]

d = ImageDraw.Draw(img)
for b, s in zip(res["boxes"], res["scores"]):
    d.rectangle(b.tolist(), outline="red", width=3)
img.save("smoke_out.jpg"); print(len(res["boxes"]), "boxes on", dev)
