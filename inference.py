import numpy as np, torch, timm
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from matplotlib import colormaps

DEVICE = "cpu"
TEMPERATURE = 1.8
CLASSES = ["NORMAL", "PNEUMONIA"]

tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

yy, xx = np.mgrid[0:224, 0:224]
lung_mask = (((xx - 112) / 95) ** 2 + ((yy - 118) / 100) ** 2 <= 1).astype(float)
inner = (((xx - 112) / 85) ** 2 + ((yy - 118) / 90) ** 2 <= 1).astype(float)

def load_model(weights_path="vit_b16_best.pth"):
    import os
    if not os.path.exists(weights_path):
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(repo_id="ramisettynavya/pneumonia-vit-weights",
                                       filename="vit_b16_best.pth")
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    return model.to(DEVICE).eval()

def get_attention_map(model, x):
    store, hooks = [], []
    for blk in model.blocks:
        blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(
            lambda m, i, o: store.append(o.detach().cpu())))
    with torch.no_grad():
        out = model(x.to(DEVICE))
    for h in hooks:
        h.remove()
    result = torch.eye(197)
    for a in store:
        a = a[0].mean(0) + torch.eye(197)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    grid = result[0, 1:].reshape(1, 1, 14, 14)
    heat = F.interpolate(grid, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    probs = F.softmax(out / TEMPERATURE, dim=1)[0].cpu()
    return heat.numpy(), probs.argmax().item(), probs.max().item()

def clean_heat(heat):
    h = heat * lung_mask
    lo, hi = np.percentile(h[lung_mask > 0], [5, 99])
    return np.clip((h - lo) / (hi - lo + 1e-8), 0, 1) * lung_mask

def zone_shares(cleaned):
    strong = (cleaned > 0.5) * inner
    if strong.sum() == 0:
        strong = cleaned * inner
    names = {("left", 0): "upper right lung", ("left", 1): "middle right lung", ("left", 2): "lower right lung",
             ("right", 0): "upper left lung", ("right", 1): "middle left lung", ("right", 2): "lower left lung"}
    y0, y1 = 18, 218
    shares = {}
    for (side, band), name in names.items():
        xs = slice(0, 112) if side == "left" else slice(112, 224)
        ys = slice(int(y0 + band * (y1 - y0) / 3), int(y0 + (band + 1) * (y1 - y0) / 3))
        shares[name] = strong[ys, xs].sum()
    total = sum(shares.values()) + 1e-8
    return sorted(((n, v / total) for n, v in shares.items()), key=lambda t: -t[1])

def build_report(label, conf, ranked):
    if conf < 0.90:
        return ("**Result: uncertain.** The model is not confident enough about this image. "
                "Please have the scan reviewed by a doctor.")
    level = "high" if conf >= 0.97 else "moderate"

    if label == 0:
        return (
            f"**Finding:** No signs of pneumonia were detected (confidence {conf:.2%}, {level}).\n\n"
            "**Please note:** this does not rule out other conditions.\n\n"
            "**Suggestions:**\n"
            "- If you have cough, fever, chest pain or breathlessness, please see a doctor anyway.\n"
            "- Stay hydrated and rest if you are unwell.\n\n"
            "**Precautions:**\n"
            "- Avoid smoking and smoky or polluted air.\n"
            "- Wash hands regularly and keep vaccinations up to date.\n\n"
            "**Please consult a doctor** if symptoms continue. This is a decision-support tool, not a diagnosis."
        )

    right = sum(s for n, s in ranked if "right lung" in n)
    left = sum(s for n, s in ranked if "left lung" in n)
    if right >= 0.7:
        side = "mainly the right lung"
    elif left >= 0.7:
        side = "mainly the left lung"
    else:
        side = "both lungs"
    top_name, top_share = ranked[0]
    region = f", most strongly in the {top_name}" if top_share >= 0.30 else ""

    return (
        f"**Finding:** The model found patterns consistent with pneumonia (confidence {conf:.2%}, {level}).\n\n"
        f"**Where it looked:** its attention was on {side}{region}, as highlighted in the image.\n\n"
        "**What this tool cannot tell:** the type (bacterial or viral), the severity, or the cause.\n\n"
        "**Suggestions:**\n"
        "- Please see a doctor soon for confirmation. Do not wait for symptoms to worsen.\n"
        "- Take medicines only as prescribed by a doctor, and do not start antibiotics on your own.\n"
        "- Rest well and drink plenty of fluids.\n"
        "- Monitor your temperature and breathing.\n\n"
        "**Precautions:**\n"
        "- Avoid smoking and smoke exposure.\n"
        "- Cover your mouth when coughing, and wash hands often, to protect others.\n"
        "- Keep away from infants, elderly people and people with weak immunity until you are assessed.\n\n"
        "**Seek urgent care immediately for:** difficulty breathing, chest pain, bluish lips, confusion, "
        "or a persistent high fever.\n\n"
        "**Please consult a doctor.** This is a decision-support tool, not a diagnosis."
    )

def make_overlay(img, cleaned):
    base = np.array(img.convert("RGB").resize((224, 224))).astype(float)
    colors = colormaps["jet"](cleaned)[..., :3] * 255
    alpha = (0.5 * (lung_mask > 0))[..., None]
    out = base * (1 - alpha) + colors * alpha
    return Image.fromarray(out.astype("uint8")).resize((448, 448))

def analyze(model, pil_img):
    x = tf(pil_img.convert("RGB")).unsqueeze(0)
    heat, pred, conf = get_attention_map(model, x)
    cleaned = clean_heat(heat)
    ranked = zone_shares(cleaned)
    return {"label": CLASSES[pred], "confidence": conf,
            "overlay": make_overlay(pil_img, cleaned),
            "report": build_report(pred, conf, ranked)}
