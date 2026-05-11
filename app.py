import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from einops import rearrange
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAKE Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }

.stApp { background-color: #0a0a0a; color: #e8e4dc; }

#MainMenu, footer, header { visibility: hidden; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: #e8e4dc;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 0.78rem;
    color: #555;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

div[data-testid="stSelectbox"] label {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #555 !important;
}

.result-card {
    border: 1px solid #1e1e1e;
    border-radius: 2px;
    padding: 2rem;
    margin-top: 1rem;
    background: #0f0f0f;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.result-card.real::before { background: #4ade80; }
.result-card.fake::before { background: #f87171; }

.verdict {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}
.verdict.real { color: #4ade80; }
.verdict.fake { color: #f87171; }

.confidence-label {
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #444;
    margin-top: 1.4rem;
    margin-bottom: 0.5rem;
}

.bar-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.5rem; }
.bar-name { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #888; width: 3.2rem; flex-shrink: 0; }
.bar-bg { flex: 1; height: 4px; background: #1a1a1a; border-radius: 2px; overflow: hidden; }
.bar-fill-real { height: 100%; background: #4ade80; border-radius: 2px; }
.bar-fill-fake { height: 100%; background: #f87171; border-radius: 2px; }
.bar-val { font-size: 0.72rem; color: #666; width: 3.5rem; text-align: right; flex-shrink: 0; }

.model-tag {
    display: inline-block;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #333;
    border: 1px solid #222;
    padding: 0.2rem 0.5rem;
    border-radius: 1px;
    margin-top: 1.2rem;
}

.divider { border: none; border-top: 1px solid #1a1a1a; margin: 2rem 0; }

.info-box {
    background: #0f0f0f;
    border: 1px solid #1a1a1a;
    border-left: 2px solid #333;
    padding: 0.9rem 1rem;
    font-size: 0.75rem;
    color: #555;
    line-height: 1.7;
    margin-top: 1rem;
}

[data-testid="stFileUploaderDropzone"] {
    background: #0f0f0f !important;
    border: 1px dashed #222 !important;
    border-radius: 2px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #444 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model definitions (must match training code exactly) ──────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)
        ) if (in_c != out_c or stride != 1) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class FakeDetectorCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = ResBlock(64,  64)
        self.layer2 = ResBlock(64,  128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 512, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.head(self.pool(self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden     = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        n = self.norm1(x)
        x = x + self.attn(n, n, n)[0]
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=192):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return rearrange(self.proj(x), 'b e h w -> b (h w) e')


class FakeDetectorViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=192, depth=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)
        self.blocks    = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)])
        self.norm      = nn.LayerNorm(embed_dim)
        self.head      = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.patch_embed(x)], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return self.head(self.norm(self.blocks(x)[:, 0]))


class CNNTokenizer(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = ResBlock(64,  64)
        self.layer2 = ResBlock(64,  128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.proj   = nn.Sequential(nn.Conv2d(256, embed_dim, kernel_size=1, bias=False), nn.BatchNorm2d(embed_dim))
        self.n_tokens = 64

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(self.stem(x))))
        return rearrange(self.proj(x), 'b e h w -> b (h w) e')


class HybridFakeDetector(nn.Module):
    def __init__(self, embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.tokenizer = CNNTokenizer(embed_dim)
        n = self.tokenizer.n_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)
        self.blocks    = nn.Sequential(*[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm      = nn.LayerNorm(embed_dim)
        self.head      = nn.Sequential(nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 2))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = torch.cat([self.cls_token.expand(B, -1, -1), self.tokenizer(x)], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        return self.head(self.norm(self.blocks(x)[:, 0]))


# ── Constants ──────────────────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CONFIGS = {
    "Hybrid (CNN + ViT)": ("Hybrid_best.pt", HybridFakeDetector),
    "CNN (ResNet-style)": ("CNN_best.pt",     FakeDetectorCNN),
    "ViT (from scratch)": ("ViT_best.pt",     FakeDetectorViT),
}

infer_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


# ── Model loader ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    fname, cls = MODEL_CONFIGS[model_name]
    model = cls().to(DEVICE)
    try:
        model.load_state_dict(torch.load(fname, map_location=DEVICE))
        model.eval()
        return model, None
    except FileNotFoundError:
        return None, fname


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, img: Image.Image):
    x = infer_transform(img).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return float(probs[1]), float(probs[0])   # real_prob, fake_prob


# ── UI ─────────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">CIFAKE<br>Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Real vs AI-Generated Image Classification</div>', unsafe_allow_html=True)

model_choice = st.selectbox(
    "Model",
    list(MODEL_CONFIGS.keys()),
    index=0,
)

model, missing_file = load_model(model_choice)

if missing_file:
    st.markdown(f"""
    <div class="info-box">
    ⚠️ &nbsp; Weight file <code>{missing_file}</code> not found in the current directory.<br>
    Place your <code>.pt</code> files alongside <code>app.py</code> and restart the app.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

uploaded = st.file_uploader(
    label="Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded and model:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns([1, 1.6], gap="large")

    with col1:
        st.image(img, use_container_width=True)

    with col2:
        with st.spinner("Analysing..."):
            time.sleep(0.2)
            real_p, fake_p = predict(model, img)

        verdict    = "REAL" if real_p >= 0.5 else "FAKE"
        card_class = verdict.lower()
        confidence = real_p if verdict == "REAL" else fake_p

        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="verdict {card_class}">{verdict}</div>
            <div style="font-size:0.75rem; color:#444; margin-top:0.2rem;">
                {confidence*100:.1f}% confidence
            </div>
            <div class="confidence-label">Probability breakdown</div>
            <div class="bar-row">
                <div class="bar-name">Real</div>
                <div class="bar-bg">
                    <div class="bar-fill-real" style="width:{real_p*100:.1f}%"></div>
                </div>
                <div class="bar-val">{real_p*100:.1f}%</div>
            </div>
            <div class="bar-row">
                <div class="bar-name">Fake</div>
                <div class="bar-bg">
                    <div class="bar-fill-fake" style="width:{fake_p*100:.1f}%"></div>
                </div>
                <div class="bar-val">{fake_p*100:.1f}%</div>
            </div>
            <div class="model-tag">{model_choice}</div>
        </div>
        """, unsafe_allow_html=True)

elif uploaded and not model:
    st.error("Model weights not loaded. See the warning above.")

else:
    st.markdown("""
    <div class="info-box">
    Upload any image — photograph, screenshot, or AI-generated — and the model will classify it
    as <strong style="color:#4ade80">REAL</strong> or <strong style="color:#f87171">FAKE</strong>.<br><br>
    The <strong>Hybrid</strong> model (CNN + ViT) generally performs best.
    Switch models using the selector above to compare predictions.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="font-size:0.65rem; color:#2a2a2a; letter-spacing:0.08em; text-transform:uppercase;">
    Trained on CIFAKE · 60k real (CIFAR-10) + 60k Stable Diffusion images
</div>
""", unsafe_allow_html=True)
