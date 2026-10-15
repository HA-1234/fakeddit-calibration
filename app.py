"""
Multimodal Misinformation Detection — Gradio Demo
Loads the trained fusion model (RoBERTa + CLIP -> MLP) and applies temperature-scaled
calibration to the output confidence, so the displayed confidence is the corrected one,
not the raw (overconfident) one.
"""

import json
import numpy as np
import torch
import joblib
import gradio as gr
from transformers import RobertaTokenizer, RobertaModel, CLIPProcessor, CLIPModel

LABELS = {0: "Real", 1: "Satire", 2: "Misleading"}

print("Loading models — this takes a moment on first launch...")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta = RobertaModel.from_pretrained("roberta-base").eval()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

fusion_model = joblib.load("models/fusion_shared_split.joblib")

with open("models/temperature_scaling_summary.json") as f:
    TEMPERATURE = json.load(f)["temperature"]

print(f"Models loaded. Using calibration temperature T={TEMPERATURE:.4f}")


@torch.no_grad()
def get_text_embedding(text: str) -> np.ndarray:
    enc = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
    out = roberta(**enc)
    return out.last_hidden_state[:, 0, :].numpy()


@torch.no_grad()
def get_image_embedding(image) -> np.ndarray:
    inputs = clip_processor(images=image, return_tensors="pt")
    output = clip_model.get_image_features(**inputs)
    # Output type varies across transformers versions — handle all cases
    if hasattr(output, "image_embeds"):
        feats = output.image_embeds
    elif hasattr(output, "pooler_output"):
        feats = output.pooler_output
    else:
        feats = output
    return feats.numpy()


def softmax_with_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    """Same temperature-scaling formula used in Stage 5: log(proba) as pseudo-logits,
    mathematically equivalent to true-logit scaling for this purpose."""
    logits = np.log(np.clip(proba, 1e-12, 1.0))
    scaled = logits / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def predict(caption: str, image):
    if not caption or caption.strip() == "":
        return "Please enter a caption.", None
    if image is None:
        return "Please upload an image.", None

    text_emb = get_text_embedding(caption)
    image_emb = get_image_embedding(image)
    fusion_emb = np.concatenate([text_emb, image_emb], axis=1)

    raw_proba = fusion_model.predict_proba(fusion_emb)
    calibrated_proba = softmax_with_temperature(raw_proba, TEMPERATURE)[0]

    pred_idx = int(np.argmax(calibrated_proba))
    pred_label = LABELS[pred_idx]
    pred_confidence = float(calibrated_proba[pred_idx])

    summary = (
        f"### Prediction: **{pred_label}**\n"
        f"Calibrated confidence: **{pred_confidence:.1%}**\n\n"
        f"_(Confidence shown is temperature-scaled, T={TEMPERATURE:.3f} — "
        f"see the README for why this matters and how much it improved calibration.)_"
    )
    class_scores = {LABELS[i]: float(calibrated_proba[i]) for i in range(3)}
    return summary, class_scores


with gr.Blocks(title="Multimodal Misinformation Detector") as demo:
    gr.Markdown(
        """
        # Multimodal Misinformation Detection — with Calibrated Confidence

        Classifies a social-media-style post (caption + image) as **Real**, **Satire**, or **Misleading**.

        Unlike most demo classifiers, the confidence score shown here isn't the model's raw output —
        it's been corrected using **temperature scaling**, a post-hoc calibration technique, because the
        raw fusion model was found to be significantly overconfident (see the
        [project write-up](https://github.com/YOUR_USERNAME/fakeddit-calibration) for the full analysis).
        """
    )
    with gr.Row():
        with gr.Column():
            caption_input = gr.Textbox(label="Post caption", placeholder="e.g. this eye watches me when i visit my moms")
            image_input = gr.Image(type="pil", label="Post image")
            submit_btn = gr.Button("Classify", variant="primary")
        with gr.Column():
            result_markdown = gr.Markdown(label="Prediction")
            result_label = gr.Label(label="Calibrated class probabilities", num_top_classes=3)

    submit_btn.click(fn=predict, inputs=[caption_input, image_input], outputs=[result_markdown, result_label])
    caption_input.submit(fn=predict, inputs=[caption_input, image_input], outputs=[result_markdown, result_label])

if __name__ == "__main__":
    demo.launch()
