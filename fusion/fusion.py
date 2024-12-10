import torch

def unified_prediction(text_pred, text_conf, audio_pred, audio_conf, video_pred, video_conf):
    unified_labels = ['Anger', 'Disgust','Fear','Happy', 'Sad', 'Surprise', 'Neutral']

    # Map predictions to indices in the unified label space
    text_idx = unified_labels.index(text_pred) if text_pred in unified_labels else None
    audio_idx = unified_labels.index(audio_pred) if audio_pred in unified_labels else None
    video_idx = unified_labels.index(video_pred) if video_pred in unified_labels else None

    # Initialize logits for the unified label space
    num_unified_classes = len(unified_labels)
    logits = torch.zeros(num_unified_classes)

    # Weighted contribution of each model's confidence to logits
    weights = torch.tensor([0.4, 0.3, 0.3])  # Adjust these weights as needed
    if text_idx is not None:
        logits[text_idx] += weights[0] * text_conf
    if audio_idx is not None:
        logits[audio_idx] += weights[1] * audio_conf
    if video_idx is not None:
        logits[video_idx] += weights[2] * video_conf

    # Determine final prediction and its confidence
    final_idx = torch.argmax(logits).item()
    final_prediction = unified_labels[final_idx]
    final_confidence = logits[final_idx].item()

    return final_prediction, final_confidence
