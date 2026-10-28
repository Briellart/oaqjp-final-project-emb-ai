import os

import requests

EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness")
DEFAULT_URL = (
    "https://sn-watson-emotion.labs.skills.network/"
    "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {
    "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
}


def empty_result():
    """Return the required result for invalid input or HTTP 400."""
    return dict.fromkeys((*EMOTIONS, "dominant_emotion"))


def emotion_detector(text_to_analyze):
    """Return five emotion scores and the highest-scoring emotion.

    Empty input and HTTP 400 return None for every result field.
    Other service/network errors propagate so they are not confused with
    invalid input. WATSON_EMOTION_URL can select another authorized runtime.
    """
    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return empty_result()

    response = requests.post(
        os.environ.get("WATSON_EMOTION_URL", DEFAULT_URL),
        json={"raw_document": {"text": text_to_analyze}},
        headers=HEADERS,
        timeout=20,
    )
    if response.status_code == 400:
        return empty_result()
    response.raise_for_status()
    predictions = response.json()["emotionPredictions"]
    scores = predictions[0]["emotion"]
    result = {emotion: scores[emotion] for emotion in EMOTIONS}
    result["dominant_emotion"] = max(result, key=result.get)
    return result
