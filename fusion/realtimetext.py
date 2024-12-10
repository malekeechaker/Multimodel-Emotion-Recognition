import speech_recognition as sr
import tensorflow as tf
from nltk.corpus import stopwords
import re
import nltk
import os
from transformers import BertTokenizer, TFBertModel
import threading

nltk.download('stopwords')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.compat.v1.disable_eager_execution()

# Charger le modèle d'émotion
loaded_model = tf.keras.models.load_model(
    "C:\\Users\\Home\\Desktop\\ids5\\YOLO\\final_results\\Text\\emotion_detector-v2.h5", 
    custom_objects={"TFBertModel":TFBertModel}
)

input_ids = loaded_model.input
input_tensor = tf.convert_to_tensor(input_ids.numpy(), dtype=tf.int32)

recognizer = sr.Recognizer()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    

# Mapping des labels
map_labels = {
    'Anger': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Sad': 4,
    'Surprise': 5,
    'Neutral': 6
}

# Fonction pour convertir le discours en texte
def speech_to_text(recognizer):
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=1, phrase_time_limit=1)
            text = recognizer.recognize_google(audio, language="en-EN")  # Changez la langue si nécessaire
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Erreur avec le service de reconnaissance vocale : {e}")
            return None
        except Exception as e:
            print(f"Une erreur est survenue : {e}")
            return None

# Prétraitement du texte
def remove_digits(text):
    pattern = r'[^a-zA-Z.,!?/:;"\'\s]'
    return re.sub(pattern, '', text)

def remove_special_characters(text):
    pat = r'[^a-zA-Z0-9.,!?/:;"\'\s]'
    return re.sub(pat, '', text)

def preprocess(text):
    cachedStopWords = set(stopwords.words("english"))
    cachedStopWords.update(('and', 'I', 'A', 'http', 'And', 'So', 'arnt', 'This', 'When', 'It', 'many', 'Many', 'so', 'cant', 'Yes', 'yes', 'No', 'no', 'These', 'these', 'mailto', 'regards', 'ayanna', 'like', 'email'))
    new_str = ' '.join([word for word in text.split() if word not in cachedStopWords])
    new_str = re.sub(r'[\w.-]+@[\w.-]+', '', new_str)
    new_str = re.sub(r'http\S+', '', new_str)
    new_str = re.sub(r'<.*?>', '', new_str)
    new_str = remove_digits(new_str)
    new_str = remove_special_characters(new_str)
    new_str = new_str.lower()
    return new_str.strip()

# Fonction pour prédire l'émotion à partir du texte
def predict(text, model = loaded_model, tokenizer = tokenizer,  max_len=60):
    """
    Prédit l'émotion à partir d'un texte et retourne le label prédit ainsi que la confiance.

    Args:
        model: Le modèle entraîné pour la prédiction.
        tokenizer: Le tokenizer associé au modèle.
        text (str): Le texte à analyser.
        max_len (int): La longueur maximale de séquence pour le padding et la troncature.

    Returns:
        tuple: (émotion prédite, score de confiance)
    """
    if not text:
        return None, None
    
    # Prétraitement du texte
    text = preprocess(text)
    
    # Tokenisation
    inputs = tokenizer(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf',
        return_token_type_ids=False,
        return_attention_mask=True
    )
    
    inputs["input_ids"] = tf.convert_to_tensor(inputs["input_ids"], dtype=tf.int32)
    inputs["attention_mask"] = tf.convert_to_tensor(inputs["attention_mask"], dtype=tf.int32)

    # Prédictions du modèle
    preds = model.predict([inputs["input_ids"], inputs["attention_mask"]])
    
    # Obtenir la classe prédite et la confiance
    predicted_classes = tf.argmax(preds, axis=1).numpy()[0]
    confidence_score = tf.reduce_max(tf.nn.softmax(preds, axis=1)).numpy()
    
    # Mapper la classe prédite au label correspondant
    emotion = map_labels.get(predicted_classes, "Unknown")
    
    return emotion, confidence_score


# Boucle principale pour l'écoute continue
def main():

    print("Appuyez sur 'q' pour quitter.")

    def listen():
        while True:
            try:
                text = speech_to_text(recognizer)
                if text:
                    print(f"Texte reconnu : {text}")
                    predict(loaded_model, tokenizer, text)
            except KeyboardInterrupt:
                break

    listener_thread = threading.Thread(target=listen)
    listener_thread.start()

    while True:
        cmd = input().strip()
        if cmd.lower() == 'q':
            print("Arrêt du programme.")
            break

if __name__ == "__main__":
    main()
