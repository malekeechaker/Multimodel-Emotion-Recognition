import speech_recognition as sr
import tensorflow as tf
from nltk.corpus import stopwords
import re
import nltk
nltk.download('stopwords')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as t
from transformers import BertTokenizer, TFBertModel
import speech_recognition as sr

# Charger le modèle d'émotion
loaded_model = tf.keras.models.load_model(
    ".\\emotion_detector-v2.h5", 
    custom_objects={"TFBertModel": TFBertModel}
)

# Mapping des labels
map_labels = {
    6: "neutral",
    2: "Joy",
    0: "Anger",
    5: "Surprise",
    1: "Fear",
    3: "Love",
    4: "Sadness",
}

# Fonction pour convertir le discours en texte
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Parlez maintenant...")
        try:
            audio = recognizer.listen(source, timeout=10)
            print("Conversion de l'audio en texte...")
            text = recognizer.recognize_google(audio, language="en-EN")  # Changez la langue si nécessaire
            print(f"Texte reconnu : {text}")
            return text
        except sr.UnknownValueError:
            print("Impossible de comprendre l'audio.")
            return None
        except sr.RequestError as e:
            print(f"Erreur avec le service de reconnaissance vocale : {e}")
            return None
        except Exception as e:
            print(f"Une erreur est survenue : {e}")
            return None
 
# remove digits and special characters
def remove_digits(text):
    pattern = r'[^a-zA-z.,!?/:;\"\'\s]' 
    #when the ^ is on the inside of []; we are matching any character that is not included in this expression within the []
    return re.sub(pattern, '', text)

def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)

def preprocess(text):
    """function to perform pre-processing lowercasing,removing special characters,keeping only important anonym tokens etc.. 
    text:str
    return:str
    """
    #Select english StopWords
    cachedStopWords = set(stopwords.words("english"))
    #add custom words
    cachedStopWords.update(('and','I','A','http','And','So','arnt','This','When','It','many','Many','so','cant','Yes','yes','No','no','These','these','mailto','regards','ayanna','like','email'))
    #remove stop words
    new_str = ' '.join([word for word in text.split() if word not in cachedStopWords]) 
    email = re.compile(r'[\w\.-]+@[\w\.-]+')
    links = re.compile(r'http')
    new_str = email.sub(r'',new_str)
    new_str = links.sub(r'',new_str)
    new_str = re.sub(r'\[.*?\]', '', new_str)
    new_str = re.sub(r'http\S+', '', new_str) # remove http links
    new_str = re.sub(r'bit.ly/\S+', '',new_str) #remove bit.ly links
    html = re.compile('<.*?>')
    new_str = html.sub(r'',new_str)
    new_str = remove_digits(new_str)
    new_str = remove_special_characters(new_str)
    new_str = new_str.lower()#lowercasing the text
    new_str = new_str.replace("  "," ")
    return new_str
# Fonction pour prédire l'émotion à partir du texte
def predict(model, tokenizer, text, max_len=60):
    if not text:
        print("Aucun texte à analyser.")
        return
    text = preprocess(text)
    # Prétraitement du texte
    inputs = tokenizer(
        text, 
        max_length=max_len, 
        padding='max_length', 
        truncation=True, 
        return_tensors='tf', 
        return_token_type_ids=False, 
        return_attention_mask=True
    )
    
    # Prédiction
    preds = model.predict([inputs["input_ids"], inputs["attention_mask"]])
    predicted_classes = tf.argmax(preds, axis=1).numpy()[0]
    emotion = map_labels.get(predicted_classes, "Unknown")
    
    print(f"Émotion prédite : {emotion}")
    return emotion

# Pipeline complet : Discours -> Texte -> Émotion
if __name__ == "__main__":
    # Charger le tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Adapter si un autre tokenizer est utilisé

    # Obtenir le texte à partir du discours
    text = speech_to_text()
    if text:
        predict(loaded_model, tokenizer, text)
