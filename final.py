import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import speech_recognition as sr
import winsound

# Initialize the Snowball stemmer
stemmer = SnowballStemmer("english")
nltk.download('stopwords')

# Define audio capture parameters
RECORD_SECONDS = 2  # Duration of audio capture in seconds

# Load your custom model (replace 'your_model_filename' with the actual filename)
loaded_model = load_model("hate&abusive_model.h5")

# Load your custom tokenizer (replace 'your_tokenizer_filename' with the actual filename)
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# Define a threshold for offensive word detection
offensive_threshold = 0.5

# Perform continuous real-time speech recognition and processing
while True:
    # Perform speech recognition using the microphone
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening for 1 seconds...")

        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)

        try:
            # Listen to the audio input for 3 seconds and convert it to text
            audio = recognizer.listen(source, timeout=RECORD_SECONDS)
            transcribed_text = recognizer.recognize_google(audio)

            print("Recognized text:", transcribed_text)


            # Clean and preprocess the transcribed text
            def clean_text(text):
                text = str(text).lower()
                text = re.sub('\[.*?\]', '', text)
                text = re.sub('https?://\S+|www\.\S+', '', text)
                text = re.sub('<.*?>+', '', text)
                text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
                text = re.sub('\n', '', text)
                text = re.sub('\w*\d\w*', '', text)
                text = [word for word in text.split(' ') if word not in stopwords.words('english')]
                text = " ".join(text)
                text = [stemmer.stem(word) for word in text.split(' ')]
                text = " ".join(text)
                return text


            transcribed_text = clean_text(transcribed_text)

            # Tokenize and classify the transcribed text using your custom model
            seq = loaded_tokenizer.texts_to_sequences([transcribed_text])
            padded = pad_sequences(seq, maxlen=300)

            # Split the transcribed text into words
            words = transcribed_text.split()

            # Iterate through the words and check for offensive words
            for word in words:
                # Classify the word using your model (you may need to adapt this part)
                word_seq = loaded_tokenizer.texts_to_sequences([word])
                word_padded = pad_sequences(word_seq, maxlen=300)
                word_prediction = loaded_model.predict(word_padded)

                # Check if the word is offensive
                if word_prediction[0][0] >= offensive_threshold:
                    print(f"Offensive word detected: {word}")
                    # Use the built-in system e sound
                    winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 milliseconds

        except sr.WaitTimeoutError:
            print("Timeout: No speech detected after seconds.")
        except sr.UnknownValueError:
            print("Google Web Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech Recognition service; {e}")
