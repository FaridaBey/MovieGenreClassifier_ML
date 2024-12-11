# Flask Server Code
from flask import Flask, request, jsonify
import sqlite3
import csv
import io
import keras
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gensim.downloader as api
from scipy import sparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.utils import get_custom_objects


# dependency for loading model
@keras.saving.register_keras_serializable()
# @register_keras_serializable()
# Focal loss definition
# def focal_loss(alpha=0.25, gamma=2.0):
def loss(y_true, y_pred):
    alpha = 0.25
    gamma = 2.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_values(y_pred, K.epsilon(), 1 - K.epsilon())
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_loss_values = -alpha_t * (1 - p_t) ** gamma * K.log(p_t)
    return K.mean(focal_loss_values)
    # return loss

# Preprocessing text dependencies
# Load pretrained Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")
# Initialize resources for text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# Initialize App
app = Flask(__name__)

# Load the trained model
def load_model():
    try:
        my_model = keras.models.load_model("final_model.keras", custom_objects={'loss': loss})
        print("Model loaded successfully.")
        return my_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model
model = load_model()


# text Pre-processing function
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stop words
    text = ' '.join(word for word in text.split() if word not in stop_words)
    # Perform stemming
    text = ' '.join(stemmer.stem(word) for word in text.split())
    return text
# text embedding function
def get_word2vec_embedding(text, model):
    words = text.split()  # Assuming the text is already preprocessed
    word_vectors = [model[word] for word in words if word in model]
    if len(word_vectors) == 0:  # If no word is found in the model
        return np.zeros(model.vector_size)  # Return a zero vector
    return np.mean(word_vectors, axis=0)  # Average the word vectors to represent the document

# Scaling functions for numerical values
def scale_vote_average_gamma(new_value):
    mean = 2.760133248392033
    std = 3.268687809405368
    return (new_value - mean) / std
def scale_vote_count_rayleigh(new_value):
    std = 425.05211545647563
    return new_value / std
def scale_runtime_rayleigh(new_value):
    std = 59.076400395192515
    return new_value / std
def scale_popularity_normal(new_value):
    mean = 1.7550629691540536
    std = 10.006306108247278
    return (new_value - mean) / std

# classify function
@app.route('/classify', methods=['POST'])
def classify():

    # Get the raw CSV data from the POST request
    csv_data = request.data.decode('utf-8')

    csv_file = io.StringIO(csv_data)
    reader = csv.reader(csv_file)
    # Read the first row of the CSV data
    row = next(reader)

    # Extract features from the CSV row
    vote_average = float(row[0])
    vote_count = int(row[1])
    runtime = int(row[2])
    popularity = float(row[3])
    adult_encoded = int(row[4])
    status_encoded = list(map(int, row[5:11]))
    original_language_encoded = list(map(int, row[11:180]))
    spoken_languages_encoded = list(map(int, row[180:364]))
    production_countries_encoded = list(map(int, row[364:612]))
    description = row[612]
    keywords = row[613]
    movie_title = row[614]
    release_day = int(row[615])
    release_month = int(row[616])
    release_year = int(row[617])


    # Scale numerical columns based on distribution
    vote_average = scale_vote_average_gamma(vote_average)
    vote_count = scale_vote_count_rayleigh(vote_count)
    runtime = scale_runtime_rayleigh(runtime)
    popularity = scale_popularity_normal(popularity)

    #pre-processing text features
    preprocessed_disc = preprocess_text(description)
    preprocessed_keywords = preprocess_text(keywords)
    preprocessed_title = preprocess_text(movie_title)
    # Get embeddings for each feature
    overview_embedding = get_word2vec_embedding(preprocessed_disc, word2vec_model)
    keywords_embedding = get_word2vec_embedding(preprocessed_keywords, word2vec_model)
    title_embedding = get_word2vec_embedding(preprocessed_title, word2vec_model)

  
    ### release day ,month , year before overview after zimababe,   then overview keywords , title 
    numerical_features = [
        vote_average,
        vote_count,
        runtime,
        popularity
    ]
    # print("numerical vavg,vcnt,run,pop")
    # print(numerical_features)

    categorical_features = [
        adult_encoded,
        status_encoded,
        original_language_encoded,
        spoken_languages_encoded,
        production_countries_encoded
    ]
    # print(categorical_features)

    #flatten the encoded features
    flat_categorical_features = pd.Series(categorical_features).explode().astype(float).tolist()
    # print("categ adut,sts,oglang,spokn,prod")
    # print(flat_categorical_features)

    release_date = [ 
        release_day, 
        release_month, 
        release_year 
    ]
    # print("date")
    # print(release_date)

    text_features = [
        overview_embedding,
        keywords_embedding, 
        title_embedding 
    ]
   
     #flatten the encoded features
    flat_text = pd.Series(text_features).explode().astype(float).tolist()
    # print("txt okt")
    # print(flat_text)

    # Combine all features into one list
    all_features = numerical_features + flat_categorical_features + release_date + flat_text
    # Create a DataFrame from the combined list
    all_features_df = pd.DataFrame(all_features)
    # print("-----------------------------------------------------------------------------------------------")
    # print(all_features_df.shape)
    # print("-----------------------------------------------------------------------------------------------")
    # print(all_features_df)
    # Convert to NumPy array first
    all_features_array = all_features_df.values

    # Reshape it to (1, 1515) for one sample with 1515 features
    all_features_array = all_features_array.reshape(1, -1)


    # #testing
    # print ("heree")
    # print(all_features)
    

    # Make predictions using the model
    predictions = model.predict(all_features_array)  # Predict with the trained model
    all_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 
    'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 
    'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']
    # Assuming predictions[0] contains the predicted probabilities for each genre
    threshold = 0.5
    predicted_genres = [genre for genre, label in zip(all_genres, predictions[0]) if label > threshold]
    print(predictions)

    # Define mappings for status, languages, and countries
    status_list = ['Canceled', 'In Production', 'Planned', 'Post Production', 'Released', 'Rumored']
    original_language_list = ['aa', 'ab', 'af', 'ak', 'am', 'ar', 'as', 'ay', 'az', 'ba', 'be', 'bg', 'bi', 'bm', 'bn', 'bo', 'br', 'bs',
                            'ca', 'ce', 'ch', 'cn', 'co', 'cr', 'cs', 'cv', 'cy', 'da', 'de', 'dv', 'dz', 'el', 'en', 'eo', 'es', 'et',
                            'eu', 'fa', 'ff', 'fi', 'fj', 'fo', 'fr', 'fy', 'ga', 'gd', 'gl', 'gn', 'gu', 'gv', 'ha', 'he', 'hi', 'hr',
                            'ht', 'hu', 'hy', 'hz', 'ia', 'id', 'ie', 'ig', 'ii', 'is', 'it', 'iu', 'ja', 'jv', 'ka', 'kg', 'ki', 'kj',
                            'kk', 'kl', 'km', 'kn', 'ko', 'ks', 'ku', 'kv', 'kw', 'ky', 'la', 'lb', 'lg', 'li', 'ln', 'lo', 'lt', 'lv',
                            'mg', 'mh', 'mi', 'mk', 'ml', 'mn', 'mo', 'mr', 'ms', 'mt', 'my', 'nb', 'nd', 'ne', 'nl', 'nn', 'no', 'nv',
                            'ny', 'oc', 'om', 'or', 'os', 'pa', 'pl', 'ps', 'pt', 'qu', 'rm', 'rn', 'ro', 'ru', 'rw','sa', 'sc', 'sd',
                            'se', 'sg', 'sh', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'ss', 'st', 'su', 'sv', 'sw', 'ta', 'te',
                            'tg', 'th', 'ti', 'tk', 'tl', 'tn', 'to', 'tr', 'ts', 'tt', 'tw', 'ty', 'ug', 'uk', 'ur', 'uz','vi','wo','xh',
                            'xx', 'yi', 'yo','za', 'zh', 'zu']
    spoken_languages_list = ['Abkhazian', 'Afar', 'Afrikaans', 'Akan', 'Albanian', 'Amharic', 'Arabic', 'Aragonese', 'Armenian', 'Assamese', 'Avaric', 'Avestan', 'Aymara', 'Azerbaijani',
          'Bambara', 'Bashkir', 'Basque', 'Belarusian', 'Bengali', 'Bislama', 'Bosnian', 'Breton', 'Bulgarian', 'Burmese',
          'Cantonese', 'Catalan', 'Chamorro', 'Chechen', 'Chichewa; Nyanja', 'Chuvash', 'Cornish', 'Corsican', 'Cree', 'Croatian', 'Czech',
          'Danish', 'Divehi', 'Dutch', 'Dzongkha',
          'English', 'Esperanto', 'Estonian', 'Ewe',
          'Faroese', 'Fijian', 'Finnish', 'French', 'Frisian', 'Fulah',
          'Gaelic', 'Galician', 'Ganda', 'Georgian', 'German', 'Greek', 'Guarani', 'Gujarati',
          'Haitian; Haitian Creole', 'Hausa', 'Hebrew', 'Herero', 'Hindi', 'Hiri Motu', 'Hungarian',
          'Icelandic', 'Ido', 'Igbo', 'Indonesian', 'Interlingua', 'Interlingue', 'Inuktitut', 'Inupiaq', 'Irish', 'Italian',
          'Japanese', 'Javanese',
          'Kalaallisut', 'Kannada', 'Kanuri', 'Kashmiri', 'Kazakh', 'Khmer', 'Kikuyu', 'Kinyarwanda', 'Kirghiz', 'Komi', 'Kongo', 'Korean', 'Kuanyama', 'Kurdish',
          'Lao', 'Latin', 'Latvian', 'Letzeburgesch', 'Limburgish', 'Lingala', 'Lithuanian', 'Luba-Katanga',
          'Macedonian', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Mandarin', 'Maori', 'Marathi', 'Marshall', 'Moldavian', 'Mongolian',
          'Nauru', 'Navajo', 'Ndebele', 'Ndonga', 'Nepali', 'No Language', 'Northern Sami', 'Norwegian', 'Norwegian Bokmål', 'Norwegian Nynorsk',
          'Occitan', 'Ojibwa', 'Oriya', 'Oromo', 'Ossetian; Ossetic',
          'Pali', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Pushto',
          'Quechua', 'Raeto-Romance', 'Romanian', 'Rundi',
          'Russian', 'Samoan', 'Sango', 'Sanskrit', 'Sardinian', 'Serbian', 'Serbo-Croatian', 'Shona', 'Sindhi', 'Sinhalese', 'Slavic',
          'Slovak', 'Slovenian', 'Somali', 'Sotho', 'Spanish', 'Sundanese', 'Swahili', 'Swati', 'Swedish',
          'Tagalog', 'Tahitian', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tibetan', 'Tigrinya', 'Tonga', 'Tsonga', 'Tswana', 'Turkish', 'Turkmen', 'Twi',
          'Uighur', 'Ukrainian', 'Urdu', 'Uzbek', 'Venda', 'Vietnamese', 'Volapük', 'Walloon', 'Welsh', 'Wolof', 'Xhosa', 'Yi', 'Yiddish', 'Yoruba', 'Zulu']

    production_countries_list = ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan',
          'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil',
          'British Indian Ocean Territory', 'British Virgin Islands', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi',
          'Cambodia', 'Cameroon', 'Canada', 'Cape Verde', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 'Christmas Island',
          'Cocos  Islands', 'Colombia', 'Comoros', 'Congo', 'Cook Islands', 'Costa Rica', "Cote D'Ivoire", 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic',
          'Czechoslovakia', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'East Germany', 'East Timor', 'Ecuador', 'Egypt', 'El Salvador',
          'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Faeroe Islands', 'Falkland Islands', 'Fiji', 'Finland', 'France', 'French Guiana',
          'French Polynesia', 'French Southern Territories', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada',
          'Guadaloupe', 'Guam', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Heard and McDonald Islands', 'Holy See', 'Honduras', 'Hong Kong',
          'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
          'Kosovo', 'Kuwait', 'Kyrgyz Republic', "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libyan Arab Jamahiriya', 'Liechtenstein',
          'Lithuania', 'Luxembourg', 'Macao', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Martinique',
          'Mauritania', 'Mauritius', 'Mayotte', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique',
          'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'Netherlands Antilles', 'New Caledonia', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue',
          'Norfolk Island', 'North Korea', 'Northern Ireland', 'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestinian Territory', 'Panama',
          'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn Island', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Reunion', 'Romania', 'Russia',
          'Rwanda', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Serbia and Montenegro', 'Seychelles', 'Sierra Leone',
          'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Georgia and the South Sandwich Islands', 'South Korea',
          'South Sudan', 'Soviet Union', 'Spain', 'Sri Lanka', 'St. Helena', 'St. Kitts and Nevis', 'St. Lucia', 'St. Pierre and Miquelon', 'St. Vincent and the Grenadines',
          'Sudan', 'Suriname', 'Svalbard & Jan Mayen Islands', 'Swaziland', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste',
          'Togo', 'Tokelau', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu', 'US Virgin Islands', 'Uganda', 'Ukraine', 'United Arab Emirates',
          'United Kingdom', 'United States Minor Outlying Islands', 'United States of America', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Wallis and Futuna Islands', 'Western Sahara', 'Yemen',
          'Yugoslavia', 'Zaire', 'Zambia', 'Zimbabwe']
    # Map encoded values to actual category names
    status = [status for status, encoded in zip(status_list, status_encoded) if encoded == 1]
    original_languages = [language for language, encoded in zip(original_language_list, original_language_encoded) if encoded == 1]
    spoken_languages = [language for language, encoded in zip(spoken_languages_list, spoken_languages_encoded) if encoded == 1]
    production_countries = [country for country, encoded in zip(production_countries_list, production_countries_encoded) if encoded == 1]
    # Convert the list of predicted genres to a comma-separated string
    predicted_genres_str = ', '.join(predicted_genres)
    status_str = ', '.join(status)
    original_languages_str = ', '.join(original_languages)
    spoken_languages_str = ', '.join(spoken_languages)
    production_countries_str = ', '.join(production_countries)

    # Save input and predictions to the database
    with sqlite3.connect('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO classifications (title, description, runtime, vote_average, vote_count, popularity,
                status, original_language, spoken_languages, production_countries, release_day, release_month, 
                release_year, genres)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (movie_title, description, runtime, vote_average, vote_count, popularity,
             status_str, original_languages_str, spoken_languages_str, production_countries_str,
             release_day, release_month, release_year, predicted_genres_str)
        )
        conn.commit()


    #Retraining database insert
    # Establish a connection to the SQLite database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Insert data directly without placeholders (be cautious with this method)
    for row in all_features_array:
        cursor.execute(f'''
            INSERT INTO movie_features VALUES ({', '.join(map(str, row))})
        ''')

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    return jsonify({"predicted_genres": predicted_genres})

# View past predictions
@app.route('/history', methods=['GET'])
def view_predictions():
    with sqlite3.connect('app.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM classifications')
        rows = cursor.fetchall()
        
        # Convert rows to a list of dictionaries for easy JSON response
        predictions = []
        for row in rows:
            prediction = {
                "id": row[0],
                "title": row[1],
                "description": row[2],
                "runtime": row[3],
                "vote_average": row[4],
                "vote_count": row[5],
                "popularity": row[6],
                "status": row[7],
                "original_language": row[8],
                "spoken_languages": row[9],
                "production_countries": row[10],
                "release_day": row[11],
                "release_month": row[12],
                "release_year": row[13],
                "genres": row[14].split(',')
            }
            predictions.append(prediction)

    return jsonify(predictions)


if __name__ == '__main__':
    # Create the database if not exists
    conn = sqlite3.connect('app.db')
    cursor = conn.cursor()
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS classifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        runtime INTEGER,
        vote_average REAL,
        vote_count INTEGER,
        popularity REAL,
        status TEXT,
        original_language TEXT,
        spoken_languages TEXT,
        production_countries TEXT,
        release_day INTEGER,
        release_month INTEGER,
        release_year INTEGER,
        genres TEXT
    )
    ''')
    conn.commit()

    app.run(debug=True)


