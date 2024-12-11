import tkinter as tk
from tkinter import ttk, messagebox
import requests
from tkinter import PhotoImage
import csv
import io

class MovieGenreApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Movie Genre Classification")

        # Initialize the intro page
        self.intro_page()

    def intro_page(self):
        # Clear any existing widgets
        for widget in self.root.winfo_children():
            widget.destroy()


        # Intro frame with an image
        intro_frame = tk.Frame(self.root)
        intro_frame.pack(padx=20, pady=20)

        # Welcome message
        welcome_label = tk.Label(intro_frame, text="Welcome to the Movie Genre Classifier!",
                                 font=("Courier", 14))
        welcome_label.grid(row=0, column=0, columnspan=2, pady=10)

        self.image = PhotoImage(file="icon.png")
        self.image = self.image.subsample(2, 2)

        image_label = tk.Label(intro_frame, image=self.image)
        image_label.grid(row=1, column=0, columnspan=2, pady=10)

        # Buttons
        btn_view_history = ttk.Button(intro_frame, text="View History", command=self.view_history_page)
        btn_view_history.grid(row=2, column=0, padx=10, pady=10)

        btn_classify_movie = ttk.Button(intro_frame, text="Classify a New Movie", command=self.classify_movie_page)
        btn_classify_movie.grid(row=2, column=1, padx=10, pady=10)


# classification page
    def classify_movie_page(self):
        # Clear existing widgets and create classification page
        for widget in self.root.winfo_children():
            widget.destroy()
        # label
        userlabel = tk.Label(text="Please enter your movie's data",
                                 font=("Courier", 15,"bold"),fg="red")
        userlabel.grid(row=0, column=1,columnspan=1, pady=10)

        # Input fields
        # text features
        # title
        ttk.Label(self.root, text="Movie Title:", font=("Courier", 13, "bold")).grid(row=1, column=0, padx=10, pady=5)
        self.title_entry = ttk.Entry(self.root)
        self.title_entry.grid(row=1, column=1, padx=10, pady=5)

        # overview
        ttk.Label(self.root, text="Description:", font=("Courier", 13, "bold")).grid(row=2, column=0, padx=10, pady=5)
        self.description_entry = ttk.Entry(self.root, width=40)
        self.description_entry.grid(row=2, column=1, padx=10, pady=5)

        # keywords
        ttk.Label(self.root, text="Keywords:", font=("Courier", 13, "bold")).grid(row=3, column=0, padx=10, pady=5)
        self.keywords_entry = ttk.Entry(self.root)
        self.keywords_entry.grid(row=3, column=1, padx=10, pady=5)

        #Numerical features
        # runtime
        ttk.Label(self.root, text="Runtime (min):", font=("Courier", 13, "bold")).grid(row=4, column=0, padx=10, pady=5)
        self.runtime_entry = ttk.Entry(self.root)
        self.runtime_entry.grid(row=4, column=1, padx=10, pady=5)

        # vote_avgerage
        ttk.Label(self.root, text="Vote Average:", font=("Courier", 13, "bold")).grid(row=5, column=0, padx=10, pady=5)
        self.vote_average_entry = ttk.Entry(self.root)
        self.vote_average_entry.grid(row=5, column=1, padx=10, pady=5)

        # vote_count
        ttk.Label(self.root, text="Vote Count:", font=("Courier", 13, "bold")).grid(row=6, column=0, padx=10, pady=5)
        self.vote_count_entry = ttk.Entry(self.root)
        self.vote_count_entry.grid(row=6, column=1, padx=10, pady=5)

        # popularity
        ttk.Label(self.root, text="Popularity:", font=("Courier", 13, "bold")).grid(row=7, column=0, padx=10, pady=5)
        self.popularity_entry = ttk.Entry(self.root)
        self.popularity_entry.grid(row=7, column=1, padx=10, pady=5)


        # Dropdown menus for the categorical features
        # status
        ttk.Label(self.root, text="Status:", font=("Courier", 13, "bold")).grid(row=8, column=0, padx=10, pady=5)
        self.status_var = tk.StringVar()
        self.status_menu = ttk.Combobox(self.root, textvariable=self.status_var,
                                        values=["Rumored", "Released", "Post Production", "Planned", "In Production", "Canceled"])
        self.status_menu.grid(row=8, column=1, padx=10, pady=5)

        # original language
        ttk.Label(self.root, text="Original Language:", font=("Courier", 13, "bold")).grid(row=9, column=0, padx=10, pady=5)
        self.original_language_var = tk.StringVar()
        self.original_language_menu = ttk.Combobox(self.root, textvariable=self.original_language_var,
                                                   values=['aa', 'ab', 'af', 'ak', 'am', 'ar', 'as', 'ay', 'az',
                                                           'ba', 'be', 'bg', 'bi', 'bm', 'bn', 'bo', 'br', 'bs',
                                                           'ca', 'ce', 'ch', 'cn', 'co', 'cr', 'cs', 'cv', 'cy',
                                                           'da', 'de', 'dv', 'dz',
                                                           'el', 'en', 'eo', 'es', 'et', 'eu',
                                                           'fa', 'ff', 'fi', 'fj', 'fo', 'fr', 'fy',
                                                           'ga', 'gd', 'gl', 'gn', 'gu', 'gv',
                                                           'ha', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'hz',
                                                           'ia', 'id', 'ie', 'ig', 'ii', 'is', 'it', 'iu',
                                                           'ja', 'jv',
                                                           'ka', 'kg', 'ki', 'kj', 'kk', 'kl', 'km', 'kn', 'ko', 'ks', 'ku', 'kv', 'kw', 'ky',
                                                           'la', 'lb', 'lg', 'li', 'ln', 'lo', 'lt', 'lv',
                                                           'mg', 'mh', 'mi', 'mk', 'ml', 'mn', 'mo', 'mr', 'ms', 'mt', 'my',
                                                           'nb', 'nd', 'ne', 'nl', 'nn', 'no', 'nv', 'ny',
                                                           'oc', 'om', 'or', 'os',
                                                           'pa', 'pl', 'ps', 'pt',
                                                           'qu', 'rm', 'rn', 'ro', 'ru', 'rw','sa', 'sc', 'sd', 'se', 'sg', 'sh', 'si',
                                                           'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'ss', 'st', 'su', 'sv', 'sw',
                                                           'ta', 'te', 'tg', 'th', 'ti', 'tk', 'tl', 'tn', 'to', 'tr', 'ts', 'tt', 'tw', 'ty',
                                                           'ug', 'uk', 'ur', 'uz','vi','wo','xh', 'xx', 'yi', 'yo','za', 'zh', 'zu'])
        self.original_language_menu.grid(row=9, column=1, padx=10, pady=5)

        # Dropdown menus with checkboxes

        # Spoken Languages
        ttk.Label(self.root, text="Spoken Languages:", font=("Courier", 13, "bold")).grid(row=10, column=0, padx=10, pady=5)
        self.spoken_languages_var = tk.StringVar(value="Select Spoken Languages")
        self.spoken_languages_button = ttk.Button(self.root, textvariable=self.spoken_languages_var,
                                                  command=self.show_spoken_languages_menu)
        self.spoken_languages_button.grid(row=10, column=1, padx=10, pady=5)

        #Production Countries
        ttk.Label(self.root, text="Production Countries:", font=("Courier", 13, "bold")).grid(row=11, column=0, padx=10, pady=5)
        self.production_countries_var = tk.StringVar(value="Select Production Countries")
        self.production_countries_button = ttk.Button(self.root, textvariable=self.production_countries_var,
                                                      command=self.show_production_countries_menu)
        self.production_countries_button.grid(row=11, column=1, padx=10, pady=5)

        #  #  #  #  #  #  #  #  #  #  #  #  #  #   #  #  #  #  #  #  #  #  #  #  #

        # adult
        ttk.Label(self.root, text="Adult Content:", font=("Courier", 13, "bold")).grid(row=12, column=0, padx=10, pady=5)
        self.adult_var = tk.StringVar()
        self.adult_menu = ttk.Combobox(self.root, textvariable=self.adult_var, values=["True", "False"])
        self.adult_menu.grid(row=12, column=1, padx=10, pady=5)

        # Release Date Inputs (Day, Month, Year)
        ttk.Label(self.root, text="Release Day:", font=("Courier", 13, "bold")).grid(row=13, column=0, padx=10, pady=5)
        self.release_day_var = tk.StringVar()
        self.release_day_menu = ttk.Combobox(self.root, textvariable=self.release_day_var,
                                             values=[str(i) for i in range(1, 32)])  # Days 1-31
        self.release_day_menu.grid(row=13, column=1, padx=10, pady=5)

        ttk.Label(self.root, text="Release Month:", font=("Courier", 13, "bold")).grid(row=14, column=0, padx=10, pady=5)
        self.release_month_var = tk.StringVar()
        self.release_month_menu = ttk.Combobox(self.root, textvariable=self.release_month_var,
                                               values=[str(i) for i in range(1, 13)])  # Months 1-12
        self.release_month_menu.grid(row=14, column=1, padx=10, pady=5)

        ttk.Label(self.root, text="Release Year:", font=("Courier", 13, "bold")).grid(row=15, column=0, padx=10, pady=5)
        self.release_year_var = tk.StringVar()
        self.release_year_menu = ttk.Combobox(self.root, textvariable=self.release_year_var,
                                              values=[str(i) for i in range(1900, 2025)])  # Years from 1900 to 2024
        self.release_year_menu.grid(row=15, column=1, padx=10, pady=5)

        # Buttons
        ttk.Button(self.root, text="Classify Genre", command=self.classify_genre).grid(row=16, column=0, columnspan=1, pady=10)
        ttk.Button(self.root, text="View History", command=self.view_history_page).grid(row=16, column=1, columnspan=1, pady=10)
        ttk.Button(self.root, text="Back to Home", command=self.intro_page).grid(row=16, column=2, columnspan=1, pady=10)


        # Result display
        self.result_label = ttk.Label(self.root, text="Predicted Genres:", font=("Courier", 15,"bold"), foreground="navyblue")
        self.result_label.grid(row=17, column=0, columnspan=2, pady=10)


        
    #Helper Functions for the dropdown checkbox
    def show_spoken_languages_menu(self):
        self.show_checkbox_menu(
            "Select Spoken Languages",
            ['Abkhazian', 'Afar', 'Afrikaans', 'Akan', 'Albanian', 'Amharic', 'Arabic', 'Aragonese', 'Armenian', 'Assamese', 'Avaric', 'Avestan', 'Aymara', 'Azerbaijani',
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
          'Uighur', 'Ukrainian', 'Urdu', 'Uzbek', 'Venda', 'Vietnamese', 'Volapük', 'Walloon', 'Welsh', 'Wolof', 'Xhosa', 'Yi', 'Yiddish', 'Yoruba', 'Zulu'],
            self.spoken_languages_var
        )

    def show_production_countries_menu(self):
        self.show_checkbox_menu(
            "Select Production Countries",
            ['Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 'Azerbaijan',
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
          'Yugoslavia', 'Zaire', 'Zambia', 'Zimbabwe'],
            self.production_countries_var
        )

    def show_checkbox_menu(self, title, options, var):
        # Create a new top-level window
        menu_window = tk.Toplevel(self.root)
        menu_window.title(title)
        menu_window.grab_set()  # Modal window

        # Scrollable area
        canvas = tk.Canvas(menu_window, width=300, height=400)
        scrollable_frame = ttk.Frame(canvas)
        scrollbar = ttk.Scrollbar(menu_window, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        # Keep track of selected options
        selected_options = []

        def update_selected():
            var.set(", ".join(selected_options) if selected_options else "None")

        # Generate checkboxes
        for option in options:
            var_option = tk.BooleanVar(value=False)

            def toggle_option(opt=option, var_opt=var_option):
                if var_opt.get():
                    selected_options.append(opt)
                else:
                    selected_options.remove(opt)
                update_selected()

            checkbutton = ttk.Checkbutton(scrollable_frame, text=option, variable=var_option, command=toggle_option)
            checkbutton.pack(anchor="w", padx=10, pady=5)

        scrollable_frame.update_idletasks()  # Needed to adjust the scrollable frame's size
        canvas.config(scrollregion=canvas.bbox("all"))

        # Done button
        ttk.Button(menu_window, text="Done", command=menu_window.destroy).pack(pady=10)


    def classify_genre(self):
        # Collect data from fields
        title = self.title_entry.get()
        description = self.description_entry.get()
        keywords = self.keywords_entry.get()
        runtime = self.runtime_entry.get()
        vote_average = self.vote_average_entry.get()
        vote_count = self.vote_count_entry.get()
        popularity = self.popularity_entry.get()
        status = self.status_var.get()
        original_language = self.original_language_var.get()
        
        # Process Spoken Languages
        spoken_languages = self.spoken_languages_var.get()
        if spoken_languages == "Select Spoken Languages" or spoken_languages == "None":
            spoken_languages = []  # No selection
        else:
            spoken_languages = [lang.strip() for lang in spoken_languages.split(",")]
        # Process Production Countries
        production_countries = self.production_countries_var.get()
        if production_countries == "Select Production Countries" or production_countries == "None":
            production_countries = []  # No selection
        else:
            production_countries = [country.strip() for country in production_countries.split(",")]

        adult = self.adult_var.get()
        release_day = self.release_day_var.get()
        release_month = self.release_month_var.get()
        release_year = self.release_year_var.get()

        # Validate input
        if (not title or not description or not runtime or not keywords or not vote_average or not vote_count
            or not popularity or not status or not original_language or not spoken_languages or not production_countries
            or not adult or not release_day or not release_month or not release_year):
            messagebox.showerror("Input Error", "Please fill all fields.")
            return

        try:
            runtime = int(runtime)
            vote_average = float(vote_average)
            vote_count = int(vote_count)
            popularity = float(popularity)
        except ValueError:
            messagebox.showerror("Input Error", "Runtime, Vote Average, Vote Count, Popularity must be numbers.")
            return

        # One-hot encoding for categorical fields
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


        # Encode features
        original_language_encoded = [1 if lang == original_language else 0 for lang in original_language_list]
        spoken_languages_encoded = [1 if lang in spoken_languages else 0 for lang in spoken_languages_list]
        production_countries_encoded = [1 if country in production_countries else 0 for country in production_countries_list]
        status_encoded = [1 if stat == status else 0 for stat in status_list]
        adult_encoded = 1 if adult == "True" else 0

        # Prepare CSV data (without header) using io.StringIO to create CSV as string
        output = io.StringIO()
        writer = csv.writer(output)

        # Prepare CSV payload
        csv_data = [vote_average, vote_count, runtime, popularity, adult_encoded, *status_encoded,
                    *original_language_encoded, *spoken_languages_encoded, *production_countries_encoded,
                    description, keywords, title, release_day, release_month, release_year]

        # Write data to the CSV string
        writer.writerow(csv_data)
        csv_string = output.getvalue()
        output.close()

        # #Testing
        # print("Here:")
        # print(csv_string)

        # Send data to the server as raw CSV string (without writing to a file)
        try:
            response = requests.post("http://127.0.0.1:5000/classify", data=csv_string,
                                     headers={"Content-Type": "text/csv"})
            response_data = response.json()
            self.result_label.config(text=f"Predicted Genres: {', '.join(response_data['predicted_genres'])}")
        except Exception as e:
            messagebox.showerror("Server Error", f"Failed to connect to server: {e}")




    def view_history_page(self):
        # Clear existing widgets and create history page
        for widget in self.root.winfo_children():
            widget.destroy()
        # label
        histlabel = tk.Label(text="Previous Predictions:",
                             font=("Courier", 12))
        histlabel.grid(row=0, column=0, columnspan=2, pady=10)
        # Fetch history from the server
        try:
            response = requests.get("http://127.0.0.1:5000/history")
            history_data = response.json()
           # Display history data in the same window
            row = 1  # Starting from row 1 for history data
            for record in history_data:
                title_label = tk.Label(self.root, text=f"Title: {record['title']}", font=("Courier", 10, "bold"))
                title_label.grid(row=row, column=0, sticky="w", padx=10, pady=5)
                genres_label = tk.Label(self.root, text=f"Genres: {record['genres']}", font=("Courier", 10, "bold"))
                genres_label.grid(row=row, column=1, sticky="w", padx=10, pady=5)
                status_label = tk.Label(self.root, text=f"Status: {record['status']}", font=("Courier", 10, "bold"))
                status_label.grid(row=row, column=2, sticky="w", padx=10, pady=5)
                row += 1

        except requests.exceptions.RequestException as e:
            error_label = tk.Label(self.root, text=f"Error fetching history: {str(e)}", font=("Courier", 10), fg="red")
            error_label.grid(row=1, column=0, columnspan=3, pady=10)

        ttk.Button(self.root, text="Back to Home", command=self.intro_page).grid(row=16, column=2, columnspan=1,
                                                                                 pady=10)


if __name__ == "__main__":
    root = tk.Tk()
    app = MovieGenreApp(root)
    root.mainloop()
