#!/usr/bin/env python3
import os
import json
import argparse
import base64
import time
import csv
import re
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

# Initialize OpenAI client with OpenRouter base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Define the models to use for localization - using specified model IDs
MODELS = {
    "1-gemini flash 1.5 8B": "google/gemini-flash-1.5-8b",  # Updated based on example
    "2-gemini flash 2.0": "google/gemini-2.0-flash-001",
    "3-GPT 4-o mini": "openai/gpt-4o-mini",
    "4-GPT 4.1": "openai/gpt-4.1-mini",
    "5-Claude 3.7 Sonnet": "anthropic/claude-3-7-sonnet",
    "6-Grok 3": "x-ai/grok-3-beta"  # Updated based on example
}

# Vision model for image descriptions
VISION_MODEL = "google/gemini-pro-vision"

# Define language codes for CSV columns
LANGUAGE_CODES = {
    "turkish": "tr",
    "french": "fr",
    "german": "de"
}

# Create OpenAI client configured for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

def encode_image(image_path):
    """Encode image to base64 for API request"""
    with open(image_path, "rb") as image_file:
        return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

def get_image_dimensions(image_path):
    """Get dimensions of the image"""
    with Image.open(image_path) as img:
        return img.size

def get_image_path(imgs_dir, image_id):
    """Get the path to an image based on its ID"""
    if not image_id:
        return None
        
    # Convert image_id to string to handle potential numeric IDs
    image_id = str(image_id).strip()
    if not image_id:
        return None
        
    # Look for files with patterns like 1.png, 01.png, or similar
    pattern = re.compile(r"^0*" + image_id + r"\.\w+$")
    
    for filename in os.listdir(imgs_dir):
        if pattern.match(filename):
            return os.path.join(imgs_dir, filename)
    
    # If no specific match found, try other common patterns
    pattern = re.compile(r".*?" + image_id + r"[^0-9].*")
    for filename in os.listdir(imgs_dir):
        if pattern.match(filename):
            return os.path.join(imgs_dir, filename)
    
    return None

def get_image_description(image_path, debug=False):
    """Get description of image using Gemini Pro Vision"""
    print(f"\n🔍 Getting image description for {os.path.basename(image_path)}...")
    
    if debug:
        print("  DEBUG MODE: Returning mock description instead of calling API")
        return f"This is a debug description for image {os.path.basename(image_path)}"
    
    # Encode image to base64
    image_url = encode_image(image_path)
    
    system_prompt = """
    You are a detailed image description expert.
    
    Examine the game screenshot provided and create a comprehensive description that includes:
    
    1. What is happening in the image - describe all visual elements
    2. The puzzle or challenge shown in the image
    3. Any text visible in the image
    4. The game mechanics or logic required to solve the puzzle (if apparent)
    5. The overall theme or context of the game screen
    
    Your description should be detailed enough to help someone understand the image without seeing it.
    The description will be used to guide localization efforts for a brain teaser game.
    """
    
    try:
        # Create request using OpenAI client for OpenRouter
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://cascade.ai",  # Site URL for rankings
                "X-Title": "Game Localization Tool",   # Site title for rankings
            },
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]}
            ]
        )
        
        # Extract description
        description = completion.choices[0].message.content
        
        print("✓ Successfully obtained image description")
        return description
        
    except Exception as e:
        print(f"✗ Error getting image description: {str(e)}")
        return "Error: Could not generate image description"

def process_localization(description, english_text, model_id, model_name, debug=False):
    """Process localization using a specific model"""
    print(f"\n🔄 Processing localization with model: {model_id}")
    
    if debug:
        print("  DEBUG MODE: Returning mock translations instead of calling API")
        return {
            "localization": {
                "turkish": f"[TR] {english_text}",
                "french": f"[FR] {english_text}",
                "german": f"[DE] {english_text}"
            }
        }
    
    # Context prompt explaining what we want
    system_prompt = f"""
    You are a game localization translator expert.

    You have been provided with an image description and English text from a 'Brain Test' puzzle game.
    You can use the following information for localization:
        
        You're localizing a 'Brain Test' named children's brain teaser game that uses word play, 
        Brain Test is childen's brain teaser game that is popular worldwide, known for its tricky and often unexpected brain teasers designed to challenge players' logic and intelligence. 
        Emphasizing that appearances can be deceiving, the game encourages players to think outside the box while offering a hint system for help. 
        Developed by Unico Studio, Brain Test games aim to capture the atmosphere loved by players and offer an addictive, free and family-friendly experience that can be enjoyed offline as a relaxing brain workout.
        
        also you can use the following information for localization:
        
        1. The game is a children's brain teaser game that uses word play.
        2. The game is popular worldwide, known for its tricky and often unexpected brain teasers designed to challenge players' logic and intelligence.
        3. The game encourages players to think outside the box while offering a hint system for help.
        4. The game is developed by Unico Studio.
        5. The game is free and family-friendly.
        6. The game is offline and can be enjoyed as a relaxing brain workout.
        7. The game is a children's brain teaser game that uses word play.
        
        more information about the game:
        
        https://play.google.com/store/apps/details?id=com.unicostudio.braintest&hl=tr
    Image Description:
    {description}
    
    Your task is to provide culturally-appropriate localizations of the English text in:
    Turkish, French, and German. For localization, use cultural references, idioms, and wordplay specific to each language. Do NOT provide direct translations.
    Examine the game screenshot and the corresponding English text carefully. 
    
    These should preserve the game mechanics, humor, and puzzle elements but adapt them to feel natural 
    in each target language. Use cultural references, idioms, and wordplay specific to each language.

    For best localization results, use the following approach:
    - Use idioms and expressions that are common in the target language
    - Use cultural references that are relevant to the target language
    - Use humor that is appropriate for the target language
    - If you need more specific details, use the description to guide your localization.

    Localization results should not be too long, close to the length of the English input.
    The number of input sentences must be the same as the number of output sentences.

    Do not use all the content written in the description in exactly the same way. 
    Description only may help you for localization.

    Localization results should not be too long, close to the length of the English input.
    The number of input sentences must be the same as the number of output sentences
    For example, if the English text is 1 sentence, the localization output should be 1 sentence. do not output very long texts 
    
    Example localization for you:
    
    English text :	Some parts of the board are covered.	localizated in turkish :	Tahtanın bir kısmı görünmüyor.	localized in german :	Einige Teile der Tafel sind verdeckt.	localizated in french :	Certaines parties du tableau sont cachées.
English text :	Drag the curtain above to reveal what is x.	localizated in turkish :	Perdeyi kaldırıp x'i görebilirsin.	localized in german :	Zieh den Vorhang hoch, um das x zu zeigen.	localizated in french :	Remontez le store pour découvrir la valeur de x.
English text :	Are you insulting me with this easy question?!	localizated in turkish :	Böyle basit bir soruyla dalga mı geçiyorsun benimle?!	localized in german :	Willst du mich mit dieser trivialen Frage beleidigen?	localizated in french :	C’est tellement facile que je suis vexée !
English text :	I just wanted to make sure that you know basic math.	localizated in turkish :	Kızma ya! Önce bi matematik temelini ölçmek istedim.	localized in german :	Ich wollte nur sicherstellen, dass du Grundlagenmathe kannst.	localizated in french :	Je voulais vérifier si tu avais les bases en maths.
English text :	Doctor Worry must get some venom for his new medicine.	localizated in turkish :	Doktor Civan'ın yeni ilacı için yılan zehrine ihtiyacı var.	localized in german :	Dr. Kummer muss Gift für seine neue Medizin abschöpfen.	localizated in french :	Le Docteur Névrose a besoin de venin pour préparer un traitement.
English text :	The snake spits venom if a prey approaches it.	localizated in turkish :	Yılana yiyecek yaklaştırırsan zehrini salgılar.	localized in german :	Die Schlange spuckt Gift, wenn sich eine Beute nähert.	localizated in french :	Le serpent crache du venin quand une proie s’en approche.
English text :	Drag the mouse close to the snake.	localizated in turkish :	Fareyi yılana yaklaştır.	localized in german :	Ziehe die Maus nah an die Schlange heran.	localizated in french :	Amenez la souris près du serpent.
English text :	Drag the empty bottle to collect the venom.	localizated in turkish :	Boş şişeyle yere dökülen zehri topla.	localized in german :	Ziehe dir leere Flasche, um das Gift zu sammeln.	localizated in french :	Utilisez la fiole vide pour prendre du venin.
English text :	I will use this venom to make a medicine to cure the Kardashian fans.	localizated in turkish :	Bu zehri kullanarak yapacağım ilaç ile halkımızı Müge Anlı izleme hastalığından kurtaracağım!	localized in german :	Ich werde dieses Gift verwenden, um eine Medizin zu erschaffen, welche die Helene Fischer-Fans heilen soll.	localizated in french :	Je vais préparer un traitement pour soigner les fans des Marseillais ! 
English text :	This man wants some hair.	localizated in turkish :	Adam başında saç istiyor.	localized in german :	Dieser Mann will Haar.	localizated in french :	Il rêve d’avoir des cheveux.
English text :	Drag some seeds on top of his head.	localizated in turkish :	Tohumları kafasına ek.	localized in german :	Ziehe Samen auf seinen Kopf.	localizated in french :	Mettez des graines sur sa tête.
English text :	Water it after putting the seeds.	localizated in turkish :	Tohumları koyduktan sora başını sula.	localized in german :	Gieße sie nachdem du die Samen hingezogen hast.	localizated in french :	Arrosez les graines.
English text :	You can stop worrying about your hair now. There are 100s of more things to worry after all.	localizated in turkish :	İşte ben buna çim adam derim!	localized in german :	Ich verschreibe dir Dünger und viel Sonnenlicht.	localizated in french :	Arrêtez de penser à ses cheveux. Il y a plus grave dans le monde.
English text :	What's Uncle Bubba's secret?	localizated in turkish :	Şakir Dayının sırrını keşfet.	localized in german :	Was ist Onkel Bubbas Geheimnis?	localizated in french :	Quel est le secret de Tonton Tanguy ?
English text :	Try to find his tattoo.	localizated in turkish :	Gizli dövmesini bul.	localized in german :	Finde sein Tattoo.	localizated in french :	Cherchez son tatouage.
English text :	Use Doctor Worry's x-ray glasses.	localizated in turkish :	Doktor Civan'ın gözlüğünü bir x-ray cihazı gibi kullanabilirsin.	localized in german :	Verwende Dr. Kummers Röntgenbrille.	localizated in french :	Utilisez les lunettes à rayons X du docteur Névrose.
English text :	The tattoo is hidden somewhere around his belly.	localizated in turkish :	Dövme göbeğinin oralarda bir yerlerde.	localized in german :	Das Tattoo ist irgendwo bei seinem Bauch versteckt.	localizated in french :	Le tatouage est proche de son ventre.
English text :	Wow uncle, what's that tattoo?! You must've had a wild youth!	localizated in turkish :	Vay Dayıma bak ne dövmeler yaptırmış! Çılgın bir gençlik geçirmişsin galiba!	localized in german :	Wow Onkel, was ist mit diesem Tattoo?! Du musst eine wilde Jugend gehabt haben!	localizated in french :	Tonton, c’est quoi ce tatouage ? Tu as eu une sacrée jeunesse !
English text :	Hey! I'm still young and I'm still wild!	localizated in turkish :	Ne münasebet! Ben hala genç ve çılgınım!	localized in german :	Hey! Ich bin immer noch jung, und immer noch wild!	localizated in french :	Hé ! Je suis toujours jeune !
English text :	Doctor Worry needs some gas for the new formula.	localizated in turkish :	Doktor Civan'ın hazırladığı formül için gaza ihtiyacı var.	localized in german :	Dr. Kummer braucht Gas für seine neue Rezeptur.	localizated in french :	Le Docteur Névrose a besoin de gaz pour une expérience.
English text :	Try to get some 'natural' gas.	localizated in turkish :	Bir yerden 'doğal' gaz bulabilirsin.	localized in german :	Versuche, 'natürliches' Gas zu besorgen.	localizated in french :	Essayez de produire du gaz « naturel ».
English text :	There is a bean can hidden behind the canisters.	localizated in turkish :	Varillerin ardında saklı bir kurufasulye konservesi var.	localized in german :	Hinter dem Kanister ist eine Dose Bohnen.	localizated in french :	Une boîte de haricots est cachée derrière les bidons.
English text :	Drag the bean can at Uncle Bubba's mouth.	localizated in turkish :	Konserveyi Dayının ağzına sürükle.	localized in german :	Ziehe die Dose zu Onkel Bubbas Mund.	localizated in french :	Amenez la boîte de haricots sur la bouche de Tonton Tanguy.
English text :	Well, thanks for the gas.	localizated in turkish :	Sağolasın, gazına bereket!	localized in german :	Nun, danke für das Gas.	localizated in french :	Merci pour le gaz !
English text :	It wasn't me!	localizated in turkish :	Ben yapmadım! Miki yaptı!	localized in german :	Ich wars nicht!	localizated in french :	C’est pas moi !
English text :	Which golden coin is real?	localizated in turkish :	Hangisi gerçek altın para acaba?	localized in german :	Welches Goldstück ist echt?	localizated in french :	Quelle pièce d’or est la vraie ?
English text :	Doctor Worry can test them with his mouth.	localizated in turkish :	Doktor Civan dişleriyle test edebilir.	localized in german :	Dr. Kummer kann sie mit seinem Mund austesten.	localizated in french :	Le Docteur Névrose peut les croquer pour les tester.
English text :	Drag them at Worry's mouth one by one.	localizated in turkish :	Gerçeğini bulana kadar teker teker alıp Civan'ın ağzına sürükle.	localized in german :	Ziehe sie zu Dr. Kummers Mund, eins nach dem anderen.	localizated in french :	Amenez-les une par une dans sa bouche.
English text :	Yes! I'm rich now!	localizated in turkish :	Bu hakiki altın! Zenginiz!	localized in german :	Ja! Ich bin jetzt reich!	localizated in french :	Oui ! Je suis riche !
English text :	Zorry, Wily. Kould we vuse it wor my tental dreatment?	localizated in turkish :	Övce bevim kıvılan difimi hayletsek vu pavayla olvaz mı?	localized in german :	Entfuldige Lilli, könnten fir ef für einen Fahnarft ferfenden?	localizated in french :	Euh, on beut udiliser l’argent pour aller jez le dendisse ?
English text :	Doctor Worry needs to get blood samples from that fly.	localizated in turkish :	Doktor Civan'ın sinekten kan örneği alması gerekli.	localized in german :	Dr. Kummer braucht eine Blutprobe von dieser Mücke.	localizated in french :	Le Docteur Névrose a besoin d’un échantillon de sang du moustique.
English text :	Your finger might attract the fly.	localizated in turkish :	Parmağın sineği çekebilir.	localized in german :	Dein Finger zieht vielleicht die Mücke an.	localizated in french :	Elle est attirée par votre doigt.
English text :	Keep your finger on the screen until the fly is filled with blood.	localizated in turkish :	Sineğin karnı kanla dolana kadar parmağını telefona basılı tut.	localized in german :	Halte deinen Finger auf dem Bildschirm, bis sich die Mücke vollgesogen hat.	localizated in french :	Laissez votre doigt sur l’écran jusqu’à ce que la mouche soit pleine de sang.
English text :	Use the swatter to kill it.	localizated in turkish :	Karnı dolunca sinek raketiyle öldür.	localized in german :	Verwende die Klatsche, um sie zu töten.	localizated in french :	Écrasez-la avec la tapette.
English text :	Drag the blood into the bottle.	localizated in turkish :	Boş şişeyi kanına sürükle.	localized in german :	Ziehe das Blut in die Flasche.	localizated in french :	Mettez le sang dans le flacon.
English text :	This level is just an excuse to get back at those pesky mosquitos!	localizated in turkish :	Bu bölümü sivrisineklerden tüm yazı mahvolan insanlara ithaf ediyoruz.	localized in german :	Dieses Level ist bloß da, um unseren Frust über diese lästigen Moskitos auszudrücken!	localizated in french :	Bon d’accord, c’est votre sang.
English text :	Doctor Worry's new robot requires some electricity.	localizated in turkish :	Doktor Civan'ın yeni robotuna elektrik lazım.	localized in german :	Dr. Kummers neuer Roboter braucht Strom.	localizated in french :	Le nouveau robot du Docteur Névrose n’est pas alimenté.
English text :	Try to get electricity from that cord.	localizated in turkish :	O kablodan elektrik almaya çalış.	localized in german :	Hol dir Strom vom Kabel.	localizated in french :	Le câble est trop court.
English text :	Make them hold hands to convey the electricity.	localizated in turkish :	Onları el ele tutuşturup elektriği robota ilet.	localized in german :	Lasse sie Hände halten, um Strom zu übertragen.	localizated in french :	Il faut qu’ils se tiennent la main pour conduire le courant.
English text :	Put Worry next to the robot, and put Lily right next to them and make her hand touch the plug.	localizated in turkish :	Civan'ı robotun yanına ve onun hemen yanına da bediş'yı koy. bediş'nın eli kabloya değsin.	localized in german :	Gib Kummer neben den Roboter, Lilli gleich daneben, und lasse ihre Hand das Kabel berühren.	localizated in french :	Placez Névrose près du robot, et Lily à côté de Névrose. Ensuite, elle doit toucher la prise.
English text :	Don't try this at home, kids. Well, you adults either actually!	localizated in turkish :	Bunu sakın evde denemeyin, elektriğin şakası olmaz! Olsa da komik olmaz!	localized in german :	Versucht das nicht zu Hause, Kinder. Und ihr Erwachsenen am Besten auch nicht!	localizated in french :	N’essayez pas de faire ça chez vous !
English text :	We need a cleaner energy for this robot.	localizated in turkish :	Robot için daha temiz bir enerji kaynağına ihtiyaç var.	localized in german :	Wir brauchen saubere Energie für diesen Roboter.	localizated in french :	Ce robot fonctionne à l’énergie propre.
English text :	Find a way to make that battery cleaner.	localizated in turkish :	O pili daha temiz hale getirmeyi dene.	localized in german :	Mache die Batterie irgendwie sauberer.	localizated in french :	Comment rendre la batterie propre ?
English text :	Drag the battery at the sink and wash it.	localizated in turkish :	Pili lavaboya sürükleyip yıka. Suyu açmayı unutma.	localized in german :	Ziehe die Batterie zum Waschbecken und wasche sie.	localizated in french :	Mettez la batterie dans l’évier et nettoyez-la.
English text :	Drag the washed battery back to the robot.	localizated in turkish :	Yıkanmış pili robota sürükle.	localized in german :	Ziehe die gewaschene Batterie zurück zum Roboter.	localizated in french :	Remettez la batterie propre dans le robot.
English text :	That's not how clean energy works. But anything goes with this crazy game, I guess.	localizated in turkish :	Yaptığımız bu kelime esprisinden ötürü özür dileriz dersek yalan olur. Bu oyunda daha neler neler göreceksin!	localized in german :	So funktioniert saubere Energie nicht. Aber in diesem irren Spiel geht anscheinend alles.	localizated in french :	Ce n’est pas vraiment ça, l’énergie propre. Plus rien ne m’étonne, dans ce jeu.
English text :	I must find a way to discharge.	localizated in turkish :	Üstümdeki cereyanı boşaltmalıyım.	localized in german :	Ich muss irgendwie entladen.	localizated in french :	Je dois me débarrasser de cette électricité statique.
English text :	There is dirt on the right.	localizated in turkish :	Sağ tarafta toprak var.	localized in german :	Rechts ist Erde.	localizated in french :	Il y a de la terre sur la droite.
English text :	Drag the screen to see the dirt.	localizated in turkish :	Ekranı kaydırıp toprağı gör.	localized in german :	Ziehe den Bildschirm, um die Erde zu sehen.	localizated in french :	Faites glisser l’écran pour voir la terre.
English text :	Put away her socks and drag her on to the dirt.	localizated in turkish :	Çoraplarını çıkar ve toprağa sürükle.	localized in german :	Mache sie barfuß und ziehe sie zur Erde.	localizated in french :	Retirez ses chaussures et chaussettes, et mettez-la sur la terre.
English text :	Easy for Lily! Where can you find some dirt to get onto these days?!	localizated in turkish :	Bediş'e kolay tabi! Biz bu devirde böyle toprağı nerede bulalım?! Her yer beton!	localized in german :	Wo kann man sonst heutzutage noch Erde finden? Alles nur noch Beton!	localizated in french :	Pauvre Lily ! C’est de plus en plus dur de se reconnecter à la nature !

    Format your response exactly as a valid JSON object with this structure:
    {{
        "localization": {{
            "turkish": "Turkish localization...",
            "french": "French localization...",
            "german": "German localization..."
        }}
    }}
    """
    
    try:
        # Create request using OpenAI client for OpenRouter
        print(f"  Sending request to model: {model_name}")
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://cascade.ai",  # Site URL for rankings
                "X-Title": "Game Localization Tool",   # Site title for rankings
            },
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"English text: {english_text}"}
            ],
            timeout=60  # Add timeout to prevent hanging requests
        )
        
        # Extract response text
        response_text = completion.choices[0].message.content
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            
            # Validate that the result contains the expected keys
            if 'localization' not in result:
                print(f"⚠️ Response JSON doesn't contain 'localization' key: {result}")
                return {
                    "localization": {
                        "turkish": f"[ERROR: Missing localization data] {english_text}",
                        "french": f"[ERROR: Missing localization data] {english_text}",
                        "german": f"[ERROR: Missing localization data] {english_text}"
                    }
                }
            
            # Check if all languages are included
            missing_langs = []
            for lang in ['turkish', 'french', 'german']:
                if lang not in result['localization']:
                    missing_langs.append(lang)
                    result['localization'][lang] = f"[ERROR: Missing {lang} translation] {english_text}"
            
            if missing_langs:
                print(f"⚠️ Response missing translations for: {', '.join(missing_langs)}")
                
            print(f"✓ Successfully processed localization with model {model_id}")
            return result
            
        except json.JSONDecodeError as json_err:
            print(f"✗ Error parsing JSON from model {model_id}: {str(json_err)}")
            print(f"Response text: {response_text[:200]}...")
            return {
                "localization": {
                    "turkish": f"[ERROR: Invalid JSON] {english_text}",
                    "french": f"[ERROR: Invalid JSON] {english_text}",
                    "german": f"[ERROR: Invalid JSON] {english_text}"
                }
            }
        
    except Exception as e:
        print(f"✗ Error processing localization with model {model_id}: {str(e)}")
        return {
            "localization": {
                "turkish": f"[ERROR: {str(e)}] {english_text}",
                "french": f"[ERROR: {str(e)}] {english_text}",
                "german": f"[ERROR: {str(e)}] {english_text}"
            }
        }

def read_semicolon_csv(csv_file):
    """Read a CSV file with semicolons as separators"""
    rows = []
    headers = ["KEY", "LEVEL_ID", "Text_ID", "image_id", "en", "tr", "de", "fr"]
    
    with open(csv_file, 'r', encoding='utf-8') as file:
        # Skip BOM if present
        content = file.read()
        if content.startswith('\ufeff'):
            content = content[1:]
            
    lines = content.split('\n')
    
    # Skip header line
    for line in lines[1:]:
        if not line.strip():  # Skip empty lines
            continue
            
        # Split by semicolon
        values = line.split(';')
        
        if len(values) >= len(headers):
            row = {}
            for i, header in enumerate(headers):
                row[header] = values[i] if i < len(values) else ""
            rows.append(row)
    
    return headers, rows

def process_row(row, imgs_dir, descriptions_cache, model_id, model_name, debug=False):
    """Process a single row from the CSV file"""
    # Extract the necessary information
    image_id = row.get('image_id', '').strip() if row.get('image_id') else ''
    english_text = row.get('en', '').strip() if row.get('en') else ''
    key = row.get('KEY', '').strip() if row.get('KEY') else ''
    level_id = row.get('LEVEL_ID', '').strip() if row.get('LEVEL_ID') else ''
    text_id = row.get('Text_ID', '').strip() if row.get('Text_ID') else ''
    
    # Skip rows without image_id or English text
    if not image_id or not english_text:
        print(f"\n⚠️ Skipping row with KEY={key}: Missing image_id or English text")
        return row, None
    
    print(f"\n🔄 Processing {key} for Level {level_id}, Text {text_id}, Image {image_id}")
    
    # Get or generate image description
    if image_id in descriptions_cache:
        description = descriptions_cache[image_id]
    else:
        # Find the image file
        image_path = get_image_path(imgs_dir, image_id)
        if not image_path:
            print(f"⚠️ Image not found for image_id={image_id}")
            return row, None
        
        # Get the description
        description = get_image_description(image_path, debug=debug)
        descriptions_cache[image_id] = description
    
    # Process localization
    result = process_localization(description, english_text, model_id, model_name, debug=debug)
    
    # Update the row with localization results
    localizations = result.get('localization', {})
    for lang_name, lang_code in LANGUAGE_CODES.items():
        if lang_name in localizations:
            row[lang_code] = localizations[lang_name]
    
    return row, description

def write_semicolon_csv(rows, headers, output_file):
    """Write rows to a CSV file with semicolons as separators using UTF-8 encoding.
    Special focus on ensuring Turkish characters are properly handled.
    """
    # Explicitly use utf-8-sig to include BOM (Byte Order Mark) for better compatibility
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as file:
        # Write headers with BOM to ensure encoding is recognized
        header_line = ';'.join(headers) + '\n'
        file.write(header_line)
        
        # For each row, ensure proper encoding especially for Turkish characters
        for row in rows:
            # Turkish characters that need special attention: ı, İ, ğ, Ğ, ü, Ü, ş, Ş, ç, Ç, ö, Ö
            values = []
            for header in headers:
                value = row.get(header, '')
                if value is None:
                    value = ''
                # Ensure proper string representation with Turkish characters
                if isinstance(value, str):
                    # Already a string, just use as is (will be encoded as UTF-8-sig)
                    values.append(value)
                else:
                    # Convert to string
                    values.append(str(value))
            
            # Join with semicolons and write to file
            line = ';'.join(values) + '\n'
            file.write(line)
    
    print(f"✓ Saved CSV results to {output_file} (UTF-8 encoded with BOM for Turkish character support)")

def process_csv_file(data_file, imgs_dir, output_dir, debug=False, limit=None):
    """Process CSV data file and create output for each model"""
    print(f"\n🚀 Starting CSV-based localization process with OpenAI client")
    
    # Create a dictionary to cache image descriptions
    descriptions_cache = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    try:
        headers, rows = read_semicolon_csv(data_file)
        print(f"Loaded {len(rows)} rows from {data_file}")
        
        if debug:
            print(f"\n🔍 DEBUGGING: First few rows")
            for i, row in enumerate(rows[:3]):
                print(f"\nRow {i+1}:")
                for key, value in row.items():
                    print(f"  {key}: '{value}'")
                
    except Exception as e:
        print(f"✗ Error reading CSV file: {str(e)}")
        return
    
    # Check imgs directory
    if debug:
        print(f"\n🔍 DEBUGGING: Checking imgs directory: {imgs_dir}")
        if os.path.exists(imgs_dir):
            print(f"Directory exists. Contains {len(os.listdir(imgs_dir))} files.")
            print(f"Sample files: {', '.join(os.listdir(imgs_dir)[:5])}")
        else:
            print(f"⚠️ Directory does not exist!")
    
    # Limit rows if specified
    if limit and limit > 0:
        rows = rows[:limit]
        print(f"Processing only the first {limit} rows")
    
    # Process with each model
    for model_id, model_name in MODELS.items():
        print(f"\n🌐 Processing with model: {model_id}")
        
        # Create a copy of the rows for this model
        model_rows = [row.copy() for row in rows]
        
        # Prepare output data structures
        complete_data = []
        
        # Process each row
        for i, row in enumerate(model_rows):
            try:
                updated_row, description = process_row(row, imgs_dir, descriptions_cache, model_id, model_name, debug=debug)
                
                if updated_row and description:
                    # Update the row with the processed data
                    model_rows[i] = updated_row
                    
                    # Don't add description to CSV output, only for JSON
                    
                    # Get the image ID for the JSON record
                    image_id = str(row.get('image_id', ''))
                    
                    # Create a complete data record for JSON output
                    json_record = {
                        'KEY': updated_row.get('KEY', ''),
                        'LEVEL_ID': updated_row.get('LEVEL_ID', ''),
                        'image_id': image_id,
                        'en': updated_row.get('en', ''),
                        'description': description,
                        'localization': {
                            'tr': updated_row.get('tr', ''),
                            'fr': updated_row.get('fr', ''),
                            'de': updated_row.get('de', '')
                        }
                    }
                    complete_data.append(json_record)
                
                # Add a small delay between API calls to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"✗ Error processing row {i}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Generate the model file name prefix
        model_name_safe = model_id.replace('/', '_').replace(' ', '_').replace('-', '_').replace('.', '_')
        
        # Don't add description to CSV headers
        
        # Save the resulting CSV with UTF-8 encoding
        output_csv_path = os.path.join(output_dir, f"output_{model_name_safe}.csv")
        write_semicolon_csv(model_rows, headers, output_csv_path)
        
        # Save the complete data JSON including all fields
        json_path = os.path.join(output_dir, f"output_{model_name_safe}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(complete_data, f, ensure_ascii=False, indent=4)
        print(f"✓ Saved JSON output to {json_path}")
    
    print("\n✅ Processing completed successfully")

def main():
    parser = argparse.ArgumentParser(description='OpenAI Client Processor for Game Localization')
    parser.add_argument('--data', required=True, help='Path to the data.csv file (semicolon-separated)')
    parser.add_argument('--imgs', required=True, help='Directory containing game screenshots')
    parser.add_argument('--output', required=True, help='Directory to save output CSV files')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (no API calls, mock responses)')
    parser.add_argument('--limit', type=int, help='Limit number of rows to process')
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isfile(args.data):
        print(f"Error: Data file {args.data} does not exist")
        return
    
    if not os.path.isdir(args.imgs):
        print(f"Error: Images directory {args.imgs} does not exist")
        return
    
    # Process the CSV file
    process_csv_file(args.data, args.imgs, args.output, debug=args.debug, limit=args.limit)

if __name__ == "__main__":
    main()
