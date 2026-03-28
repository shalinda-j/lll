"""
Self-Study Training Loop for Zero-Base LLM.

Implements training with:
- 500+ diverse English sentences
- 80/20 train/validation split
- Overfitting detection (val_loss > 1.5×train_loss)
- Dropout regularization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Dict, Any, List
import random
from pathlib import Path

from ..config import ZeroBaseConfig
from ..model.model import ZeroBaseLLM


# 500+ diverse English sentences for training
DIVERSE_SEED_TEXTS = [
    # ==================== GREETINGS AND CONVERSATIONS (80 sentences) ====================
    "hello world how are you doing today",
    "good morning everyone have a wonderful day",
    "good afternoon to you all my friends",
    "good evening and welcome to the show",
    "good night and sweet dreams to you",
    "how are you doing on this fine day",
    "i am doing very well thank you for asking",
    "nice to meet you my name is john",
    "pleased to make your acquaintance today",
    "thank you very much for your kind help",
    "you are most welcome my dear friend",
    "please excuse me for being late today",
    "no problem at all i understand completely",
    "i really appreciate your help with this",
    "it was my pleasure to assist you today",
    "hey there what is going on with you",
    "hi how have you been lately my friend",
    "hey long time no see how are things",
    "good to see you again after so long",
    "i missed you so much while you were gone",
    "welcome back home we missed you here",
    "goodbye for now see you later today",
    "take care of yourself and stay safe",
    "have a great day ahead of you",
    "see you tomorrow at the usual time",
    "catch you later alligator as they say",
    "bye bye have a safe trip home",
    "farewell until we meet again soon",
    "it was nice talking with you today",
    "i hope to see you again very soon",
    "let us keep in touch from now on",
    "do not forget to write me sometime",
    "call me when you get back home",
    "text me later if you have time",
    "i will be waiting for your reply",
    "looking forward to hearing from you",
    "what is new with you these days",
    "anything interesting happening lately",
    "how is your family doing these days",
    "i heard you got a new job recently",
    "congratulations on your new position",
    "that is great news i am so happy",
    "i am sorry to hear about that situation",
    "do not worry everything will be fine",
    "is there anything i can do to help",
    "just let me know if you need anything",
    "i am always here for you my friend",
    "you can count on me anytime you need",
    "that sounds like a good plan to me",
    "what do you think about this idea",
    "i agree with what you are saying",
    "i see your point but i disagree slightly",
    "that is a very interesting perspective",
    "i had not thought about it that way",
    "you make a very good point there",
    "i understand what you are trying to say",
    "could you explain that a bit more please",
    "what exactly do you mean by that statement",
    "i am not sure i follow what you mean",
    "let me think about it for a moment",
    "give me a second to process this information",
    "i need some time to consider this carefully",
    "that is something worth thinking about",
    "i will get back to you on this matter",
    "let us discuss this further later on",
    "we should talk about this more in depth",
    "i would love to continue this conversation",
    "this has been a very productive discussion",
    "thank you for sharing your thoughts with me",
    "i appreciate your honest opinion about this",
    "your feedback is very valuable to me",
    "i value your input on this matter greatly",
    "that gives me a lot to think about",
    "you have given me a new perspective here",
    "i learned something new from our chat",
    "our conversation was very enlightening for me",
    "i enjoyed talking with you about this topic",

    # ==================== QUESTIONS AND ANSWERS (80 sentences) ====================
    "what is your name and where are you from",
    "my name is mary and i am from london",
    "where did you go yesterday afternoon",
    "i went to the park to take a walk",
    "when will you be coming back home",
    "i will be back sometime next week",
    "why did you choose this particular path",
    "because it seemed like the best option",
    "how do you solve this difficult problem",
    "let me think about it for a moment",
    "which book would you recommend reading",
    "i would suggest reading the great novel",
    "who is your favorite author and why",
    "my favorite writer writes mystery books",
    "what time does the movie start tonight",
    "the movie begins at seven in the evening",
    "where is the nearest grocery store located",
    "the store is just around the corner",
    "how much does this item cost here",
    "that will be twenty dollars total please",
    "can you help me with my homework please",
    "of course i would be happy to assist you",
    "do you know the answer to this question",
    "i think the correct answer is option b",
    "would you like something to drink now",
    "yes please i would love some water",
    "could you please pass me the salt shaker",
    "here you go i hope you enjoy your meal",
    "may i ask you a personal question now",
    "sure go ahead i am open to discussing it",
    "should i take the bus or the train today",
    "the train would be faster in this traffic",
    "what is the weather going to be like tomorrow",
    "it should be sunny and warm all day long",
    "how long does it take to get there by car",
    "it usually takes about thirty minutes driving",
    "where can i find a good restaurant nearby",
    "there are several great options downtown",
    "what is the best way to learn a new language",
    "practice every day and speak with native speakers",
    "when is the best time to visit your city",
    "spring and autumn are the most beautiful seasons",
    "why do people enjoy traveling so much these days",
    "it broadens their horizons and creates memories",
    "how often do you exercise during the week",
    "i try to work out at least three times weekly",
    "what are your plans for the upcoming weekend",
    "i plan to relax and spend time with family",
    "which sports do you enjoy watching on television",
    "i love watching football and basketball games",
    "who inspired you to pursue your current career",
    "my high school teacher motivated me greatly",
    "what is your biggest achievement in life so far",
    "graduating from university was a major milestone",
    "how do you handle stress in your daily life",
    "i practice meditation and go for regular walks",
    "what makes you feel truly happy in life",
    "spending quality time with loved ones brings joy",
    "where would you like to travel in the future",
    "i dream of visiting japan and exploring europe",
    "what is your favorite memory from childhood",
    "playing in the garden with my siblings was wonderful",
    "how has your life changed over the past decade",
    "i have grown more confident and independent",
    "what advice would you give to younger generations",
    "always follow your dreams and never give up",
    "which skills do you think are most important today",
    "communication and problem solving are essential skills",
    "what do you value most in your friendships",
    "honesty and loyalty mean everything to me",
    "how do you stay motivated during difficult times",
    "i remind myself of my goals and stay positive",
    "what would you do if you won the lottery tomorrow",
    "i would travel the world and help my family",
    "which historical figure do you admire the most",
    "i admire leaders who fought for human rights",
    "what is your favorite season of the year",
    "autumn with its beautiful colors is my favorite",

    # ==================== DESCRIPTIONS - OBJECTS, ANIMALS, PLACES (100 sentences) ====================
    "the cat is a small furry animal that meows",
    "dogs are loyal pets that love to play fetch",
    "birds can fly high in the sky above the trees",
    "fish swim gracefully through the clear blue water",
    "horses are magnificent creatures used for riding",
    "elephants are the largest land animals on earth",
    "lions are known as the kings of the jungle",
    "penguins live in cold places and cannot fly",
    "dolphins are intelligent mammals that live in oceans",
    "butterflies have beautiful colorful wings to fly",
    "the old wooden table has scratches from years of use",
    "a comfortable chair makes reading more enjoyable",
    "the computer screen displays bright colorful images",
    "books contain knowledge and stories to explore",
    "a clock on the wall shows the current time",
    "the refrigerator keeps our food fresh and cold",
    "a warm blanket provides comfort on cold nights",
    "the lamp illuminates the dark corner of the room",
    "a mirror reflects our image back to ourselves",
    "the window allows sunlight to enter the room",
    "the kitchen is where we prepare our daily meals",
    "a bedroom is a place for rest and relaxation",
    "the bathroom has a shower and a sink for washing",
    "the living room is where families gather together",
    "a garden provides fresh vegetables and beautiful flowers",
    "the garage is used for parking cars and storage",
    "the attic stores old items we rarely use anymore",
    "a basement can be used for many different purposes",
    "the front door welcomes visitors into our home",
    "a fireplace keeps the house warm during winter months",
    "mountains reach high into the clouds above us",
    "rivers flow steadily towards the open sea",
    "forests are filled with trees and wildlife everywhere",
    "deserts are dry lands with very little rainfall yearly",
    "oceans cover most of the earths surface area",
    "islands are small pieces of land surrounded by water",
    "valleys lie between mountains with green grass fields",
    "waterfalls cascade down from high rocky cliffs",
    "lakes provide peaceful settings for relaxation and fishing",
    "beaches have soft sand and crashing ocean waves",
    "the sun provides light and warmth to our planet",
    "the moon illuminates the night sky with silver light",
    "stars twinkle like diamonds in the dark night sky",
    "clouds float across the sky in various shapes",
    "rain brings necessary water to plants and crops",
    "snow covers the ground in a white blanket during winter",
    "wind blows leaves across the ground in autumn",
    "thunder and lightning accompany heavy storms sometimes",
    "the rainbow appears after rain with beautiful colors",
    "sunsets paint the sky in orange and pink hues",
    "sunrises mark the beginning of a new fresh day",
    "a car has four wheels and takes us places quickly",
    "an airplane can fly across the world in hours",
    "a train travels on tracks between different cities",
    "bicycles are healthy ways to travel short distances",
    "ships carry goods and people across the vast oceans",
    "a motorcycle is a fast two wheeled vehicle",
    "buses transport many passengers along fixed routes daily",
    "a taxi takes you exactly where you want to go",
    "subway trains run underground in big cities efficiently",
    "trucks deliver goods to stores and homes everywhere",
    "ambulances rush sick people to hospitals quickly",
    "fire trucks help put out dangerous fires safely",
    "police cars help keep our communities safe and secure",
    "a library is a quiet place to read and study",
    "schools are where children learn and grow together",
    "hospitals help sick people get better and recover",
    "restaurants serve delicious food to hungry customers",
    "museums display art and artifacts for public viewing",
    "parks offer green spaces for recreation and relaxation",
    "movie theaters show films on big screens for entertainment",
    "stadiums host sporting events and concerts for thousands",
    "airports are where planes take off and land daily",
    "train stations connect cities through railway networks",
    "shopping malls have many stores under one roof",
    "banks keep our money safe and provide financial services",
    "post offices handle mail and packages for delivery",
    "churches are places of worship for religious communities",
    "universities offer higher education and research opportunities",
    "factories manufacture products we use every single day",
    "farms grow crops and raise animals for food production",

    # ==================== NATURE AND WEATHER (60 sentences) ====================
    "the sun is shining brightly in the sky today",
    "dark clouds are gathering overhead right now",
    "it looks like it might rain very soon here",
    "the flowers are blooming in the garden this spring",
    "trees lose their leaves in the autumn season",
    "birds sing beautiful songs at dawn every morning",
    "the ocean waves crash on the sandy shore continuously",
    "mountains are covered with white snow in winter",
    "the river flows gently through the green valley",
    "stars twinkle in the night sky above us all",
    "spring brings new life to the world around us",
    "summer days are long and filled with warm sunshine",
    "autumn leaves turn beautiful shades of red and orange",
    "winter brings cold weather and sometimes heavy snowfall",
    "every new year brings fresh opportunities and possibilities",
    "the seasons change as the earth orbits the sun",
    "nature provides us with food water and fresh air",
    "plants need sunlight and water to grow healthy and strong",
    "animals adapt to their environments in remarkable ways",
    "the ecosystem is a delicate balance of living things",
    "protecting nature is important for future generations to enjoy",
    "pollution harms the environment and the creatures living in it",
    "recycling helps reduce waste and protect our planet",
    "clean water is essential for all life on earth",
    "forests produce oxygen and provide homes for wildlife",
    "coral reefs are colorful underwater ecosystems in oceans",
    "rainforests are home to many unique plant and animal species",
    "the arctic is a frozen region at the north pole",
    "deserts receive very little rain throughout the entire year",
    "wetlands filter water and provide important wildlife habitats",
    "sunny weather is perfect for outdoor activities and fun",
    "rainy days are good for staying inside and reading books",
    "windy weather makes the trees sway back and forth",
    "foggy conditions reduce visibility on the roads significantly",
    "hot weather makes us want to swim and stay cool",
    "cold weather requires warm clothes and hot drinks",
    "humid weather feels sticky and uncomfortable to many people",
    "dry weather can lead to drought in some regions",
    "stormy weather brings thunder lightning and heavy rain",
    "pleasant weather makes everyone feel happy and relaxed",
    "changing weather patterns affect our daily plans often",
    "the sunrise paints the eastern sky with golden colors",
    "the sunset creates a beautiful display of colors",
    "the moon cycles through phases each month regularly",
    "meteor showers light up the night sky occasionally",
    "the northern lights are a stunning natural phenomenon",
    "earthquakes shake the ground and can cause damage",
    "volcanoes erupt with hot lava and volcanic ash",
    "tornadoes are powerful spinning columns of destructive wind",
    "hurricanes are large storms that form over warm oceans",
    "tsunamis are giant waves caused by underwater earthquakes",
    "avalanches are sudden falls of snow down mountainsides",
    "floods happen when too much rain falls too quickly",
    "droughts occur when there is not enough rainfall",
    "wildfires can spread rapidly through dry forests",
    "blizzards are severe snowstorms with strong winds",
    "heat waves bring dangerously high temperatures to regions",
    "cold snaps bring freezing temperatures unexpectedly to areas",
    "the weather affects our mood and daily activities",
    "meteorologists predict the weather using scientific methods",

    # ==================== FOOD AND DRINKS (50 sentences) ====================
    "i love eating pizza with extra cheese on top",
    "fresh vegetables are very healthy for our bodies",
    "cooking is one of my favorite hobbies to do",
    "the restaurant serves delicious food at reasonable prices",
    "would you like some tea or coffee with your meal",
    "breakfast is the most important meal of every day",
    "i prefer home cooked meals over fast food always",
    "chocolate cake is my absolute favorite dessert ever",
    "drinking enough water is very important for good health",
    "the soup was hot and very tasty indeed today",
    "fruits contain natural sugars and important vitamins",
    "rice is a staple food in many countries worldwide",
    "bread can be used for sandwiches and toast",
    "pasta comes in many different shapes and sizes",
    "meat provides protein for building strong muscles",
    "fish is a healthy source of omega fatty acids",
    "eggs can be cooked in many different ways",
    "cheese adds flavor to many dishes and recipes",
    "salads are light and refreshing meals for lunch",
    "sandwiches are convenient meals for busy people",
    "ice cream is a delicious frozen treat in summer",
    "cookies are perfect with a glass of cold milk",
    "pie is a classic dessert for special occasions",
    "cake is served at birthdays and celebrations everywhere",
    "pancakes are fluffy and perfect for breakfast time",
    "waffles have crispy edges and soft centers inside",
    "cereal is a quick and easy breakfast option daily",
    "oatmeal is a warm and filling morning meal choice",
    "yogurt contains beneficial bacteria for digestive health",
    "milk is rich in calcium for strong bones and teeth",
    "juice provides vitamins from fresh fruits and vegetables",
    "smoothies blend fruits into refreshing healthy drinks",
    "tea has antioxidants and can be served hot or cold",
    "coffee gives energy and helps people wake up mornings",
    "soda is a sweet carbonated beverage enjoyed by many",
    "wine is made from fermented grapes in vineyards",
    "beer is a popular alcoholic drink in many cultures",
    "soup warms you up on cold winter days nicely",
    "stew is a hearty meal with meat and vegetables",
    "grilled food has a delicious smoky flavor to it",
    "fried food is crispy but should be eaten in moderation",
    "baked goods fill the kitchen with wonderful aromas",
    "steamed food retains more nutrients during cooking process",
    "roasted vegetables have a rich and caramelized flavor",
    "boiled eggs are simple to make and nutritious too",
    "scrambled eggs are soft and fluffy when cooked right",
    "grilled cheese sandwiches are comfort food for many people",
    "spaghetti with tomato sauce is a classic italian dish",
    "hamburgers are popular fast food around the world",
    "hot dogs are often served at baseball games",

    # ==================== ACTIONS AND DESCRIPTIONS (60 sentences) ====================
    "the cat jumped over the high wooden fence gracefully",
    "she carefully opened the mysterious old box slowly",
    "the car drove slowly down the narrow winding street",
    "he ran quickly to catch the departing bus on time",
    "the baby laughed happily at the funny colorful toy",
    "they walked together through the green forest quietly",
    "the wind blew strongly across the open grassy field",
    "she beautifully played the classical piano piece perfectly",
    "the children played happily in the safe park together",
    "the dog barked loudly at the passing stranger warningly",
    "the bird sang sweetly from the tall tree branch",
    "he swam swiftly across the clear blue swimming pool",
    "the dancers moved gracefully across the polished wooden stage",
    "she painted a beautiful landscape with vibrant colors",
    "the students listened attentively to the wise teacher",
    "he typed quickly on his laptop computer keyboard",
    "the chef cooked a delicious meal in the kitchen",
    "she planted colorful flowers in her small garden",
    "the athlete trained hard for the upcoming competition",
    "he fixed the broken bicycle with new parts",
    "she carefully wrapped the gift in pretty paper",
    "the children built a tall sandcastle on the beach",
    "he climbed the steep mountain with great determination",
    "she wrote a letter to her faraway friend",
    "the scientist conducted experiments in the modern laboratory",
    "he played the guitar and sang a sweet melody",
    "she taught her students how to solve problems",
    "the doctor examined the patient with great care",
    "he repaired the old car engine skillfully",
    "she designed a beautiful dress for the party",
    "the actor performed brilliantly on the theater stage",
    "he programmed the computer to solve the puzzle",
    "she carefully arranged the flowers in a vase",
    "the team worked together to complete the project",
    "he translated the document from english to spanish",
    "she edited the video for her youtube channel",
    "the photographer captured the beautiful sunset perfectly",
    "he carefully built a model airplane from parts",
    "she prepared a presentation for the important meeting",
    "the musicians rehearsed for their upcoming concert",
    "he cleaned the entire house from top to bottom",
    "she organized her bookshelf by author and genre",
    "the artist sketched a portrait of his subject",
    "he carefully repaired the torn old book pages",
    "she baked fresh bread that smelled absolutely wonderful",
    "the waiter served the guests with friendly smiles",
    "he polished his shoes until they shone brightly",
    "she knitted a warm scarf for the cold winter",
    "the mechanic diagnosed the problem with the engine",
    "he carefully assembled the furniture following instructions",
    "she ironed all the wrinkled clothes neatly",
    "the gardener watered the plants in the morning",
    "he sharpened the pencils for the drawing class",
    "she carefully folded the clean laundry items",
    "the janitor cleaned the floors until they sparkled",
    "he tuned the piano to perfect musical pitch",
    "she arranged the chairs for the big meeting",
    "the baker kneaded the dough for fresh bread",
    "he carefully polished the silver antique items",
    "she sorted the mail into different categories efficiently",

    # ==================== DAILY ACTIVITIES AND ROUTINE (50 sentences) ====================
    "i wake up early every single morning at dawn",
    "the sun rises in the east each new day",
    "i usually have coffee and toast for my breakfast",
    "after breakfast i go to work by the city bus",
    "i work in an office building in the city center",
    "during lunch i eat at a small cafe nearby",
    "i finish work around five in the late evening",
    "after work i like to read interesting books",
    "in the evening i watch television shows with family",
    "before bed i always brush my teeth carefully",
    "my daily routine helps me stay organized and productive",
    "i start my day with a healthy breakfast meal",
    "exercise is part of my daily morning routine now",
    "i check my emails first thing every work morning",
    "meetings take up much of my afternoon work schedule",
    "i make a list of tasks to complete each day",
    "taking breaks helps me stay focused on my work",
    "i review my progress at the end of each day",
    "planning ahead makes my days more efficient overall",
    "i try to maintain a healthy work life balance",
    "weekends are for relaxing and spending time with family",
    "i enjoy sleeping in on my days off from work",
    "sunday is my favorite day of the entire week",
    "i do my grocery shopping on saturday mornings usually",
    "cleaning the house is part of my weekly routine",
    "i visit the gym three times every single week",
    "cooking dinner is something i enjoy doing each evening",
    "i help my children with their homework after school",
    "reading before bed helps me relax and sleep better",
    "i write in my journal before going to sleep",
    "meditation helps me clear my mind each morning",
    "i take my dog for a walk twice daily",
    "calling my parents is part of my weekly routine",
    "i try to learn something new every single day",
    "keeping a schedule helps me manage my time better",
    "i prepare my clothes the night before each work day",
    "setting goals helps me stay motivated throughout the year",
    "i review my accomplishments at the end of each month",
    "taking time for myself is important for my wellbeing",
    "i enjoy cooking special meals on weekend evenings",
    "spending time outdoors refreshes my mind and body",
    "i listen to music while doing household chores",
    "practicing gratitude improves my outlook on life daily",
    "i try to go to bed at the same time each night",
    "waking up at the same time helps my daily routine",
    "i drink water throughout the day to stay hydrated",
    "healthy habits contribute to a better lifestyle overall",
    "i avoid using my phone before going to sleep",
    "stretching in the morning helps wake up my body",
    "planning my meals helps me eat healthier each week",

    # ==================== COMMON IDIOMS AND PHRASES (60 sentences) ====================
    "a piece of cake means something very easy to do",
    "break a leg is said before a performance for good luck",
    "hit the sack means to go to bed and sleep",
    "under the weather means feeling sick or unwell",
    "spill the beans means to reveal a secret accidentally",
    "when pigs fly means something that will never happen",
    "costs an arm and a leg means very expensive to buy",
    "actions speak louder than words in most situations",
    "the ball is in your court means your turn to decide",
    "once in a blue moon means very rarely happening",
    "a dime a dozen means something very common and ordinary",
    "bite the bullet means to endure a painful situation",
    "call it a day means to stop working for the day",
    "cut to the chase means get to the main point directly",
    "easy come easy go as the old saying goes",
    "get out of hand means the situation became uncontrollable",
    "give someone the cold shoulder means to ignore them deliberately",
    "hang in there means do not give up keep trying",
    "it takes two to tango in most relationships",
    "let the cat out of the bag means reveal the secret",
    "miss the boat means lost the opportunity completely",
    "no pain no gain as athletes often say",
    "on the ball means being alert and prepared",
    "pull someone leg means to joke or tease playfully",
    "rain cats and dogs means raining very heavily",
    "see eye to eye means to agree completely with someone",
    "speak of the devil when someone appears unexpectedly",
    "time flies when you are having fun and enjoying",
    "when in rome do as the romans do as they say",
    "you can say that again means i completely agree",
    "better late than never as the proverb goes",
    "curiosity killed the cat but satisfaction brought it back",
    "the early bird catches the worm so wake up early",
    "every cloud has a silver lining in difficult situations",
    "fortune favors the bold and courageous people",
    "practice makes perfect in any skill you learn",
    "where there is smoke there is fire usually",
    "you cannot judge a book by its cover ever",
    "all that glitters is not gold in this world",
    "birds of a feather flock together often",
    "the pen is mightier than the sword in disputes",
    "when the going gets tough the tough get going",
    "look before you leap into any important decision",
    "two wrongs do not make a right in conflicts",
    "the grass is always greener on the other side",
    "rome was not built in a day so be patient",
    "do not count your chickens before they hatch",
    "an apple a day keeps the doctor away perhaps",
    "beauty is in the eye of the beholder truly",
    "ignorance is bliss in certain situations sometimes",
    "knowledge is power as the philosopher said",
    "patience is a virtue worth cultivating always",
    "time heals all wounds eventually they say",
    "laughter is the best medicine for many things",
    "home is where the heart is for most people",
    "actions speak louder than words in relationships",
    "honesty is the best policy in almost all cases",
    "practice what you preach to others always",

    # ==================== MORE Q&A PAIRS (50 sentences) ====================
    "what time is it right now asks the stranger",
    "it is currently three thirty in the afternoon",
    "where did you put my keys i cannot find them",
    "i think they are on the kitchen counter there",
    "how do you spell that difficult word correctly",
    "let me check the dictionary for the correct spelling",
    "why is the sky blue during the day time",
    "it appears blue because of how light scatters",
    "what is for dinner tonight at our house",
    "we are having spaghetti and meatballs for dinner",
    "can you pick up some milk on your way home",
    "sure i will stop at the store on my way",
    "did you remember to lock the front door tonight",
    "yes i double checked it before leaving the house",
    "how was your day at the office today dear",
    "it was quite busy but productive overall today",
    "what are your plans for this coming weekend",
    "i am thinking about going hiking on saturday",
    "have you seen my glasses anywhere around here",
    "check on the coffee table where you were sitting",
    "what should i wear to the party tonight",
    "something casual but nice would be appropriate",
    "do we need anything from the grocery store",
    "we could use some bread and eggs for breakfast",
    "are you feeling better now after resting some",
    "yes i feel much more refreshed after my nap",
    "when does the new movie start at the theater",
    "the first showing begins at seven in the evening",
    "who is coming to dinner tomorrow night with us",
    "my sister and her husband are joining us tomorrow",
    "what did you think of the book you just read",
    "i found it very engaging and well written overall",

    # ==================== FEELINGS AND EMOTIONS (40 sentences) ====================
    "i feel very happy about the good news today",
    "she was sad when she heard the tragic story",
    "he felt angry about the unfair decision made",
    "they were excited to go on their vacation soon",
    "i am worried about the upcoming exam results",
    "she felt nervous before her big job interview",
    "he was surprised by the unexpected gift received",
    "they felt proud of their team winning the game",
    "i am confused about what to do in this situation",
    "she felt relieved after solving the difficult problem",
    "he was disappointed with the poor test results",
    "they felt grateful for all the help they received",
    "i am tired after working all day at the office",
    "she felt lonely when her friends moved away",
    "he was jealous of his brother getting attention",
    "they felt embarrassed about making the mistake",
    "i am hopeful that things will get better soon",
    "she felt guilty about lying to her parents",
    "he was scared watching the horror movie alone",
    "they felt satisfied after completing the project",
    "i am curious about how things work around here",
    "she felt confused by the complex instructions given",
    "he was amazed by the beautiful sunset view",
    "they felt frustrated with the slow progress made",
    "i am comfortable in my new home now finally",
    "she felt anxious about the presentation tomorrow",
    "he was delighted to see his old friends again",
    "they felt refreshed after their short vacation",
    "i am determined to succeed in my goals this year",
    "she felt inspired by the motivational speaker",
    "he was annoyed by the constant interruptions today",
    "they felt peaceful walking through the quiet forest",
    "i am enthusiastic about starting my new job",
    "she felt overwhelmed by all the work assigned",
    "he was shocked to hear about the sudden news",
    "they felt blessed to have such a loving family",
    "i am grateful for every opportunity that comes",
    "she felt nostalgic looking at old family photos",
    "he was touched by the kind gesture from strangers",
    "they felt accomplished after finishing the marathon",
]


class SeedTextDataset(Dataset):
    """Dataset for training on seed text with train/val split."""

    def __init__(
        self,
        texts: List[str],
        seq_len: int = 64,
        stride: int = 1,
        augment: bool = True
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.augment = augment

        # Combine all texts with separators
        all_text = " ".join(texts)

        # Encode text to character IDs (ASCII)
        self.data = torch.tensor(
            [min(ord(c), 127) for c in all_text if ord(c) < 128],
            dtype=torch.long
        )

        # Create samples with sliding windows
        self.samples = []
        for i in range(0, max(1, len(self.data) - seq_len), stride):
            sample = self.data[i:i + seq_len + 1]
            if len(sample) >= 2:
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.samples[idx]

        # Data augmentation: random shift
        if self.augment and random.random() < 0.3:
            shift = random.randint(-3, 3)
            new_idx = max(0, min(len(self.samples) - 1, idx + shift))
            sample = self.samples[new_idx]

        return sample


def collate_sequences(batch: List[torch.Tensor]) -> torch.Tensor:
    """Collate variable-length sequences into a padded batch."""
    max_len = max(t.size(0) for t in batch)
    padded = []
    for t in batch:
        if t.size(0) < max_len:
            padding = torch.zeros(max_len - t.size(0), dtype=t.dtype)
            t = torch.cat([t, padding])
        padded.append(t)
    return torch.stack(padded)


class SelfStudyTrainer:
    """Trainer with validation split and overfitting detection."""

    def __init__(
        self,
        model: ZeroBaseLLM,
        config: Optional[ZeroBaseConfig] = None,
        device: Optional[torch.device] = None,
        seed_texts: Optional[List[str]] = None
    ):
        self.model = model
        self.config = config or model.config
        self.device = device or torch.device("cpu")
        self.seed_texts = seed_texts or DIVERSE_SEED_TEXTS

        # Move model to device
        self.model = self.model.to(self.device)

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        # Learning rate scheduler (cosine decay with linear warmup)
        self.warmup_steps = getattr(self.config, "warmup_steps", 200)
        self.total_steps = 20000

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return (step + 1) / max(self.warmup_steps, 1)
            progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            import math
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # Label smoothing loss
        label_smoothing = getattr(self.config, "label_smoothing", 0.1)
        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=-100
        )

        self.current_step = 0
        self.step = 0
        self.best_val_loss = float("inf")
        self.overfitting_detected = False

        # Create datasets with train/val split
        self._create_datasets()

    def _create_datasets(self):
        """Create train and validation datasets (80/20 split)."""
        full_dataset = SeedTextDataset(
            texts=self.seed_texts,
            seq_len=min(self.config.max_seq_len // 2, 128),
            stride=1,
            augment=True
        )

        # 80/20 split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_sequences,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_sequences,
            drop_last=False
        )

    def get_lr(self) -> float:
        """Get current learning rate."""
        lrs = self.scheduler.get_last_lr()
        return lrs[0] if lrs else self.config.learning_rate

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step with label-smoothed loss."""
        self.model.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)
        targets = batch[:, 1:].contiguous()
        inputs = batch[:, :-1].contiguous()

        outputs = self.model(inputs, use_self_study=True)
        char_hidden = outputs["char_hidden"]
        char_logits = self.model.char_output_layer(char_hidden)

        T = min(char_logits.size(1), targets.size(1))
        loss = self.criterion(
            char_logits[:, :T, :].reshape(-1, char_logits.size(-1)),
            targets[:, :T].reshape(-1)
        )

        if "self_study_loss" in outputs and outputs["self_study_loss"] is not None:
            ss_weight = getattr(self.config, "backward_study_weight", 0.2)
            loss = loss + ss_weight * outputs["self_study_loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        self.current_step += 1
        self.step += 1

        return {"loss": loss.item(), "lr": self.get_lr()}

    def validate(self) -> float:
        """Compute validation loss using the same criterion as training."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                targets = batch[:, 1:].contiguous()
                inputs = batch[:, :-1].contiguous()

                outputs = self.model(inputs, use_self_study=False)
                char_hidden = outputs["char_hidden"]
                char_logits = self.model.char_output_layer(char_hidden)

                T = min(char_logits.size(1), targets.size(1))
                loss = self.criterion(
                    char_logits[:, :T, :].reshape(-1, char_logits.size(-1)),
                    targets[:, :T].reshape(-1)
                )
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        num_steps: int = 20000,
        log_interval: int = 200,
        save_interval: int = 2000,
        save_dir: Optional[str] = None,
        generate_interval: int = 1000,
        generate_prompts: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """Training loop with validation and overfitting detection."""
        history = {"train_loss": [], "val_loss": [], "lr": [], "loss": []}

        # Sync scheduler's total steps with the actual training length
        self.total_steps = num_steps
        import math

        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return (step + 1) / max(self.warmup_steps, 1)
            progress = (step - self.warmup_steps) / max(num_steps - self.warmup_steps, 1)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

        for g in self.optimizer.param_groups:
            g["lr"] = self.config.learning_rate
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

        generate_prompts = generate_prompts or ["the ", "hello ", "once upon", "i think", "we should"]

        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Training for {num_steps} steps...\n")

        train_iter = iter(self.train_loader)
        best_train_loss = float("inf")

        for step in range(num_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            metrics = self.train_step(batch)
            history["train_loss"].append(metrics["loss"])
            history["loss"].append(metrics["loss"])
            history["lr"].append(metrics["lr"])

            # Logging with validation
            if (step + 1) % log_interval == 0:
                val_loss = self.validate()
                history["val_loss"].append(val_loss)

                avg_train = sum(history["train_loss"][-log_interval:]) / log_interval

                print(f"Step {step + 1}/{num_steps} | "
                      f"Train: {avg_train:.4f} | Val: {val_loss:.4f} | "
                      f"LR: {metrics['lr']:.6f}")

                # Overfitting detection: val_loss > 1.5 × train_loss
                if val_loss > avg_train * 1.5 and avg_train < 3.0:
                    print(f"\n  WARNING: OVERFITTING DETECTED!")
                    print(f"     Val loss ({val_loss:.4f}) > 1.5 × Train loss ({avg_train:.4f})")
                    print(f"     Stopping early at step {step + 1}")
                    self.overfitting_detected = True
                    break

                # Track best validation
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    print(f"     [OK] New best validation loss!")

            # Generate samples
            if (step + 1) % generate_interval == 0:
                self._show_generation(generate_prompts)

            # Save checkpoint
            if save_dir and (step + 1) % save_interval == 0:
                checkpoint_path = str(Path(save_dir) / f"checkpoint_{step + 1}.pt")
                self.model.save(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        return history

    def _show_generation(self, prompts: List[str]):
        """Show generated samples."""
        self.model.eval()
        print("\n" + "=" * 60)
        print("Generation Samples:")
        print("=" * 60)

        with torch.no_grad():
            for prompt in prompts:
                generated = self.model.generate(prompt, max_new_tokens=30, temperature=0.7)
                # Clean up output for display
                clean_output = ''.join(c if c.isprintable() else '?' for c in generated[:60])
                print(f"  '{prompt}' -> '{clean_output}'")
        print("=" * 60 + "\n")
        self.model.train()


def train_model(
    model: ZeroBaseLLM,
    seed_texts: Optional[List[str]] = None,
    num_steps: int = 20000,
    save_path: Optional[str] = None
) -> ZeroBaseLLM:
    """Train model with validation and overfitting detection."""
    trainer = SelfStudyTrainer(model, seed_texts=seed_texts)

    print(f"\n{'='*60}")
    print(f"Training Zero-Base LLM")
    print(f"{'='*60}")
    print(f"Model: {model.count_parameters():,} parameters")
    print(f"Size: {model.get_model_size_mb():.2f} MB")
    print(f"Seed texts: {len(trainer.seed_texts)} sentences")
    print(f"Dropout: {model.config.attention_dropout}")
    print(f"{'='*60}\n")

    # Show before training
    print("Before training:")
    for prompt in ["the ", "hello ", "once upon"]:
        gen = model.generate(prompt, max_new_tokens=15, temperature=0.8)
        clean = ''.join(c if c.isprintable() else '?' for c in gen[:30])
        print(f"  '{prompt}' -> '{clean}'")
    print()

    save_dir = str(Path(save_path).parent) if save_path else None

    trainer.train(
        num_steps=num_steps,
        log_interval=200,
        generate_interval=1000,
        save_interval=2000,
        save_dir=save_dir
    )

    if save_path:
        model.save(save_path)
        print(f"\nModel saved to {save_path}")

    return model