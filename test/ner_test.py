import spacy
from spacy import displacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained(
    "Babelscape/wikineural-multilingual-ner")
model = AutoModelForTokenClassification.from_pretrained(
    "Babelscape/wikineural-multilingual-ner")

nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
raw_text = "Blasphemous is a Metroidvania action-adventure game taking place in the fictional region of Cvstodia. Players assume control of the Penitent One, a silent knight wielding a sword named Mea Culpa, as he travels the land in a pilgrimage.\
\
The game involves exploring Cvstodia while fighting enemies, which appear in most areas. The Penitent One can fight enemies by attacking them with his sword at close range, or by casting spells that can be learned. By damaging enemies with melee attacks, the player gains Fervor, which is consumed to cast spells. Each enemy has a certain attack pattern which players must learn in order to dodge them and avoid taking damage. Some enemy attacks can be parried by blocking at the right time, leaving foes vulnerable and allowing the Penitent One to counterattack them for increased damage. When getting hit, the protagonist's health decreases, but it can be recovered by consuming Bile Flasks. Defeating enemies rewards Tears of Atonement, the game's currency, that can be spent on shops to upgrade the player character and obtain items.\
\
Numerous upgrades can be acquired at various points of the adventure, which include increasing the Penitent One's maximum health, Fervor and amount of Bile Flasks carried, and unlocking new abilities for world exploration and combat. By exploring, interacting with NPCs and completing sidequests, multiple items can be found which, when equipped, provide stat bonuses, reduce or nullify certain types of damage or provide access to otherwise inaccessible areas. There are collectibles in the form of bones that can be delivered in a certain place to receive rewards, and Children of Moonlight – trapped angels that can be freed by attacking the cages they are in.\
\
There are multiple checkpoints in the forms of altars located in multiple areas of the map. The player can rest in these checkpoints to fully replenish their health and refill any used Bile Flasks, save their progress and equip certain abilities, but doing so will also cause all previously slain enemies (excluding bosses) to respawn\
\
The Penitent One will die if his health is fully depleted, or if he falls into spikes or into a bottomless pit. Upon death, he will respawn in the last checkpoint visited, and a Guilt Fragment will appear in the location of his death (or near it, if he was killed by spikes or falling). The player will have reduced maximum Fervor, and gain less Fervor and Tears of Atonement from enemies, until the Guilt Fragment is recovered by reaching its location and interacting with it. Alternatively, there are certain points where this penalty can be eliminated for a fee. Additionally, after every boss all guilt is alleviated as well. "

ner_results = nlp(raw_text)
for entity in ner_results:
    print(
        f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']}")

NER = spacy.load("en_core_web_sm")

text1 = NER(raw_text)

for word in text1.ents:
    print(word.text, word.label_)
