import pandas as pd
import string

csv_path = 'Dataset/data.csv'
domainYML_path = 'Test/domain.yml'
nluYML_path = 'Test/data/nlu.yml'
storiesYML_path = 'Test/data/stories.yml'

def preprocess_intent(intent):
    intent = intent.lower()
    intent = intent.translate(str.maketrans("", "", string.punctuation))
    intent = ''.join(char if char != '’' else "'" for char in intent)
    return '_'.join(intent.split())

def dummitize(sentence):
    return ("Hello, " + sentence + " Thanks!")

def generate_files(data_csv_path, domain_yml_path, nlu_yml_path, stories_yml_path):
    # Load data from CSV
    data = pd.read_csv(data_csv_path)

    unq = set()

    intents = []
    responses = []

    nlu_content = """version: "3.1"

nlu:
- intent: greet
  examples: |
    - hey
    - hello
    - hi
    - hello there
    - good morning
    - good evening
    - moin
    - hey there
    - let's go
    - hey dude
    - goodmorning
    - goodevening
    - good afternoon

- intent: goodbye
  examples: |
    - cu
    - good by
    - cee you later
    - good night
    - bye
    - goodbye
    - have a nice day
    - see you around
    - bye bye
    - see you later

- intent: affirm
  examples: |
    - yes
    - y
    - indeed
    - of course
    - that sounds good
    - correct

- intent: deny
  examples: |
    - no
    - n
    - never
    - I don't think so
    - don't like that
    - no way
    - not really

- intent: mood_great
  examples: |
    - perfect
    - great
    - amazing
    - feeling like a king
    - wonderful
    - I am feeling very good
    - I am great
    - I am amazing
    - I am going to save the world
    - super stoked
    - extremely good
    - so so perfect
    - so good
    - so perfect

- intent: mood_unhappy
  examples: |
    - my day was horrible
    - I am sad
    - I don't feel very well
    - I am disappointed
    - super sad
    - I'm so sad
    - sad
    - very sad
    - unhappy
    - not good
    - not very good
    - extremly sad
    - so saad
    - so sad

- intent: bot_challenge
  examples: |
    - are you a bot?
    - are you a human?
    - am I talking to a bot?
    - am I talking to a human?

- intent: fallback
  examples: |
    - Tell me a joke.
    - What's the weather like today?
    - Who won the latest football match?
    - Can you recommend a good book?
    - What's your favorite color?
    - How tall is Mount Everest?
    - What's the capital of France?
    - Explain the concept of quantum entanglement.
    - Who is the president of the United States?
    - How do I bake chocolate chip cookies?
    - What's the latest news?
    - Can you play a game with me?
    - Recommend a movie to watch.
    - Tell me a fun fact.
    - What's the meaning of life?

"""

    stories_content = []

    for i, (index, row) in enumerate(data.iterrows()):
        user_question = str(row['User Questions']).replace("‘", "'").replace("’", "'")
        intent = "custom_int_" + str(i + 1)
        response = str(row['Chatbot Answers']).strip().replace("\\", "\\\\").replace("\"","\\\"").replace("\n", " ") 

        n = len(unq)
        unq.add(user_question)
        m = len(unq)
        if m == n: continue

        intents.append(intent)
        responses.append({'utterance': user_question, 'intent': intent, 'response': response})

        # Create NLU content with Dummy cause Minimum 2 Examples were needed to bypass the Warnings
        user_question_dummy = dummitize(user_question)
        nlu_content += f"- intent: {intent}\n  examples: |\n    - {user_question}\n    - {user_question_dummy}\n\n"

        # Create Stories content
        stories_content.append({
            'story': f'Story {i + 1}',
            'steps': [
                {'intent': intent},
                {'action': f'utter_{intent}'}
            ]
        })

    # Create domain.yml content
    domain_content = """version: "3.1"

intents:
  - greet
  - goodbye
  - affirm
  - deny
  - mood_great
  - mood_unhappy
  - bot_challenge
  - fallback

"""
    
    domain_content += "\n".join(f"  - {intent}" for intent in intents) + "\n\nresponses:\n"

    for response_data in responses:
        intent = response_data['intent']
        response = response_data['response']
        domain_content += f"  utter_{intent}:\n    - text: \"{response}\"\n"

    domain_content += """
  utter_greet:
  - text: "Hey! How are you?"

  utter_cheer_up:
  - text: "Here is something to cheer you up:"
    image: "https://i.imgur.com/nGF1K8f.jpg"

  utter_did_that_help:
  - text: "Did that help you?"

  utter_happy:
  - text: "Great, carry on!"

  utter_goodbye:
  - text: "Bye"

  utter_iamabot:
  - text: "I am a bot, powered by Rasa."

  utter_default:
    - text: "I am sorry, I didn't understand that. Can you please rephrase or ask something else?"

session_config:
  session_expiration_time: 180
  carry_over_slots_to_new_session: true

"""

    # Write to domain.yml file with UTF-8 encoding
    with open(domain_yml_path, 'w', encoding='utf-8') as domain_file:
        domain_file.write(domain_content)

    # Write to nlu.yml file with UTF-8 encoding
    with open(nlu_yml_path, 'w', encoding='utf-8') as nlu_file:
        nlu_file.write(nlu_content)

    # Write to stories.yml file with UTF-8 encoding
    with open(stories_yml_path, 'w', encoding='utf-8') as stories_file:
        flag = True
        for story_data in stories_content:
            if flag: 
                stories_file.write("""version: "3.1"

stories:

- story: happy path
  steps:
  - intent: greet
  - action: utter_greet
  - intent: mood_great
  - action: utter_happy

- story: sad path 1
  steps:
  - intent: greet
  - action: utter_greet
  - intent: mood_unhappy
  - action: utter_cheer_up
  - action: utter_did_that_help
  - intent: affirm
  - action: utter_happy

- story: sad path 2
  steps:
  - intent: greet
  - action: utter_greet
  - intent: mood_unhappy
  - action: utter_cheer_up
  - action: utter_did_that_help
  - intent: deny
  - action: utter_goodbye

- story: Fallback
  steps:
  - intent: fallback
  - action: utter_default
                                   
""")
                flag = False
            stories_file.write(f"\n- story: {story_data['story']}\n  steps:\n")
            for step in story_data['steps']:
                for step_type, value in step.items():
                    stories_file.write(f"  - {step_type}: {value}\n")

generate_files(csv_path, domainYML_path, nluYML_path, storiesYML_path)
