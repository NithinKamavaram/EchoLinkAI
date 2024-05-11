import pandas as pd
from slack_integration import get_user_id, get_latest_message
from calendly import generate_calendly_invitation_link
from conversation import GPT
from config import Config
import os
import time
import emoji
from langchain_experimental.generative_agents.generative_agent import GenerativeAgent
from langchain_experimental.generative_agents.memory import GenerativeAgentMemory
from langchain.chat_models import AzureChatOpenAI
from utils import create_new_memory_retriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
import slack
import warnings
warnings.filterwarnings('ignore')


def main():
    """
    Main function to run the EchoLink AI application. This function initializes
    the conversation environment, sets up the communication with Slack, and handles
    interactions with a professional user via a structured conversation flow.
    """
    # Dictionary mapping conversation stage identifiers to their descriptions.
    conversation_stages = {'1': "Introduction: Start the conversation by introducing yourself. Be polite and respectful while keeping the tone of the conversation professional.",
                       '2': "Value proposition1: Explain that firm is releasing 3 innovative new products(FINANCIAL STATEMENTS AUTOMATION, AUDITING AUTOMATION and COMPLIANCE AUTOMATION) which helps professional in their day to day work. Prior to rolling out functionality, firm has put together a training to be done by June 15th, which helps understanding the functionalities/features in the products.",
                       '3': "Value proposition2: Briefly explain how products like FINANCIAL STATEMENTS AUTOMATION(This tool automates the generation and management of financial statements, reducing manual errors and saving significant time), AUDITING AUTOMATION(t enhances the auditing process by automating routine tasks and analytics, thus increasing the accuracy and speed of audit reports), and COMPLIANCE AUTOMATION(This product ensures that financial practices adhere to the latest regulations automatically, reducing the risk of non-compliance and associated penalties.) helps accounting professional to use technology in their work.",
                       '4': "Needs analysis: Ask open-ended questions to uncover the professional needs and pain points. Listen carefully to their responses and take notes.",
                       '5': "Solution presentation: Based on the professional needs, present your products/services as the solution that can address their pain points.",
                       '6': "Objection handling: Address any objections that the professional may have regarding your products/services. Be prepared to provide evidence or testimonials to support your claims.",
                       '7': "Close: Ask professional if he is interested to know more about any product or interested in demo on any product to understand better.",
                       '8': "End conversation: It's time to end the chat by telling professional that they can find more information regarding products/services at https://aimakerspace.io/ and https://www.youtube.com/@AI-Makerspace/featured"
                       }
    # Create a Slack client using the token from environment variables.
    client = slack.WebClient(token=os.environ["SLACK_TOKEN"])
    # Initialize two instances of AzureChatOpenAI with different configurations.
    llm = AzureChatOpenAI(openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"], azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature = 0.2)
    llm_lucas = AzureChatOpenAI(openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"], azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"], temperature = 0)
    # Load and preprocess the dataset from a CSV file.
    df = pd.read_csv('./data/dataset.csv')
    df = df.groupby(['USER_ID', 'FIRST_NAME', 'LAST_NAME', 'EMAIL', 'TRAINING','PRODUCT_NAME_USED', 'RECOMMENDATION', 'FEEDBACK']).aggregate({'TRAINING_COMPLETED':list,
                                                                                                                                          'TRAINING_IN_PROGRESS':list, 
                                                                                                                                            'TRAINING_NOT_STARTED':list}).reset_index()
    # Load and preprocess the dataset from a CSV file.
    professionals_email = []
    professional_email = input("Enter the professional's email: ")
    professionals_email.append(professional_email)
    professionals_email = [x.lower() for x in professionals_email]
    for p in professionals_email:
        print(p)
        for i,r in df[df['USER_ID']==p].iterrows():
            try:
                professional_slack_id = get_user_id(p)
                professional_email = p
                professional_first_name = r['FIRST_NAME']
                professional_last_name = r['LAST_NAME']
                used = r['PRODUCT_NAME_USED']
                recommendation = r['RECOMMENDATION']
                feedback = r['FEEDBACK']
                training_completed = r['TRAINING_COMPLETED']
                training_in_progress = r['TRAINING_IN_PROGRESS']
                training_not_started = r['TRAINING_NOT_STARTED']
            except:
                continue
        config = dict(
            person_name = "Sophia",
            person_role = "To promote new products, features, trainings and gathering feedbacks from accounting professionals",
            team_name = "R&D",
            conversation_type = "chat",
            conversation_purpose = f"Introduce {recommendation} product/products explainng how it helps them, and recommend taking training and exploring {training_not_started}, aslo recommend professional to complete training on {training_in_progress} product since we know based on trainings data. Ask for {feedback} on  {used} product and later congratulate them for completing the {training_completed} training. Finally ask if there are interested in demo on any products they like.",
            conversation_history = [],
            conversation_stage = conversation_stages.get('1'),
            professional_name = professional_first_name
            )
        # Create and seed the marketing agent.
        marketing_agent = GPT.from_llm(llm, verbose=False, **config)
        marketing_agent.seed_agent()
        # Open a conversation channel and send the initial message.
        response = client.conversations_open(users=[professional_slack_id])
        if response["ok"]:
            channel_id = response["channel"]["id"]
        else:
            raise Exception("Failed to open conversation channel.")
        
        sending_message = marketing_agent.step()
        response = client.chat_postMessage(channel=channel_id, text=sending_message)
        response_count = 0
        # Handle incoming messages and responses.
        start_time = time.time()
        while True:
            latest_message = get_latest_message(channel_id)
            m_10 = []
            for m  in latest_message:   
                m['text'] = m['text'].replace("blush","smiling_face_with_smiling_eyes")
                m['text'] = m['text'].replace("wave","waving_hand")
                m['text'] = m['text'].replace("+1","thumbs_up")
                m['text'] = m['text'].replace("star2","glowing_star")
                m['text'] = m['text'].replace("bulb","light_bulb")
                m['text'] = m['text'].replace("tada","party_popper")
                m['text'] = m['text'].replace("smile","grinning_face_with_smiling_eyes")
                m['text'] = m['text'].replace("point_right","backhand_index_pointing_right")
                m['text'] = m['text'].replace("raised_hands","raising_hands")
                m['text'] = m['text'].replace("female-technologist","woman_technologist")
                m['text'] = m['text'].replace("male-technologist","man_technologist")
                m['text'] = m['text'].replace("hugging_face","smiling_face_with_open_hands")
                m['text'] = m['text'].replace("grinning_face_with_smiling_eyesy","grinning_face_with_big_eyes")
                m['text'] = m['text'].replace("sweat_grinning_face_with_smiling_eyes","grinning_face_with_sweat")
                m['text'] = m['text'].replace("grinning","grinning_face")
                m['text'] = m['text'].replace("grinning_face_face_with_big_eyes","grinning_face_with_big_eyes")
                m['text'] = m['text'].replace("grinning_face_face_with_sweat","grinning_face_with_sweat")
                m['text'] = m['text'].replace("sweat_smile","grinning_face_with_sweat")
                m['text'] = m['text'].replace(":one:",":keycap_1:")
                m['text'] = m['text'].replace(":two:",":keycap_2:")
                m['text'] = m['text'].replace(":mag_right:",":magnifying_glass_tilted_right:")
                m['text'] = m['text'].replace(":mortar_board:",":graduation_cap:")
                m['text'] = m['text'].replace(":female-student:",":woman_student:")
                
                m['text'] = m['text'].replace("&amp;","&")
                m['text'] = m['text'].replace("<",'')
                m['text'] = m['text'].replace(">",'')
                m['text'] = emoji.emojize(m['text'])
                if m['text']!= sending_message:
                    m_10.append(m['text'])
                else:
                    break
            m_10.reverse()
            latest_text=''
            if len(m_10) > 0:
                latest_text = ' '.join(m_10)
            else:
                latest_text = sending_message
            
            if latest_text==sending_message and (time.time() - start_time)/60 > 15 and response_count == 0:
                print(f'{professional_first_name} is busy, no response')
                conversation_history_backup = [sending_message]
                break
            elif latest_text==sending_message and (time.time() - start_time)/60 > 20 and response_count != 0:
                print(f'{professional_first_name} not completed the conversation')
                break
            else:
                if latest_text!=sending_message:
                    if (time.time() - start_time)/60 < 30:
                        response_count = response_count+1
                        print(sending_message)
                        print('-----')
                        print(latest_text)
                        print('====================')
                        marketing_agent.human_step(latest_text)
                        if marketing_agent.determine_conversation_stage().split(':')[0]!='End conversation':
                            sending_message = marketing_agent.step()
                            response = client.chat_postMessage(channel=channel_id, text=sending_message)
                        elif marketing_agent.determine_conversation_stage().split(':')[0]=='End conversation':
                            print(f'{professional_first_name} completed the conversation')
                            break
                    elif (time.time() - start_time)/60 > 25:
                        response_count = response_count+1
                        print(sending_message)
                        print('-----')
                        print(latest_text)
                        print('====================')
                        marketing_agent.human_step(latest_text)
                        if marketing_agent.determine_conversation_stage().split(':')[0]!='End conversation':
                            sending_message = marketing_agent.step()
                            response = client.chat_postMessage(channel=channel_id, text=sending_message)
                            print(f'{professional_first_name} not completed the conversation')
                            break
                        elif marketing_agent.determine_conversation_stage().split(':')[0]=='End conversation':
                            break
        
        conversation_history_backup = marketing_agent.get_conversation_history_backup()          
        conversation_history_backup_summary = []
        for h in conversation_history_backup:
            if (conversation_history_backup.index(h)%2) == 0:
                conversation_history_backup_summary.append('Lucas - ' + h)
            else:
                conversation_history_backup_summary.append(f'{professional_first_name} - ' + h)
                
        lucas_memory = GenerativeAgentMemory(
            llm = llm_lucas,
            memory_retriever = create_new_memory_retriever(),
            verbose=False,
            reflection_threshold = 9
            )

        lucas = GenerativeAgent(
            name = "Lucas",
            status = "Lucas helps to get summary of conversation, also helps to answer for questions based on conversation history given",
            llm = llm_lucas,
            memory = lucas_memory,
            memory_retriever = create_new_memory_retriever()
            )
        
        for observation in conversation_history_backup_summary:
            lucas.memory.add_memory(observation)
        
        docs = lucas.memory.memory_retriever.memory_stream
        
        llm = llm_lucas
        
        map_template = """The following is a set of documents
        {docs}
        Based on this list of docs, does professional showed interest in demo, say only 'YES' or 'NO'
        Helpful Answer:"""
        map_prompt = PromptTemplate.from_template(map_template)
        map_chain = LLMChain(llm=llm, prompt=map_prompt)
        demo = map_chain.run(docs)
        if demo == 'YES':
            sending_message_meeting = generate_calendly_invitation_link()
            client.chat_postMessage(channel=channel_id, text=f"Below is the meeting link for demo {professional_first_name}\n===========================================\n")
            client.chat_postMessage(channel=channel_id, text=sending_message_meeting)

        chain = load_summarize_chain(llm, chain_type="stuff")
        summary = chain.run(docs)
        # Replace name_place_holder and channelID_place_holder with your own data/text
        sending_message_summary = f"Hi {name_place_holder}, Below is the summary conversation happened with {professional_first_name} {professional_last_name}\n==========\n{summary}\n==========="
        response = client.chat_postMessage(channel={channelID_place_holder}, text=sending_message_summary)
        

if __name__ == "__main__":
    main()