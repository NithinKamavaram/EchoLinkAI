# EchoLink AI

![EchoLinkAI](images/project.png "project image")

## Description
This project is designed to automate interactions with professionals via Slack, using the EchoLink AI system. It integrates Slack for communications, manages scheduling with Calendly, and uses llm for language understanding and conversation management.

## Setup
1. Clone this repository.
2. Install the required Python packages: 
`pip install -r requirements.txt`
3. Create a `.env` file in the project root directory with the necessary environmental variables:
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT_NAME
AZURE_EMBEDDING_DEPLOYMENT_NAME
LANGCHAIN_SMITH_API_KEY
CALENDLY_EVENT_UUID
CALENDLY_API_KEY
SLACK_TOKEN

NOTE: You can also other llm api providers or can use open source llm's.

## Running the Application
Run the application by executing:
`python src/main.py`

## Requirements
Ensure you are running Python 3.8 or newer. This project depends on several external libraries listed in requirements.txt, crucial for maintaining functionality across different systems.

## Features
EchoLink AI leverages advanced language understanding and interaction management to automate communications. Key features include:
- **Real-Time Messaging**: Integrates seamlessly with Slack, enabling prompt and efficient interactions with professionals.
- **Automated Scheduling**: Incorporates Calendly for easy scheduling, improving the efficiency of booking meetings and events.
- **Intelligent Conversation Handling**: Utilizes cutting-edge language models to enhance conversation flow, ensuring that interactions are both natural and effective.

## Support and Contact
If you need support with the project or have any queries, feel free to reach out to me.
- **LinkedIn**: [Nithin Kamavaram](https://www.linkedin.com/in/nkamavaram/)