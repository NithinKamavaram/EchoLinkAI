import os
import time
from config import Config
import urllib.error
from slack.errors import SlackApiError
import slack

# Initialize the Slack client using the token from configuration.
client = slack.WebClient(token=Config.SLACK_TOKEN)
def get_user_id(email):
    """
    Retrieves the Slack user ID for a given email address.
    
    Args:
        email (str): Email address to query for user ID.
    
    Returns:
        str or None: User ID if found, otherwise None.
    """
    try:
        response = client.users_lookupByEmail(email=email)
        if response["ok"]:
            return response["user"]["id"]
        else:
            print(f"Failed to get user ID for {email}: {response['error']}")
    except SlackApiError as e:
        print(f"Slack API Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    return None

def get_latest_message(channel_id, retry_attempts=5, timeout=2):
    """
    Retrieves the latest message from a specified Slack channel with retry mechanism.
    
    Args:
        channel_id (str): Channel ID from which to fetch the message.
        retry_attempts (int): Number of times to retry the fetch in case of failure.
        timeout (int): Initial timeout in seconds between retries, doubles on each retry.
    
    Returns:
        dict or None: Latest message if successful, otherwise None.
    """
    attempt = 0
    while attempt < retry_attempts:
        try:
            response = client.conversations_history(channel=channel_id, limit=1)
            if response["ok"]:
                return response['messages']
        except (SlackApiError, urllib.error.URLError) as e:
            print(f"Network error occurred: {e}. Retrying...")
            time.sleep(timeout)
            attempt += 1
            timeout *= 2  # Exponential backoff
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    return None
