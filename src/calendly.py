import requests
from config import Config

def generate_calendly_invitation_link():
    # Setup headers for the API request, including the authorization token from the config.
    headers = {
        'Authorization': f'Bearer {Config.CALENDLY_API_KEY}',
        'Content-Type': 'application/json'
    }
    # Calendly API URL for creating scheduling links.
    url = 'https://api.calendly.com/scheduling_links'
    # Payload for the POST request, defining the maximum number of events and the owner details.
    payload = {
        "max_event_count": 1,  # Limits the number of events that can be scheduled.
        "owner": f"https://api.calendly.com/event_types/{Config.CALENDLY_EVENT_UUID}",  # Specify the event type by UUID.
        "owner_type": "EventType"  # Owner type is set to 'EventType'.
    }
    
    # Send a POST request to the Calendly API to create a scheduling link.
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 201:  # Check if the request was successful (HTTP 201 Created).
        data = response.json()  # Parse the JSON response.
        return data['resource']['booking_url']  # Return the booking URL from the response.
    else:
        # Print the error if the link creation fails and return None.
        print(f"Failed to create Calendly link: {response.status_code}, {response.text}")
        return None