import requests
import urllib3
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def get_reward_response(prompt, max_retries=3):
    url = "https://35.aigcbest.top/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "YOUR_API_KEY_HERE"
    }
    data = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers, verify=False, timeout=(100, 200))
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif attempt == max_retries - 1:
                return f"Error: {response.status_code}, {response.text}"
                
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
            else:
                return f"Error: {type(e).__name__} - {str(e)}"
    
    return "Error: Max retries exceeded"

if __name__ == "__main__":
    print(get_reward_response("haha"))