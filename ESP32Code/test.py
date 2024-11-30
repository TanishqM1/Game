import requests
import time
esp_ip = '192.168.1.78'  # Replace with your ESP32's IP
port = 8080

def change_led(state):
    url = f'http://{esp_ip}:{port}/led/{state}'
    try:
        response = requests.get(url)
        print(response.text)
    except Exception as e:
        print(f"Error: {e}")

# Turn the LED on or off
change_led('dead')  # Turn the LED on
time.sleep(2)
change_led('alive')  # Turn the LED off
print("done")