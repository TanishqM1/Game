import network
from machine import Pin, PWM
import time
from microdot import Microdot

led = Pin(2, Pin.OUT)  # Replace with your LED pin
app = Microdot()

servo_pin = 23  # Change this to the GPIO pin you're using
servo = PWM(Pin(servo_pin), freq=50)  # 50Hz for servo

# Function to set servo angle
def set_servo_angle(angle):
    min_us = 500   # 0° position (adjust if needed)
    max_us = 2500  # 180° position (adjust if needed)
    
    duty = int((angle / 180) * (max_us - min_us) + min_us)
    servo.duty_u16(int(duty / 20000 * 65535))  # Convert microseconds to duty cycle
    time.sleep(0.5)  # Give time for the servo to move


# Connect to Wi-Fi
ssid = 'user'
password = 'pass'

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(ssid, password)

while not wlan.isconnected():
    print('Connecting to Wi-Fi...')
    time.sleep(1)

print('Connected, IP address:', wlan.ifconfig()[0])

@app.route('/led/<state>')
def toggle_led(request, state):
    if state == 'pos':
        # Move servo
        set_servo_angle(90)  # Move to 90 degrees
        time.sleep(2)         # Wait for 5 seconds
        set_servo_angle(0)    # Move back to 0 degrees
        return 'Motor Moved to 90deg'
    elif (state == 'neg'):
        set_servo_angle(0)    # Move back to 0 degrees
        return 'Motor Moved to 0deg'

app.run(host=wlan.ifconfig()[0], port=8080)

# Define PWM parameters










