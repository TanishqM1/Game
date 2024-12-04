import network
from machine import Pin
from time import sleep
from microdot import Microdot

led = Pin(2, Pin.OUT)  # Replace with your LED pin
app = Microdot()

# Connect to Wi-Fi
ssid = 'TELUS8280'
password = '292imzr69p'

wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(ssid, password)

while not wlan.isconnected():
    print('Connecting to Wi-Fi...')
    sleep(1)

print('Connected, IP address:', wlan.ifconfig()[0])

@app.route('/led/<state>')
def toggle_led(request, state):
    if state == 'pos':
        led.value(1)
        return 'LED turned ON'
    elif state == 'neg':
        led.value(0)
        return 'LED turned OFF'

app.run(host=wlan.ifconfig()[0], port=8080)



