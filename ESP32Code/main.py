import network
from machine import Pin
from time import sleep
from microdot import Microdot

led = Pin(2, Pin.OUT)  # Replace with your LED pin
app = Microdot()

hp = 100

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
    if state == 'dead':
        led.value(1)
        return 'LED turned ON'
    elif state == 'alive':
        led.value(0)
        return 'LED turned OFF'
    elif int(state) < hp:
        hp = state
        led.value(1)
        sleep(1)
        led.value(0)
    elif state == 100:
        hp == 100
    else:
        return 'Invalid state', 400

app.run(host=wlan.ifconfig()[0], port=8080)



