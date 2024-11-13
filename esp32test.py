from machine import Pin
from detectbullet import run_detectbullet
import time

class Pin:
    OUT = 1

    def __init__(self, pin_number, pin_type):
        self.pin_number = pin_number
        self.pin_type = pin_type

    def value(self, val):
        print(f"Pin {self.pin_number} set to {val}")
    

p = Pin(2, Pin.OUT)


def toggle(max):
    lap = 0

    while lap < max:
        p.value(1)
        time.sleep(1)
        p.value(0)
        time.sleep(1)
        lap+=1

while (True):
    # if (run_detectbullet()): 
    toggle(1)