# import RPi.GPIO as GPIO
# import time
#
# GPIO.setmode(GPIO.BOARD)
# GPIO.setup(7, GPIO.OUT)
#
# def Blink(speed):
#     GPIO.output(7,True)
#     time.sleep(speed)
#     GPIO.output(7,False)
#     time.sleep(speed)
#     GPIO.cleanup()
#
# Blink(1)
#
# print("blinker 1 activated")


from gpiozero import LED
from time import sleep

red = LED(23)

while True:

    red.on()
    print('LED ON')
    sleep(1)
    red.off()
    print('LED OFF')
    sleep(1)
