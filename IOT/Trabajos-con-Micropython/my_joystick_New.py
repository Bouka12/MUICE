#*************************************#
# Nombre y Apellido =  Mabrouka Salmi
#*************************************#

from joystick import Joystick
from rgbled import RGBLed
from time import sleep
import uasyncio
from math import atan2, pi
#pi=math.pi
# Cartesianas a polares
#r = √ (x2 + y2)

#θ = atan( y / x )

mi_joystick = Joystick(36,39)
mi_rgbled = RGBLed(18,5,19)
# red: Pin D18; [0,2pi/3[
# Blue: Pin D5  ; [2pi/3,4pi/3[
# green: Pin D19 [4pi/3, 2pi[

while True:
    #mi_rgbled.set_color(0,0,0)
    x,y = mi_joystick.read_xy()
    #print(x,y)
    #r = (x**2+y**2)**(0.5)
    a = atan2(y,x)
    if a< 0:
        a+=2* pi # to solve the problem of getting the led lights only on half of the joystick area
    if 0<=a and a< pi/3:
        # RED
        mi_rgbled.set_color(255,0,0)
    if pi/3<=a and a< 2*pi/3:
        # GREEN
        mi_rgbled.set_color(0,255,0)
    if 2*pi/3<=a and a<pi: 
        # BLUE
        mi_rgbled.set_color(0,0,255)
        
        
    