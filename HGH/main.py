"""
Home-Gym-Homie main python file
Authors: Matthew, Nouwen, and Sebastian
Proj-6100

The Application uses Open-Cv to monitor exercises, then records the exercise, and sends the data to the user in a report format

Hardware - Rasperry Pi with a hat, the pi controls servos on the hat that will act as a pan-tilt controller for the camera

Initial TODO: 
[x] set up files
[] create OOP format
[] push to github
[x] create servo.py
[x] create workout.py
[x] create data.py
[] create UX and b4a
[] ...
"""
# imports
import data
import servo
import workout
import flask
