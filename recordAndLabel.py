from tkinter import *
from pyOpenBCI import OpenBCICyton
import threading
import time
import numpy as np

# Creates a button for signal aqusition and labeling
# When button pressed - rest state, when button released - action state
# Button has to be pressed with computer mouse not touchpad
# One recording session consists of 20 hand raises
# Data is saved as numpy file

buttonCounter = 0
root = Tk()
raw_data = []
board = OpenBCICyton(port='COM5', daisy=True)
action = False

def on_click(event):
    global buttonCounter 
    global action

    action = False
    print(action)

    myButton.config(text="Release and move hand")
    if buttonCounter == 20:
        root.destroy()
        time.sleep(2)
        board.stop_stream()

def on_release(event):
    global buttonCounter 
    global action
    global root

    myButton.config(text="Click me to finish")
    buttonCounter += 1

    action = True
    print(buttonCounter)
    print("pressed")
    print(action)

def print_raw(sample): 
    global action
    if action == False:
        dataWithLabel = sample.channels_data + [0]
        raw_data.append(dataWithLabel)
        #add 0 as 17th element        
    else:
        dataWithLabel = sample.channels_data + [1]
        raw_data.append(dataWithLabel)
        #add 1 as 17th element   

def start_openbci_stream():
    board.start_stream(print_raw)

myButton = Button(root, text="Click me to start", fg="white", bg="red", width=200, height=100, font=("Arial", 40))
myButton.bind("<Button-1>", on_click)
myButton.bind("<ButtonRelease-1>", on_release)
myButton.pack()

# Start the OpenBCI stream in a separate thread
openbci_thread = threading.Thread(target=start_openbci_stream)
openbci_thread.start()

root.mainloop()

raw_data_np = np.array(raw_data)
filename = f"{'laba/data_for_right_hand'}/{int(time.time())}.npy"
np.save(filename, raw_data_np)