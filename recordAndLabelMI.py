from tkinter import *
from pyOpenBCI import OpenBCICyton
import threading
import time
import numpy as np

# Simiar to button interface - for data signal aqusition and labeling
# "Rest" displayed for 3 seconds, then "Raise hand" and count down for 3 seconds
# Could be used for motor imaginary
 
root = Tk()
raw_data = []
board = OpenBCICyton(port='COM5', daisy=True)
action = False
buttonCounter = 0

def start_countdown():
    global action
    global buttonCounter

    countdown_label.config(text="Rest")
    action = False
    print(action)
    root.update()
    time.sleep(3)

    for i in range(20):  # Perform the action 20 times
        countdown_label.config(text="Raise hand")
        action = True
        print(action)
        root.update()
        time.sleep(1)

        for j in range(2, 0, -1):  # Countdown from 5 to 1
            countdown_label.config(text=str(j))
            root.update()
            time.sleep(1)

        countdown_label.config(text="Rest")
        action = False
        print(action)
        root.update()
        time.sleep(3)

        buttonCounter += 1
        print(buttonCounter)
    
    board.stop_stream()  # Stop the stream
    root.destroy()  # Close the Tkinter window
    
def print_raw(sample):
    global action
    if action:
        print("action")
        dataWithLabel = sample.channels_data + [1]  # Label as 1 during action
    else:
        print("no")
        dataWithLabel = sample.channels_data + [0]  # Label as 0 during rest
    raw_data.append(dataWithLabel)

def start_openbci_stream():
    board.start_stream(print_raw)

countdown_label = Label(root, text="", font=("Helvetica", 48))
countdown_label.pack()

# Start the OpenBCI stream in a separate thread
openbci_thread = threading.Thread(target=start_openbci_stream)
openbci_thread.start()

# Start countdown in a separate thread
countdown_thread = threading.Thread(target=start_countdown)
countdown_thread.start()

root.mainloop()

raw_data_np = np.array(raw_data)
filename = f"{'laba/data_for_right_hand'}/{int(time.time())}.npy"
np.save(filename, raw_data_np)
