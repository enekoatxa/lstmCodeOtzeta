import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import rng
import threading

def loadRNG():
    global rngInstance
    rngInstance = rng.RNG()
    messagebox.showinfo("Info", "Sequences loaded.")

def generate_action():
    # Get the values from the text fields
    animation_length = entry_animation_length.get()
    animation_piece_length = entry_animation_piece_length.get()
    interpolation_length = entry_interpolation_length.get()
    interpolation_length_for_actions = entry_interpolation_length_for_actions.get()
    actions_entry_frames = entry_actions_entry_frames.get()
    num_animations = entry_num_animations.get()
    
    # Check if any text fields are empty
    if not animation_length or not interpolation_length or not num_animations or not animation_piece_length or not interpolation_length_for_actions:
        messagebox.showerror("Error", "Please fill in all the text fields.")
        return
    
     # Check if the values are integers
    try:
        animation_length = int(animation_length)
        animation_piece_length = int(animation_piece_length)
        interpolation_length = int(interpolation_length)
        interpolation_length_for_actions = int(interpolation_length_for_actions)
        if(actions_entry_frames!=""):
            actions_entry_frames = [int(action) for action in actions_entry_frames.split(",")]
        num_animations = int(num_animations)
    except ValueError:
        messagebox.showerror("Error", "Please enter valid integers in the text fields.")
        return
    
    # Get the selected actions
    selected_actions = [action for action, var in checkboxes.items() if var.get()]
    
    # Check if the Smooth checkbox is selected
    smooth = checkbox_smooth.get()
    loop = checkbox_loop.get()
    for i in range(0, num_animations):
        try:
            if(animation_piece_length<=interpolation_length):
                messagebox.showerror("Error", "The animation piece length can't be shorter than the interpolation length.")
            else:
                # GENERATE THE BASE SEQUENCE
                rngInstance.setState(animation_length=animation_length, animation_piece_length= animation_piece_length, 
                                    interpolation_length = interpolation_length, interpolation_length_for_actions = interpolation_length_for_actions, 
                                    num_animations = num_animations, action_list = selected_actions, loop = loop)
                rngInstance.generateBaseSequence()
                # ADD THE ACTIONS AND CHECK ALL THE POSSIBLE ERRORS
                if len(selected_actions) > 0:
                    if(len(actions_entry_frames)==0):
                        messagebox.showinfo("Info", "Even if you selected actions to introduce, you did not set any entry points for the actions. They will be ignored.")
                    elif(not all(actions_entry_frames[i] < actions_entry_frames[i + 1] for i in range(len(actions_entry_frames) - 1))):
                        messagebox.showerror("Error", "The animation entry points need to be in ascending order. They will be ignored.")
                    elif(not all(abs(actions_entry_frames[i + 1] - actions_entry_frames[i]) > 499 for i in range(len(actions_entry_frames) - 1))):
                        messagebox.showerror("Error", "We suggest the differences between the entry points to be larger than 500 frames. They will be ignored.")
                    elif(actions_entry_frames[0]<interpolation_length_for_actions):
                        messagebox.showerror("Error", f"The first action must be introduced at least at frame {interpolation_length_for_actions}, so interpolation can be done correctly.")
                    elif(max(actions_entry_frames)>animation_length):
                        messagebox.showerror("Error", "There is at least one action introduced after the total animation length.")
                    else:
                        for action_entry_frame in actions_entry_frames:
                            rngInstance.addSpecialAnimation(action_entry_frame)
                    if(loop and max(actions_entry_frames)-400>animation_length):
                        messagebox.showinfo("Info", "There is at least one action very near to the end of the animatino. Looping may not work correctly.")
                if(len(selected_actions) <= 0 and len(actions_entry_frames)>0):
                    messagebox.showinfo("Info", "Even if you set entry points for the actions, you did not choose the types of actions to introduce. They will be ignored.")
                # WRITE THE FILE
                rngInstance.writeSequence(i)
                # WRITE THE SMOOTHED FILE IF NEEDED
                if smooth:
                    rngInstance.writeSoftSequence(i)
        except Exception as ex:
            # the lenth of the action iterpolation is too long
            if(type(ex).__name__ == "IndexError"):
                messagebox.showerror("Error", "The length of the interpolation length for actions is longer than the action itself.")
            if(type(ex).__name__ == "NameError"):
                messagebox.showerror("Error", "The sequences are loading, please wait.")
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

# Create the main window
root = tk.Tk()
root.title("Animation Generator")

# Apply a modern theme
style = ttk.Style()
style.theme_use('clam')  # You can use 'clam', 'alt', 'default', or 'classic'

# Configure the main window grid
root.columnconfigure(0, weight=1)
root.rowconfigure([0, 1, 2, 3], weight=1)

# Create a frame for checkboxes
frame_checkboxes = ttk.LabelFrame(root, text="Select Actions")
frame_checkboxes.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

# List of actions for checkboxes
actions = ["Look Up", "Look Around", "Look Down", "Look Shoes", "Look Back L", "Look Back R", "Take phone", "Watch"]

# Dictionary to hold the checkbox variables
checkboxes = {}

# Create checkboxes
for action in actions:
    var = tk.BooleanVar()
    chk = ttk.Checkbutton(frame_checkboxes, text=action, variable=var)
    chk.pack(anchor='w', padx=5, pady=2)
    checkboxes[action] = var

# Create a frame for the "Smooth" checkbox
frame_other_options = ttk.LabelFrame(root, text="Smooth Option")
frame_other_options.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

# Add the "Smooth" checkbox
checkbox_smooth = tk.BooleanVar()
chk_smooth = ttk.Checkbutton(frame_other_options, text="Smooth", variable=checkbox_smooth)
chk_smooth.pack(anchor='w', padx=5, pady=2)

# Add the "Loop" checkbox
checkbox_loop = tk.BooleanVar()
chk_loop = ttk.Checkbutton(frame_other_options, text="Loop", variable=checkbox_loop)
chk_loop.pack(anchor='w', padx=5, pady=2)

# Create a frame for text fields
frame_textfields = ttk.LabelFrame(root, text="Parameters")
frame_textfields.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

# Animation length text field
ttk.Label(frame_textfields, text="Animation Length:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
entry_animation_length = ttk.Entry(frame_textfields)
entry_animation_length.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
entry_animation_length.insert(0, "3500")

# Animation piece length text field
ttk.Label(frame_textfields, text="Animation Piece Length:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
entry_animation_piece_length = ttk.Entry(frame_textfields)
entry_animation_piece_length.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
entry_animation_piece_length.insert(0, "400")

# Interpolation length text field
ttk.Label(frame_textfields, text="Interpolation Length:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
entry_interpolation_length = ttk.Entry(frame_textfields)
entry_interpolation_length.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
entry_interpolation_length.insert(0, "100")

# Interpolation length for special animations text field
ttk.Label(frame_textfields, text="Interpolation Length For Actions:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
entry_interpolation_length_for_actions = ttk.Entry(frame_textfields)
entry_interpolation_length_for_actions.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
entry_interpolation_length_for_actions.insert(0, "50")

# Actions per minute text field
ttk.Label(frame_textfields, text="Insert actions at frames:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
entry_actions_entry_frames = ttk.Entry(frame_textfields)
entry_actions_entry_frames.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
entry_actions_entry_frames.insert(0, "500,1100")

# Number of animations text field
ttk.Label(frame_textfields, text="Number of animations:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
entry_num_animations = ttk.Entry(frame_textfields)
entry_num_animations.grid(row=5, column=1, padx=5, pady=5, sticky="ew")
entry_num_animations.insert(0, "3")

# Create the generate button
generate_button = ttk.Button(root, text="Generate", command=generate_action)
generate_button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

# create the rng class
t1 = threading.Thread(target=loadRNG, name="loadThread", daemon=True)
t1.start()

# Run the application
root.mainloop()
