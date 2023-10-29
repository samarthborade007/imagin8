import numpy as np
import cv2
import cvzone
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import sys
import threading


# Global variables
img_path = None
img_display = None
live_label = None


# Create the main application window
root = tk.Tk()
root.title("imagin8")

root.configure(bg="#6082B6")  # Light gray background color


def apply_vintage_grains_effect3(input_image, overlay_path='filters\Films\kodak.png', resize_factor=(1, 1.5), exposure=0.7, noise_strength=0.3):
    try:
        # Generate random noise
        noise = np.random.normal(0, 1, input_image.shape[:2]).astype('uint8')
        noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        
        # Resize the noise to match the dimensions of input_image
        noise = cv2.resize(noise, (input_image.shape[1], input_image.shape[0]))
        
        # Add the noise to the input image with adjusted exposure
        vintage_image = cv2.addWeighted(input_image, 1 - noise_strength, noise, noise_strength, 0)

        # Load and resize overlay image
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        overlay = cv2.resize(overlay, (0, 0), None, *resize_factor)

        # Overlay the images
        result = cvzone.overlayPNG(vintage_image, overlay, [-15, -60])
        result =cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result
        
    except Exception as e:
        print(e)
        return None

def apply_lomo(image):
    lomo_image = image.copy()

    # Increase green channel and decrease blue channel
    lomo_image[:, :, 1] = cv2.addWeighted(lomo_image[:, :, 1], 1.3, 0, 0, 0)  # Increase green
    lomo_image[:, :, 0] = cv2.addWeighted(lomo_image[:, :, 0], 0.9, 0, 0, 0)  # Decrease blue
    
    # Apply vignette
    rows, cols, _ = lomo_image.shape
    kernel_x = cv2.getGaussianKernel(cols, 400)  # Increased kernel size
    kernel_y = cv2.getGaussianKernel(rows, 400)  # Increased kernel size
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    mask = mask.astype(np.uint8)  # Convert mask to uint8
    lomo_image = cv2.addWeighted(lomo_image, 1.3, cv2.merge([mask, mask, mask]), -0.3, 0)

    lomo_image =cv2.cvtColor(lomo_image, cv2.COLOR_BGR2RGB)
    return lomo_image

def apply_dynamic_fisheye(image):
    distortion_factor = 800 # Adjust this value for stronger/weaker distortion
    rows, cols = image.shape[:2]
    center_x, center_y = cols // 2, rows // 2
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    map_x = (2 * (map_x - center_x) / cols).astype(np.float32)
    map_y = (2 * (map_y - center_y) / rows).astype(np.float32)
    
    r = np.sqrt(map_x**2 + map_y**2)
    k1 = distortion_factor * 0.0005
    k2 = distortion_factor * 0.0005
    map_x = (map_x * (1 + k1*r**2 + k2*r**4)).astype(np.float32)
    map_y = (map_y * (1 + k1*r**2 + k2*r**4)).astype(np.float32)
    
    map_x = ((map_x + 1) * cols / 2).astype(np.float32)
    map_y = ((map_y + 1) * rows / 2).astype(np.float32)
    
    distorted_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    #distorted_image =cv2.cvtColor(distorted_image, cv2.COLOR_BGR2RGB)
    return distorted_image

def sketch_filter(input_image):
    try:
        img = cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2BGR)
        output = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        output = cv2.GaussianBlur(output, (3, 3), 0)
        output = cv2.Laplacian(output, -1, ksize=5)
        output = 255 - output
        ret, output = cv2.threshold(output, 150, 255, cv2.THRESH_BINARY)
        
        sketch_image = Image.fromarray(output)
        sketch_image_rgb = cv2.cvtColor(np.array(sketch_image), cv2.COLOR_BGR2RGB)
        
        return sketch_image_rgb
        
    except Exception as e:
        print(e)
        return None

def apply_polaroid_effect(input_image, exposure=90, contrast=0.7):
    try:
        polaroid_image = cv2.convertScaleAbs(input_image, alpha=1, beta=exposure)
        polaroid_image = cv2.convertScaleAbs(polaroid_image, alpha=contrast, beta=0)
        # Assuming path3 is the polaroid overlay
        path3 ='../imagin8/filters/Films/poloroid.png'
        polo = cv2.imread(path3, cv2.IMREAD_UNCHANGED)
        polo = cv2.resize(polo,(0,0),None,1.4,1.2)

        overlay = cvzone.overlayPNG(polaroid_image, polo, [-90, -30])
        overlay=cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        return overlay

    except Exception as e:
        print(e)
        return None

def apply_vintage_grains_effect(input_image, overlay_path='../imagin8/filters/Films/16mm.png', resize_factor=(1.09,1.88), exposure=0.7, noise_strength=0.3):
    try:
        # Generate an orange overlay with the same shape as input_image
        orange_overlay = np.full_like(input_image, [0, 128, 255], dtype=np.uint8)
        
        # Blend the orange overlay with the input image
        vintage_image = cv2.addWeighted(input_image, exposure, orange_overlay, 1 - exposure, 0)

        # Generate random noise
        noise = np.random.normal(0, 1, input_image.shape[:2]).astype('uint8')
        noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        
        # Resize the noise to match the dimensions of input_image
        noise = cv2.resize(noise, (input_image.shape[1], input_image.shape[0]))
        
        # Add the noise to the input image with adjusted exposure
        vintage_image = cv2.addWeighted(vintage_image, 1 - noise_strength, noise, noise_strength, 0)
        
        # Load and resize overlay image
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        overlay = cv2.resize(overlay, (0, 0), None, *resize_factor)
        # Overlay the images
        result = cvzone.overlayPNG(vintage_image, overlay, [-10, 5])
        result=cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result
        
    except Exception as e:
        print(e)
        return None

def apply_vintage_grains_effect2(input_image, overlay_path='../imagin8/filters/Films/new18.png', resize_factor=(0.8, 1.5), exposure=0.7, noise_strength=0.3):
    try:
        # Generate random noise
        noise = np.random.normal(0, 1, input_image.shape[:2]).astype('uint8')
        noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        
        # Resize the noise to match the dimensions of input_image
        noise = cv2.resize(noise, (input_image.shape[1], input_image.shape[0]))
        
        # Add the noise to the input image with adjusted exposure
        vintage_image = cv2.addWeighted(input_image, 1 - noise_strength, noise, noise_strength, 0)

        # Load and resize overlay image
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        overlay = cv2.resize(overlay, (0, 0), None, *resize_factor)

        # Overlay the images
        result = cvzone.overlayPNG(vintage_image, overlay, [-10, 0])
        result=cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result
        
    except Exception as e:
        print(e)
        return None

def apply_vintage_sepia(image):
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_matrix)
    sepia_image=cv2.cvtColor(sepia_image, cv2.COLOR_BGR2RGB)
    return sepia_image

def cartoon_filter(input_image):
    try:
        img = cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 1)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 5, -1000, -1000)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        return cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(e)
        return None

def add_vintage_grains(input_image):
    try:
        # Convert to grayscale for simplicity
        img = cv2.cvtColor(np.asarray(input_image), cv2.COLOR_RGB2GRAY)
        #img = input_image.copy()
        noise = np.random.normal(0, 1, img.shape).astype('uint8')
        vintage_image = cv2.add(img, noise)
        vintage_image=cv2.cvtColor(vintage_image, cv2.COLOR_BGR2RGB)
        return vintage_image
        
    except Exception as e:
        print(e)
        return None

def apply_vintage_sepia1(image):
    sepia_matrix = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_matrix)
    return sepia_image


def open_new_window():
    global root  # Make the root variable global so it can be accessed and destroyed
    
    # Close the main window
    root.withdraw()  # Hide the main window
    
    def back_to_main():
        # Close the current window
        new_window.destroy()
        
        # Restore the main window
        root.deiconify()  # Restore the main window
    
    # Create a new window (Toplevel window)
    new_window = tk.Toplevel(root)
    new_window.title("Upload Image")
    new_window.configure(bg="#6082B6")  # Light gray background color

    # Set the window size and make it fixed with a greater height than width
    new_window_width = 900
    new_window_height = 512
    new_window.geometry(f"{new_window_width}x{new_window_height}")

    # Center the new window on the screen
    screen_width = new_window.winfo_screenwidth()
    screen_height = new_window.winfo_screenheight()
    x_coordinate = (screen_width - new_window_width) // 2
    y_coordinate = (screen_height - new_window_height) // 2
    new_window.geometry(f"+{x_coordinate}+{y_coordinate}")
    new_window.resizable(width=False, height=False)
    font = ("Arial", 12, "bold")

    button_width = 15
    button_height = 1
    
    # Create a Label widget for image display
    image_label = tk.Label(new_window, bg="white", width=512, height=512, padx=20, pady=20)
    image_label.grid(row=0, column=0, rowspan=10, padx=20, pady=20)
    
    global img_display  # Declare img_display as global
    
    def update_image():
        # Load and display the image here
        global img_path,img_resized, img_display  # Access img_display globally
        img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
        img = cv2.imread(img_path)
        
        # Calculate the scaling factors for resizing
        height, width, _ = img.shape
        scale = min(512 / width, 512 / height)
        
        # Use INTER_LINEAR interpolation for resizing
        img_resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Convert image to PhotoImage format
        img_display = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
        
        # Update the label with the new image
        image_label.config(image=img_display)
        image_label.image = img_display  # Keep a reference to prevent garbage collection
    
    def apply_filter(filter_func):
        global img_resized  # Access the global image display variable
        
        # Check if an image is loaded
        if img_resized is not None:
            try:
                # Convert the PhotoImage to a numpy array
                img_np = np.array(img_resized)
                
                # Print the size of the image
                print("Image size:", img_np.shape)
                
                # Convert the numpy array to a format compatible with OpenCV (BGR)
                img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Apply the filter to the image
                filtered_img = filter_func(img_cv2)
                
                if filtered_img is not None:
                    # Convert the filtered image back to PhotoImage format
                    img_display_filtered = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB)))
                    
                    # Update the label with the filtered image
                    image_label.config(image=img_display_filtered)
                    image_label.image = img_display_filtered  # Keep a reference to prevent garbage collection
    
            except Exception as e:
                print(e)


    button1 = tk.Button(new_window, text="Input Image", font=font, width=button_width, height=button_height, bg="#E1C16E",command=update_image)
    button1.grid(row=0, column=3, pady=(10, 5), padx=5)

    button2 = tk.Button(new_window, text="Lomography", font=font, width=button_width, height=button_height, bg="#89CFF0",command=lambda: apply_filter(apply_lomo))
    button2.grid(row=1, column=3, pady=(5,2), padx=20)

    button3 = tk.Button(new_window, text="Kodak", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_filter(apply_vintage_grains_effect3))
    button3.grid(row=2, column=3, pady=(5,2), padx=20)

    button4 = tk.Button(new_window, text="18 mm Retro Film", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_filter(apply_vintage_grains_effect2))
    button4.grid(row=3, column=3, pady=(5,2), padx=20)

    button5 = tk.Button(new_window, text="16 mm Retro Film", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_filter(apply_vintage_grains_effect))
    button5.grid(row=4, column=3, pady=(5,2), padx=20)

    button6 = tk.Button(new_window, text="Poloroid", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_filter(apply_polaroid_effect))
    button6.grid(row=5, column=3, pady=(5,2), padx=20)

    button7 = tk.Button(new_window, text="Cartoon Effect", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_filter(cartoon_filter))
    button7.grid(row=6, column=3, pady=(5,2), padx=20)

    button8 = tk.Button(new_window, text="Sketch Effect", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_filter(sketch_filter))
    button8.grid(row=7, column=3, pady=(5,2), padx=20)

    button9 = tk.Button(new_window, text="Vintage Sepia", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_filter(apply_vintage_sepia))
    button9.grid(row=8, column=3, pady=(5,2), padx=20)


    button10 = tk.Button(new_window, text="Vintage Grains", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_filter(add_vintage_grains))
    button10.grid(row=9, column=3, pady=(5,2), padx=20)

    button11 = tk.Button(new_window, text="Back", font=font, width=button_width, height=button_height, command=back_to_main, relief="sunken", bg="#89CFF0", borderwidth=4)
    button11.grid(row=10, column=3, pady=(5,2), padx=20)

    # Configure row and column weights for centering
    new_window.grid_rowconfigure(0, weight=1)
    new_window.grid_rowconfigure(1, weight=1)
    new_window.grid_rowconfigure(2, weight=1)
    new_window.grid_rowconfigure(3, weight=1)
    new_window.grid_rowconfigure(4, weight=1)
    new_window.grid_rowconfigure(5, weight=1)
    new_window.grid_rowconfigure(6, weight=1)
    new_window.grid_rowconfigure(7, weight=1)
    new_window.grid_rowconfigure(8, weight=1)
    new_window.grid_rowconfigure(9, weight=1)
    new_window.grid_rowconfigure(10, weight=1)

    new_window.grid_columnconfigure(0, weight=1)

    new_window.mainloop()  # Start the main loop for the new window


def open_live_window():
    global root, live_label, processing

    root.withdraw()

    def back_to_main():
        global processing
        processing = False  # Stop the video processing
        new_window.destroy()
        root.deiconify()

    def apply_live(filter_function):
        global live_label, processing

        cap = cv2.VideoCapture(0)

        def process_video():
            while processing:  # Use the flag to control the loop
                ret, frame = cap.read()

                if not ret:
                    break

                # Apply the filter function
                output = filter_function(frame)

                # Convert the filtered frame to PhotoImage format
                img_display = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)))

                # Update the label with the filtered live video
                live_label.config(image=img_display)
                live_label.image = img_display

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        processing = True
        threading.Thread(target=process_video).start()

    def display_normal_video():
        global live_label, processing
        cap = cv2.VideoCapture(0)

        def process_video():
            while processing:  # Use the flag to control the loop
                ret, frame = cap.read()

                if not ret:
                    break
                # Convert the filtered frame to PhotoImage format
                img_display = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

                # Update the label with the filtered live video
                live_label.config(image=img_display)
                live_label.image = img_display

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        processing = True
        threading.Thread(target=process_video).start()


    new_window = tk.Toplevel(root)
    new_window.title("Image Display")
    new_window.configure(bg="#6082B6")

    new_window_width = 900
    new_window_height = 512
    new_window.geometry(f"{new_window_width}x{new_window_height}")

    screen_width = new_window.winfo_screenwidth()
    screen_height = new_window.winfo_screenheight()
    x_coordinate = (screen_width - new_window_width) // 2
    y_coordinate = (screen_height - new_window_height) // 2
    new_window.geometry(f"+{x_coordinate}+{y_coordinate}")
    new_window.resizable(width=False, height=False)
    display_normal_video()
    font = ("Arial", 12, "bold")

    button_width = 20
    button_height = 2

    live_label = tk.Label(new_window, bg="white", width=512, height=512, padx=20, pady=20)
    live_label.grid(row=0, column=0, rowspan=6, padx=20, pady=20)

    button1 = tk.Button(new_window, text="No Filter ", font=font, width=button_width, height=button_height,
                        bg="#E1C16E", command=display_normal_video)
    button1.grid(row=0, column=3, pady=(5, 2), padx=30)


    button7 = tk.Button(new_window, text="Cartoon Effect", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_live(cartoon_filter))
    button7.grid(row=1, column=3, pady=(5,2), padx=10)

    button8 = tk.Button(new_window, text="Sketch Effect", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_live(sketch_filter))
    button8.grid(row=2, column=3, pady=(5,2), padx=10)

    button9 = tk.Button(new_window, text="Vintage Sepia", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_live(apply_vintage_sepia1))
    button9.grid(row=3, column=3, pady=(5,2), padx=10)


    button10 = tk.Button(new_window, text="Vintage Grains", font=font, width=button_width, height=button_height, bg="#89CFF0", command=lambda: apply_live(add_vintage_grains))
    button10.grid(row=4, column=3, pady=(5,2), padx=10)

    button11 = tk.Button(new_window, text="Back", font=font, width=button_width, height=button_height, command=back_to_main, relief="sunken", bg="#89CFF0", borderwidth=4)
    button11.grid(row=5, column=3, pady=(5,2), padx=10)

    # Configure row and column weights for centering
    new_window.grid_rowconfigure(0, weight=1)
    new_window.grid_rowconfigure(1, weight=1)
    new_window.grid_rowconfigure(2, weight=1)
    new_window.grid_rowconfigure(3, weight=1)
    new_window.grid_rowconfigure(4, weight=1)
    new_window.grid_rowconfigure(5, weight=1)

    new_window.grid_columnconfigure(0, weight=1)

    new_window.mainloop()


# Function to open a file dialog and load an image
def load_and_resize():
    global img_path, img_display
    img_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])
    img = cv2.imread(img_path)
    
    # Calculate the scaling factors for resizing
    height, width, _ = img.shape
    scale = min(512 / width, 512 / height)
    
    # Use INTER_LINEAR interpolation for resizing
    img_resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding
    pad_x = (512 - img_resized.shape[1]) // 2
    pad_y = (512 - img_resized.shape[0]) // 2
    
    # Create a black canvas with the target size
    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Paste the resized image onto the canvas
    canvas[pad_y:pad_y+img_resized.shape[0], pad_x:pad_x+img_resized.shape[1]] = img_resized
    
    img_display = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)))
    panel = tk.Label(root, image=img_display)
    panel.grid(row=1, column=1)

# Function to apply a filter to the loaded image
def apply_filter(filter_func):
    pass

def apply_live(filter_function):
    pass

def exit_program():
    root.quit()
    sys.exit()

# Set the window size and make it fixed with a greater height than width
window_width = 400
window_height = 470
root.geometry(f"{window_width}x{window_height}")

# Center the window on the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width - window_width) // 2
y_coordinate = (screen_height - window_height) // 2
root.geometry(f"+{x_coordinate}+{y_coordinate}")
root.resizable(width=False, height=False)
# Create a bold font
font = ("Arial", 12, "bold")

button_width = 15
button_height = 2
button1 = tk.Button(root, text="Upload Image", font=font, width=button_width, height=button_height, bg="#89CFF0",command=open_new_window)
button1.grid(row=1, column=0, pady=(50, 10), padx=20)

button2 = tk.Button(root, text="Live Mode", font=font, width=button_width, height=button_height, bg="#89CFF0",command=open_live_window)
button2.grid(row=2, column=0, pady=10, padx=20)

# Make the "Exit" button sunken and increase its size
button3 = tk.Button(root, text="Exit", font=font, width=button_width, height=button_height, command=exit_program, relief="sunken", bg="#89CFF0", borderwidth=6)
button3.grid(row=3, column=0, pady=(10, 50), padx=20)


photoimage = tk.PhotoImage(file="wow\\nobg.png")

# Resize the image
new_width = 400  # Adjust this value to resize the image
new_height =150  # Adjust this value to resize the image
photoimage = photoimage.subsample(int(photoimage.width() / new_width), int(photoimage.height() / new_height))

# Create a canvas with a blue border
canvas = tk.Canvas(root, bg="#6082B6", width=new_width, height=new_height, highlightthickness=0)
canvas.grid(row=0, column=0, pady=(20, 0), padx=10)

# Display the image on the canvas
canvas.create_image(0, 0, image=photoimage, anchor=tk.NW)
# Configure row and column weights for centering
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)

root.mainloop()


