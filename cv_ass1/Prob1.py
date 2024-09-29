# Import opencv library for image operation
import cv2

# Import numpy library for matrix operations
import numpy as np

# Import math library for sine and cosine functions
import math

# Import random library for random selection of colors, noise ...
import random

# Import time library for timer (e.g for interactive part)
import time

# Define Radius
radius = 112

# Define Center
center = (113, 113)

# Define Color black for hands of the clock
black = (0, 0, 0)


# Function to obtain a random color for the foreground (clock) and background
def get_random_colors():
    # Function to obtain random color
    def random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Obtain random color for the clock
    foreground_color = random_color()

    # Obtain random color for the background
    background_color = random_color()

    # Function to compute the intensities
    """
	This function is required to ensure the colors of the clock and the background
	are always different.
	If we just use a conditional and == to check it, we can obtain very 
	similar looking colors, for example (255,255,255) and (255,254,255) 
	would be regarded as different but they are actually so similar 
	they will look the same.
	"""

    def calculate_intensity(color):
        # Compute intensity
        return 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]

    """
	The color for the background will be compared to the one of the clock
	based on their intensities and if required a new color at random will be generated 
    until they have different intensities.	
	Here 30 is a treshold, somewhat arbitrarily chosen based on visualization
	of the colors.
	"""
    while abs(calculate_intensity(foreground_color) - calculate_intensity(background_color)) < 30:
        background_color = random_color()

    # Then the color for clock and background are returned
    return foreground_color, background_color


# Function to add gaussian noise to the image
def add_gaussian_noise(image, mean=0, sigma=1):
    # Obtain noise using the normal distributin based on mean and standard deviation
    noise = np.random.normal(mean, sigma, image.shape).astype("uint8")

    # Add the noise to the image
    noisy_image = cv2.add(image, noise, dtype=cv2.CV_8U)

    # Return the image with noise
    return noisy_image


# Function to obtain the coordinates for the text of the hour
def get_hour_coordinates():
    # Initialize array to store hour text coordinate
    h_coord = []

    # Loop for every 30 angles to obtain 12 coordinates
    for i in range(0, 360, 30):
        # Obtain x coordinate
        x_coord = int(center[0] + (radius - 12) * math.cos(i * math.pi / 180))

        # Obtain y coordinate
        y_coord = int(center[1] + (radius - 12) * math.sin(i * math.pi / 180))

        # Append both coordinates together in the array
        h_coord.append((x_coord, y_coord))

    # Return the array of the coordiantes
    return h_coord


# Function to set hands and hour text
def set_hands_and_text(image, hours_coord, hour, minute):
    # Define minute hand length to 90 pixels
    minute_hand_length = 90

    # Define hour hand length to 45 pixels
    hour_hand_length = 45

    # Obtain minute angle
    minute_angle = math.fmod(minute * 6 + 270, 360)

    # Obtain hour angle
    hour_angle = math.fmod((hour * 30) + (minute / 2) + 270, 360)

    # Obtain x coordinate of minute
    minute_x = int(center[0] + minute_hand_length * math.cos(minute_angle * math.pi / 180))

    # Obtain y coordinate of minute
    minute_y = int(center[1] + minute_hand_length * math.sin(minute_angle * math.pi / 180))

    # Set minute hand length to black and the thickness to 3
    cv2.line(image, center, (minute_x, minute_y), black, 3)

    # Obtain x coordinate of hour
    hour_x = int(center[0] + hour_hand_length * math.cos(hour_angle * math.pi / 180))

    # Obtain y coordinate of hour
    hour_y = int(center[1] + hour_hand_length * math.sin(hour_angle * math.pi / 180))

    # Set hour hand length to black and the thickness to 5
    cv2.line(image, center, (hour_x, hour_y), black, 5)

    # Set font to FONT_HERSHEY_SIMPLEX
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Set font scale
    font_scale = 0.7

    # Set font thickness
    font_thickness = 2

    # Set h to 2 to put text on image from calculated coordinates for hour text
    h = 2

    # Loop over each pair of coordinates for hour text
    for i in range(len(hours_coord)):
        # Add one to h
        h += 1

        # If the h (the hour counter) is greater than 12 divide h by 12 and set h to the remainder
        if h > 12:
            h = h % 12

        # Convert h to text for the hour
        text = str(h)

        # Obtain text size
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # Calculate the position for centering the text: obtain x coordinate
        text_x = hours_coord[i][0] - (text_size[0] // 2)

        # Calculate the position for centering the text: obtain y coordinate
        text_y = hours_coord[i][1] + (text_size[1] // 2)

        # Put text in the image
        cv2.putText(image, str(h), (text_x, text_y), font, font_scale, black, font_thickness, cv2.LINE_AA)

    # Return the image
    return image


# Function to create the clock given the hour and minute
def create_clock(hour, minute, interactive=False):
    # Get a clock color and a background color
    foreground_color, background_color = get_random_colors()

    # Create full image base first
    image = np.zeros((227, 227, 3), dtype=np.uint8)

    # Set background color
    image[:] = background_color

    # Create circle for the clock
    cv2.circle(image, center, radius, foreground_color, -1)

    # Choose whether the clock will have noise, at random
    sigma_image = random.choice([0, 1])

    # Add gaussian noise to the image
    image = add_gaussian_noise(image, sigma=sigma_image)

    # Get coordinates of hour for text
    hours_coord = get_hour_coordinates()

    # Set the clock hands and text
    clock_image = set_hands_and_text(image, hours_coord, hour, minute)

    # If interactive is set to True display clock on screen
    if interactive == True:
        print("You can close by pressing ENTER or waiting 10 seconds")

        start_time = time.time()

        while True:
            cv2.imshow("clock", clock_image)
            if cv2.waitKey(1) & 0xFF == 13 or time.time() - start_time > 10:
                break

        cv2.destroyAllWindows()

    return clock_image


# Function to get hour input (for interactive only)
def get_hour():
    while True:
        try:
            hour = int(input("Enter the hour (1-12, or 0 for 12): "))
            if 0 <= hour <= 12:
                return hour
            else:
                print("Invalid input. Please enter a number between 1 and 12.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 12.")


# Function to get minute input (for interactive only)
def get_minute():
    while True:
        try:
            minute = int(input("Enter the minute (0-59): "))
            if 0 <= minute <= 59:
                return minute
            else:
                print("Invalid input. Please enter a number between 0 and 59.")
        except ValueError:
            print("Invalid input. Please enter a number between 0 and 59.")


# Problem 1 function to obtain clock image
def prob1(hour, minute, interactive=False):
    # If interactive set to True get hour and minute and display clock
    if interactive == True:
        hour = get_hour()
        minute = get_minute()
        print(f"You entered: {hour} hour(s) and {minute} minute(s).")
        clock_image = create_clock(hour, minute, interactive=True)

    # If only minute and hour are provided return clock without displaying it
    else:
        clock_image = create_clock(hour, minute)

    # Return clock
    return clock_image


### Example for non interactive mode

# hour = random.choice(np.arange(0, 12))
# minute = random.choice(np.arange(0, 60))
# clock_image = prob1(hour, minute)
# print(hour, minute)

# start_time = time.time()

# # Display if required
# while True:
#     cv2.imshow("clock", clock_image)
#     if cv2.waitKey(1) & 0xFF == 13 or time.time() - start_time > 10:
#         break

# cv2.destroyAllWindows()

### Example for interactive mode

# hour = random.choice(np.arange(0, 12))
# minute = random.choice(np.arange(0, 60))
# clock_image = prob1(hour, minute, interactive=True)
