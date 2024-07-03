import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, messagebox

class LabelDetectionMachine:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Label Detection Machine")

        # Create a label to display the result
        self.result_label = Label(self.root, text="Status: ", font=('Helvetica', 18))
        self.result_label.pack()

        # Start detection button
        self.start_button = tk.Button(self.root, text="Start Detection", command=self.start_detection)
        self.start_button.pack(side=tk.LEFT, padx=10, pady=10)

    def update_result_label(self, text):
        self.result_label.config(text=f"Status: {text}")
        self.root.update_idletasks()
        self.root.update()

    def detect_label(self):
        cap = cv2.VideoCapture(0)
        print("Camera started. Detecting labels. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected_label = False

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Adjust area threshold as needed
                    # Approximate the contour
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                    
                    if len(approx) >= 4 and len(approx) <= 10:
                        # Assuming the label contour would have 4 to 10 vertices
                        detected_label = True
                        break

            if detected_label:
                self.update_result_label("Label Detected")
                print("Label Detected")
            else:
                self.update_result_label("No Label Detected")
                print("No Label Detected")

            # Display the resulting frame
            cv2.imshow('Label Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def start_detection(self):
        self.detect_label()

    def run(self):
        self.root.mainloop()

def main():
    machine = LabelDetectionMachine()
    machine.run()

if __name__ == "__main__":
    main()
