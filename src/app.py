import pygame as p
import numpy as np
from neural_network import NeuralNetwork
from utils import load_model
from config import LAYER_DIMS, SIDEBAR_WIDTH, SCALAR, SQUARE, HEIGHT, WIDTH, COLOUR_BOARD, COLOUR_PEN

class Program:
    def __init__(self, screen):
        self.screen = screen
        self.running = True
        self.input_array = np.zeros((SCALAR, SCALAR), dtype=float) # 0 = background, 1 = drawn
        self.squares = []
        
        nn = NeuralNetwork(LAYER_DIMS)
        load_model(nn, "prototype_3.npz")
        self.nn = nn

    def handle_events(self):
        mouse_held = p.mouse.get_pressed()[0] # true if LMB held down
        mouse_pos = p.mouse.get_pos()

        for e in p.event.get():
            if e.type == p.QUIT:
                self.running = False
            
            if mouse_held and self.in_bounds(mouse_pos):
                self.handle_click(mouse_pos)

            if e.type == p.KEYDOWN and e.key == p.K_SPACE:
                self.predict()
    
    def in_bounds(self, pos):
        return (pos[0] <= WIDTH - SIDEBAR_WIDTH)

    def handle_click(self, pos):
        new_x = pos[0] // SQUARE
        new_y = pos[1] // SQUARE
        self.squares.append((new_x, new_y))

    def render(self):
        for x in range(SCALAR):
            for y in range(SCALAR):
                # colour = "#FFFFFF" if (x + y) % 2 == 0 else "#437289" # just for checking the board dimensions
                p.draw.rect(self.screen, COLOUR_BOARD, (x * SQUARE, y * SQUARE, SQUARE, SQUARE))
            
        for s in self.squares:
            p.draw.rect(self.screen, COLOUR_PEN, (s[0] * SQUARE, s[1] * SQUARE, SQUARE, SQUARE))
            self.input_array[s[1], s[0]] = 1.0 # register that the square has been drawn onto the numpy array

        p.display.flip()
    
    def predict(self):
        input_processed = self.input_array.astype(np.float32)
        input_processed = input_processed.flatten().reshape(1, -1)

        output = self.nn.forward(input_processed)
        pred = np.argmax(output[0])

        print(f"Predicted digit: {pred}")
        
        # clear for next draw
        self.input_array = np.zeros((SCALAR, SCALAR), dtype=float)
        self.squares = []

screen = p.display.set_mode((WIDTH, HEIGHT))
app = Program(screen)
while app.running:
    app.handle_events()
    app.render()
p.quit()

# TODO: increase cursor smoothness for more accurate inputs
# TODO: complete sidebar and make GUI look a bit better