import pygame as p
import numpy as np
from neural_network import NeuralNetwork
from utils import load_model
from config import CHOSEN_MODEL_NAME, LAYER_DIMS, SIDEBAR_WIDTH, SCALAR, SQUARE, HEIGHT, WIDTH, COLOUR_BOARD_LIGHT, COLOUR_BOARD_DARK, COLOUR_PEN, COLOUR_BACKGROUND, PREDICTION_TEXT_COLOUR, PREDICTION_TEXT_SIZE, PREDICTION_TEXT_COORDINATES, PREDICTION_TEXT_FONT

class Program:
    def __init__(self, screen):
        self.screen = screen
        self.running = True
        self.input_array = np.zeros((SCALAR, SCALAR), dtype=float) # 0 = background, 1 = drawn
        self.squares = []
        self.last_mouse_pos = None
        self.prediction = ""

        nn = NeuralNetwork(LAYER_DIMS)
        load_model(nn, CHOSEN_MODEL_NAME)
        self.nn = nn

    def handle_events(self):
        mouse_held = p.mouse.get_pressed()[0] # true if LMB held down
        mouse_pos = p.mouse.get_pos()

        for e in p.event.get():
            if e.type == p.QUIT:
                self.running = False

            if e.type == p.KEYDOWN and e.key == p.K_SPACE:
                self.predict()

            if mouse_held and self.in_bounds(mouse_pos):
                if self.last_mouse_pos is not None:
                    self.draw_line(self.last_mouse_pos, mouse_pos)
                else:
                    self.handle_click(mouse_pos)
                self.last_mouse_pos = mouse_pos
            else:
                self.last_mouse_pos = None
    
    def draw_line(self, start, end):
        # interpolate between two points and draw all intermediate squares
        x0, y0 = start[0] // SQUARE, start[1] // SQUARE
        x1, y1 = end[0] // SQUARE, end[1] // SQUARE
        
        # calculate maximum cardinal distance (aka the chebyshev distance) to find the maximum of the two distances
        steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
        for i in range(steps + 1):
            # apply linear interpolation using formula
            t = i / max(steps, 1)
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)

            # check if coordinates are within bounds
            if 0 <= x < SCALAR and 0 <= y < SCALAR:
                self.squares.append((x, y))

    def in_bounds(self, pos):
        return (pos[0] <= WIDTH - SIDEBAR_WIDTH)

    def handle_click(self, pos):
        new_x = pos[0] // SQUARE
        new_y = pos[1] // SQUARE
        self.squares.append((new_x, new_y))

    def render(self):
        # render background
        screen.fill(COLOUR_BACKGROUND)

        # render board
        for x in range(SCALAR):
            for y in range(SCALAR):
                colour = COLOUR_BOARD_LIGHT if (x + y) % 2 == 0 else COLOUR_BOARD_DARK # just for checking the board dimensions
                p.draw.rect(self.screen, colour, (x * SQUARE, y * SQUARE, SQUARE, SQUARE))
        
        # render drawn squares
        for s in self.squares:
            p.draw.rect(self.screen, COLOUR_PEN, (s[0] * SQUARE, s[1] * SQUARE, SQUARE, SQUARE))
            self.input_array[s[1], s[0]] = 1.0 # register that the square has been drawn onto the numpy array

        # draw prediction
        font = p.font.SysFont(PREDICTION_TEXT_FONT, PREDICTION_TEXT_SIZE)
        screen.blit(font.render(self.prediction, True, PREDICTION_TEXT_COLOUR), (PREDICTION_TEXT_COORDINATES))

        p.display.flip()
    
    def predict(self):
        input_processed = self.input_array.astype(np.float32)
        input_processed = input_processed.flatten().reshape(1, -1)

        output = self.nn.forward(input_processed)
        self.prediction = str(np.argmax(output[0]))

        # command line feedback
        # print(f"Predicted digit: {self.prediction}")
        
        # clear for next draw
        self.input_array = np.zeros((SCALAR, SCALAR), dtype=float)
        self.squares = []

p.init()
screen = p.display.set_mode((WIDTH, HEIGHT))
app = Program(screen)
while app.running:
    app.handle_events()
    app.render()
p.quit()