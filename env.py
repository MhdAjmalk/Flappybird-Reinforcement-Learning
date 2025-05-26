import numpy as np
from src.flappy import Flappy

class FlappyBirdEnv:
    def __init__(self):
        # Initialize the Flappy class from flappy.py
        self.game = Flappy()

    def reset(self):
        """Reset the environment and return the initial state"""
        return self.game.reset()

    def step(self, action):
        """
        Action: 0 = do nothing, 1 = flap
        Returns: next_state, reward, done
        """
        # Perform the action in the game and get the next state, reward, and done flag
        state, reward, done = self.game.step(action)
        return state, reward, done

    def get_state(self):
        """Get the current state of the environment"""
        # Return the current state
        return self.game.get_state()

    def render(self):
        """Render the current game state to the screen"""
        # Render the game state for visualization
        self.game.draw(self.game.config.screen)

    def is_done(self):
        """Check if the game is over"""
        # Check if the game has ended (i.e., player colliding with pipes or floor)
        return self.game.is_done()

