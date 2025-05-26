import pygame
import sys
import numpy as np
from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window

class Flappy:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )

    def reset(self):
        """Resets the game and returns the initial state"""
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)
        self.player.set_mode(PlayerMode.NORMAL)
        return self.get_state()

    def step(self, action):
        """
        Performs a single step in the game based on the action:
        0 = do nothing, 1 = flap
        Returns: next_state, reward, done
        """
        if action == 1:
            self.player.flap()

        self.background.tick()
        self.floor.tick()
        self.pipes.tick()
        self.score.tick()
        self.player.tick()

        reward = 0.1  # small reward for surviving
        done = self.player.collided(self.pipes, self.floor)
        if done:
            reward = -1.0
            return self.get_state(), reward, done

        for pipe in self.pipes.upper:
            if self.player.crossed(pipe):
                self.score.add()
                reward = 1.0  # reward for passing pipe

        return self.get_state(), reward, done

    def get_state(self):
        """Returns a simplified game state"""
        bird_y = self.player.y
        bird_vel = self.player.vel_y

        pipe = self.get_next_pipe()
        pipe_dist_x = pipe["x"] - self.player.x
        pipe_gap_y = (pipe["top_y"] + pipe["bottom_y"]) / 2

        return np.array([bird_y, bird_vel, pipe_dist_x, pipe_gap_y], dtype=np.float32)

    def get_next_pipe(self):
        """Returns the closest upcoming pipe"""
        for top_pipe in self.pipes.upper:
            if top_pipe.x + top_pipe.w > self.player.x:
                bottom_pipe = self.pipes.lower[self.pipes.upper.index(top_pipe)]
                return {
                    "x": top_pipe.x,
                    "top_y": top_pipe.y,
                    "bottom_y": bottom_pipe.y,
                }
        return {"x": 300, "top_y": 200, "bottom_y": 300}  # fallback

    # ----------------- RL Integration Methods -----------------
    
    def draw(self, screen):
        """Draw the game state to the screen for visualization (useful for monitoring training)"""
        self.background.tick()
        self.floor.tick()
        self.pipes.tick()
        self.score.tick()
        self.player.tick()
        pygame.display.update()

    def is_done(self):
        """Check if the game is over (i.e., if the player has collided)"""
        return self.player.collided(self.pipes, self.floor)

