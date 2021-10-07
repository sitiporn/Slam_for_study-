import cv2
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF

class Display(object):
    def __init__(self,W,H):
       
        pygame.init()
        self.screen = pygame.display.set_mode((W, H), DOUBLEBUF)
        self.surface = pygame.Surface(self.screen.get_size()).convert()
        
        #  sdl2.ext.init()
      #  self.W , self.H = W, H

      #  self.window = sdl2.ext.Window("AIT SLAM", size=(self.W,self.H), position=(-500,-500))
      #  self.window.show()


    def paint(self,img):

        #junk
        for event in pygame.event.get():
            pass
      #  events = sdl2.ext.get_events()

        
       # for event in events:
       #     if event.type == sdl2.SDL_QUIT: 
       #         exit(0)
        
        #bilt
        # RGB, not BGR (might have to switch in twitchslam)
        pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:, :,0:3])
        self.screen.blit(self.surface, (0,0))
        
      #  print("And I'm OK")
      #  print(img.shape)
      #  print(self.window.get_surface())
       # surf = sdl2.ext.pixels3d(self.window.get_surface())
       # 
       # surf[:,:,0:3] = img.swapaxes(0,1)
         
         #bilt
       # self.window.refresh()
        pygame.display.flip()
