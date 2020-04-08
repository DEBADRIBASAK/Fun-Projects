import pygame
import random
import numpy as np
pygame.init()
colors = {'red': (255,0,0),'blue': (0,0,255),'green': (0,255,0),'yellow': (255,255,0),
'black':(0,0,0),'white':(255,255,255)}

dis_width = 750
dis_height = 750

dis = pygame.display.set_mode((dis_width,dis_height))
dis.fill(colors['white'])


class Pieces:
	def __init__(self,x,y,r,clr):
		self.x = x
		self.y = y
		self.r = r
		self.clr = clr
	def show(dis):
		pygame.draw.circle(dis,self.clr,(x,y),r)

class Board:
	def __init__(self,dis):#,dis_width = 750,dis_height = 750):
		self.dis = dis
	def show(self):
		over = False
		while not over:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					over = True
			for i in range(15):
				for j in range(0,15,1):
					pygame.draw.rect(self.dis,colors['black'],[i*50,j*50,50,50],1)
			pygame.draw.rect(self.dis,colors['red'],[25,25,250,250],50)
			pygame.draw.rect(self.dis,colors['white'],[50,50,200,200])

			# for k1 in range(0,4,1):
			# 	for k2 in range(0,4,1):
			# 		Pieces(self.dis,k1*50+25,k*50+25,25).show()
			pygame.draw.rect(self.dis,colors['green'],[25,475,250,250],50)
			pygame.draw.rect(self.dis,colors['white'],[50,500,200,200])
			pygame.draw.rect(self.dis,colors['blue'],[475,25,250,250],50)
			pygame.draw.rect(self.dis,colors['white'],[500,50,200,200])
			pygame.draw.rect(self.dis,colors['yellow'],[475,475,250,250],50)
			pygame.draw.rect(self.dis,colors['white'],[500,500,200,200])
			pygame.draw.rect(self.dis,colors['black'],[300,300,150,150],1)
			pygame.draw.rect(self.dis,colors['white'],[300,300,150,150])


			pygame.draw.polygon(self.dis,colors['red'],[(300,300),(300,450),(375,375)])
			pygame.draw.polygon(self.dis,colors['blue'],[(300,300),(450,300),(375,375)])
			pygame.draw.polygon(self.dis,colors['green'],[(300,450),(450,450),(375,375)])
			pygame.draw.polygon(self.dis,colors['yellow'],[(450,300),(450,450),(375,375)])



			pygame.display.update()

b = Board(dis)
b.show()