import pygame
import time
import random
pygame.init()

dis_width = 800
dis_height = 600
snake_block = 10
snake_speed = 10

dis = pygame.display.set_mode((dis_width,dis_height))

pygame.display.set_caption("Snake!!")

blue = (0,0,255)
red = (255,0,0)
green = (0,255,0)
yellow = (255,255,0)
white = (255,255,255)
black = (0,0,0)

font_style = pygame.font.SysFont("bahnschrift",30)
score_font = pygame.font.SysFont("comicsansms",30)
clock = pygame.time.Clock()

def message(msg,color):
	mesg = font_style.render(msg,True,color)
	dis.blit(mesg,[dis_width/3,dis_height/3])

def our_snake(snakelist):
	for x in snakelist:
		pygame.draw.rect(dis,black,[x[0],x[1],snake_block,snake_block])
		
score = 0

def yourScore(score):
	mesg = score_font.render("(Press i to increase speed and d to decrease) Your Score: "+str(score),True,blue)
	dis.blit(mesg,[0,0])

def gameLoop():
	global score
	global snake_speed
	x = dis_width/2
	y = dis_height/2
	x_change = 0
	y_change = 0
	game_over = False
	game_close = False
	snakelist = []
	length = 1
	food_x = round(random.randrange(0,(dis_width-snake_block)/10.0)*10.0)
	food_y = round(random.randrange(0,(dis_height-snake_block)/10.0)*10.0)
	while not game_over:
		while game_close==True:
			dis.fill(white)
			message("You Lost!! Press Q to quit and C to continue",red)
			pygame.display.update()
			for event in pygame.event.get():
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_q:
						game_over = True
						game_close = False
					elif event.key == pygame.K_c:
						gameLoop()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				game_over = True
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP:
					y_change = -10
					x_change = 0
				elif event.key == pygame.K_DOWN:
					y_change = 10
					x_change = 0
				elif event.key == pygame.K_LEFT:
					y_change = 0
					x_change = -10
				elif event.key == pygame.K_RIGHT:
					y_change = 0
					x_change = 10	
				elif event.key == pygame.K_i:
					snake_speed+=5
				elif event.key == pygame.K_d:
					snake_speed-=5
		if(x>dis_width or x<0 or y<0 or y>dis_height):
			#game_close = True
			if(x<0):
				x = dis_width+x
			elif(x>dis_width):
				x = x-dis_width
			elif(y<0):
				y = dis_height+y
			elif(y>dis_height):
				y = y-dis_height
		x+=x_change
		y+=y_change
		dis.fill(white)			
		pygame.draw.rect(dis,green,[food_x,food_y,snake_block,snake_block])
		pygame.display.update()
		snake_head = []
		snake_head.append(x)
		snake_head.append(y)
		snakelist.append(snake_head)
		if len(snakelist)>length:
			del snakelist[0]
		for x1 in snakelist[:-2]:
			if x1==snake_head:
				game_close = True
		our_snake(snakelist)
		yourScore(score)
		pygame.display.update()
		if x==food_x and y==food_y:
			length+=1
			score+=1
			food_x = round(random.randrange(0,(dis_width-snake_block)/10.0)*10.0)
			food_y = round(random.randrange(0,(dis_height-snake_block)/10.0)*10.0)
		clock.tick(snake_speed)
	pygame.quit()
	quit()

gameLoop()