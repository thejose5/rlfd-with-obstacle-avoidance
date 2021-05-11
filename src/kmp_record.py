import numpy as np
import pygame
import time
import csv
import random
import os


WIDTH = 2000
HEIGHT = 1500
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0,0,0)
GREY = (200,200,200)
GREEN = (0,255,0)
COLORS = [RED,GREEN,BLUE,BLACK]
DEFAULT_FONT = 'freesansbold.ttf'
REC_STATUS = 'Idle'
GRIPPER_CLOSED_LINE_COLOR = RED
GRIPPER_OPEN_LINE_COLOR = BLACK
LINE_COLOR = BLACK
REC_TRAJECTORY = []
SAVE_LOCATION = '/home/thejus/catkin_ws/src/movement_primitives/training_data/KMP_4pts_largerboundaries/'
SAVE_FREQ = 10  # The mouse position will be saved after every SAVE_FREQ iterations of the while loop
GRIPPER_STATUS = 0  # 0 = Open, 1 = Closed
LEFT_POINT_BOUNDS = [[10,990],[400,1490]] #x1,x2; y1,y2  [[300,700],[875,1200]]
RIGHT_POINT_BOUNDS = [[1010,1990],[400,1490]]  # [[1300,1700],[875,1200]]
POINTS = []
DEMO_dT = 0.01
# status_surface = pygame.Surface()

class Button:

    def __init__(self,text,loc_x,loc_y,game_display, color=GREY, size=30):
        self.text = text
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.color = color
        textfont = pygame.font.Font(DEFAULT_FONT, size)
        textsurf = textfont.render(self.text, True, BLACK, color) #, background=GREY
        self.button_rect = textsurf.get_rect()
        self.button_rect.center = (loc_x,loc_y)
        game_display.blit(textsurf, self.button_rect)
        pygame.display.update()


def displayText(text, loc_x, loc_y, game_display, size=30, bgcolor = None, color=BLACK):
    textfont = pygame.font.Font(DEFAULT_FONT,size)
    TextSurf = textfont.render(text, True, color, bgcolor)
    TextRect = TextSurf.get_rect()
    TextRect.center = (loc_x,loc_y)
    game_display.blit(TextSurf, TextRect)
    pygame.display.update()
    return TextSurf

def updateStatus(rec_status, game_display):
    global REC_STATUS, status_surface
    displayText('Status: ' + REC_STATUS, WIDTH / 1.2, HEIGHT / 15, game_display, bgcolor=WHITE, color=WHITE)
    REC_STATUS = rec_status
    displayText('Status: ' + REC_STATUS, WIDTH / 1.2, HEIGHT / 15, game_display, bgcolor=WHITE)

def startRecording(refresh_button, stop_button, game_display):
    updateStatus('Recording',game_display)
    global LINE_COLOR, GRIPPER_STATUS
    i=0
    time_ind = 0
    while True:
        # print(pygame.mouse.get_pos())
        pygame.draw.circle(game_display, LINE_COLOR, pygame.mouse.get_pos(), 5)
        if i%SAVE_FREQ == 1:
            REC_TRAJECTORY.append((time_ind*DEMO_dT,) + tuple(float(x)/100 for x in pygame.mouse.get_pos()) + tuple([float(x)/(100*DEMO_dT) for x in pygame.mouse.get_rel()]))
            time_ind += 1
        i+=1
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if refresh_button.button_rect.collidepoint(pygame.mouse.get_pos()):
                    # pygame.quit()
                    initializeDisplay()
                if stop_button.button_rect.collidepoint(pygame.mouse.get_pos()):
                    stopRecording(game_display)
                    return
                elif pygame.mouse.get_pressed()[0]:
                    GRIPPER_STATUS = 1
                    LINE_COLOR=GRIPPER_CLOSED_LINE_COLOR
                elif pygame.mouse.get_pressed()[2]:
                    GRIPPER_STATUS = 0
                    LINE_COLOR=GRIPPER_OPEN_LINE_COLOR

                # Following lines are for stop recording by pressing anywhere on screen
                # else:
                #     stopRecording(game_display)
                #     return



def stopRecording(game_display):
    updateStatus('Recording Complete',game_display)

def generatePointPos(npoints):
    points = []

    # Static
    # static_point_set = [(500,900),(1500,750),(1500,1200),(900,1100)]
    # for i in range(npoints):
    #     points.append(static_point_set[i])

    # Dynamic
    for i in range(npoints):
        if i%2 == 0: #Left side
            points.append((random.randint(LEFT_POINT_BOUNDS[0][0],LEFT_POINT_BOUNDS[0][1]),random.randint(LEFT_POINT_BOUNDS[1][0],LEFT_POINT_BOUNDS[1][1])))
        else: #Right side
            points.append((random.randint(RIGHT_POINT_BOUNDS[0][0],RIGHT_POINT_BOUNDS[0][1]),random.randint(RIGHT_POINT_BOUNDS[1][0],RIGHT_POINT_BOUNDS[1][1])))

    return points

def createPoints(npoints, game_display):
    # print(REC_STATUS)
    positions = generatePointPos(npoints)
    for i in range(npoints):
        pygame.draw.circle(game_display,COLORS[i],positions[i],10)
        POINTS.append(positions[i])
    updateStatus('Ready to Record', game_display)

def saveTrajectory(game_display):
    # Finding the index of last file saved
    listdir = os.listdir(SAVE_LOCATION)
    listdir = list(map(int, listdir))
    listdir.sort()
    # print(listdir)
    ind_last_file = 0 if len(listdir)==0 else int(listdir[-1])
    # print(ind_last_file)
    del(REC_TRAJECTORY[0])
    with open(SAVE_LOCATION+str(ind_last_file+1),'w') as f:
        write = csv.writer(f)
        write.writerows(REC_TRAJECTORY)
    updateStatus('Trajectory Saved.. Refreshing', game_display)
    time.sleep(2)
    initializeDisplay()

def startLoop(game_display, refresh_button, start_button, stop_button, save_button):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if refresh_button.button_rect.collidepoint(pygame.mouse.get_pos()):
                    # pygame.quit()
                    initializeDisplay()

                if REC_STATUS == 'Recording':
                    startRecording(stop_button)

                elif REC_STATUS == 'Ready to Record':
                    if start_button.button_rect.collidepoint(pygame.mouse.get_pos()):
                        startRecording(refresh_button, stop_button, game_display)

                    # Following lines are for start recording by pressing anywhere on screen
                    # startRecording(refresh_button, stop_button, game_display)

                elif REC_STATUS == 'Recording Complete':
                    if save_button.button_rect.collidepoint(pygame.mouse.get_pos()):
                        saveTrajectory(game_display)

def initializeDisplay():
    global REC_TRAJECTORY, GRIPPER_STATUS, LINE_COLOR
    REC_TRAJECTORY = []
    GRIPPER_STATUS = 0
    LINE_COLOR = BLACK

    pygame.init()
    game_display = pygame.display.set_mode((WIDTH,HEIGHT))
    pygame.display.set_caption('Trajectory Recorder')
    game_display.fill(WHITE)
    pygame.display.update()
    displayText('Instructions:', WIDTH / 2, HEIGHT / 15, game_display)
    displayText('1. Left click corresponds to "Pick"', WIDTH / 2, HEIGHT / 10, game_display)
    displayText('2. Right click corresponds to "Place"', WIDTH / 2, HEIGHT / 8, game_display)
    displayText('3. After selecting no. of points, Click "Start" to start recording', WIDTH / 2, HEIGHT / 6.6, game_display)
    displayText('4. Click "Stop" to stop recording', WIDTH / 2, HEIGHT / 5.7, game_display)
    displayText('5. Click "Save" to save recording', WIDTH / 2, HEIGHT / 5, game_display)
    updateStatus('Ready to Record', game_display)
    start_button = Button('START', WIDTH / 2.2, HEIGHT / 4, game_display, color=GREEN)
    stop_button = Button('STOP', WIDTH / 1.9, HEIGHT / 4, game_display, color=RED)
    save_button = Button('Save Recording', WIDTH/1.2, HEIGHT/10, game_display, color=GREEN)
    refresh_button = Button('REFRESH', WIDTH / 1.2, HEIGHT / 5, game_display, color=BLUE)
    createPoints(2, game_display)

    pygame.draw.rect(game_display, BLUE, pygame.Rect(LEFT_POINT_BOUNDS[0][0], LEFT_POINT_BOUNDS[1][0],
                                                     LEFT_POINT_BOUNDS[0][1]-LEFT_POINT_BOUNDS[0][0],
                                                     LEFT_POINT_BOUNDS[1][1]-LEFT_POINT_BOUNDS[1][0]), 2)
    pygame.draw.rect(game_display, BLUE, pygame.Rect(RIGHT_POINT_BOUNDS[0][0], RIGHT_POINT_BOUNDS[1][0],
                                                     RIGHT_POINT_BOUNDS[0][1] - RIGHT_POINT_BOUNDS[0][0],
                                                     RIGHT_POINT_BOUNDS[1][1] - RIGHT_POINT_BOUNDS[1][0]), 2)
    pygame.display.update()

    startLoop(game_display, refresh_button, start_button, stop_button, save_button)



if __name__ == '__main__':
    initializeDisplay()
