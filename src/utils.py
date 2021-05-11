import pygame



BLUE = (0, 0, 255)
RED = (255, 0, 0)
BLACK = (0,0,0)
GREY = (200,200,200)
GREEN = (0,200,0)
WHITE = (255,255,255)

class Map:
    def __init__(self, circles = [[300, 550, 250]], rectangles = [[600, 800, 300, 500]],
                 motion_targets = [[890, 370], [200, 1000], [1300, 1100], [1060, 370]], res=1500):
        self.circles = circles
        self.rectangles = rectangles
        self.motion_targets = motion_targets
        self.res = res

def draw_everything(env_map, pred_path=None, via_pts=None):
    pygame.init()
    screen = pygame.display.set_mode((env_map.res, env_map.res))
    screen.fill(WHITE)
    pygame.display.set_caption('Visualizer')
    # pygame.display.update()

    # Draw Obstacles
    for circle in env_map.circles:
        pygame.draw.circle(screen,RED,(circle[0],circle[1]),circle[2])

    for rectangle in env_map.rectangles:
        pygame.draw.rect(screen, RED, tuple(rectangle))

    # Draw start and end points
    motion_target_colors = [GREEN, BLUE, BLUE, BLACK]
    for i,motion_target in enumerate(env_map.motion_targets):
        pygame.draw.circle(screen, motion_target_colors[i], motion_target, 20)

    # Draw motion planning path
    if via_pts:
        for via_pt in via_pts:
            via_pt = [int(x) for x in via_pt]
            pygame.draw.circle(screen, BLUE, via_pt, 10)

    # Draw path
    if pred_path:
        for i in range(1, len(pred_path)):
            pygame.draw.line(screen, (0, 0, 0), pred_path[i - 1], pred_path[i], 5)

    pygame.display.update()

    while (True):
        for event in pygame.event.get():
            # print(event)
            if event.type == pygame.QUIT:
                pygame.quit()
                return


