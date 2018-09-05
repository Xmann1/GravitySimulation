import pygame.gfxdraw
import pygame
import math
import sys

import numpy as np

def draw_circle(s, c, pos, r):
    pygame.gfxdraw.filled_circle(s, pos[0], pos[1], r, c)
    pygame.gfxdraw.aacircle(s, pos[0], pos[1], r, c)


HM_OBJECTS = 455
ALMOST_ZERO = 0.001
UNIT_MULTIPLIER = 128
UNIT_MULT = UNIT_MULTIPLIER
SCREEN_MULTIPLIER = 800
G = 0.05

positions = np.random.random((HM_OBJECTS, 2)) * UNIT_MULT * 2
masses = (np.random.random((HM_OBJECTS)) * UNIT_MULT / 2)
momentum = (np.random.random((HM_OBJECTS, 2)) * 2 - 1) * UNIT_MULT / 15 * masses[:,None]

time_scale = 1

cam_position = np.zeros(2, dtype=np.float32)
cam_zoom = 1
cam_speed = 25

#positions = np.array(((0, 64), (128, 64)), dtype=np.float32)
#masses = np.array((4000, 200))
#momentum = np.array(((16, 0), (-8, 0)), dtype=np.float32)

display = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)

keys = []

while True:
    display.fill((0, 0, 0))

    for ix in range(len(masses)):
        if masses[ix] < ALMOST_ZERO:
            continue
            
        mass = masses[ix]
        position = positions[ix]

        render_position = (position * SCREEN_MULTIPLIER / UNIT_MULTIPLIER)
        render_position = render_position + cam_position
        render_position = render_position * cam_zoom
        render_position = render_position
        render_position += (1920 / 2, 1080 / 2)
        render_position = render_position.astype(np.int32)

        outer_size = np.sqrt(mass) / UNIT_MULTIPLIER / math.pi * SCREEN_MULTIPLIER * cam_zoom
        inner_size = max(outer_size - 2, 1)

        if render_position[0] - outer_size < 0:
            continue
        if render_position[1] - outer_size < 0:
            continue
        if render_position[0] > outer_size + 1920:
            continue
        if render_position[1] > outer_size + 1080:
            continue

        try:
            draw_circle(display, (255, 255, 255), render_position, int(outer_size))
            #draw_circle(display, (0,   0,   0  ), render_position, int(inner_size))
        except OverflowError:
            print("OverFlow encountered, ignoring")

    pygame.display.flip()

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if e.type == pygame.MOUSEBUTTONDOWN:
            if e.button == 4:
                cam_zoom /= 0.9

            if e.button == 5:
                cam_zoom *= 0.9

        if e.type == pygame.KEYDOWN:
            keys.append(pygame.key.name(e.key))

        if e.type == pygame.KEYUP:
            keys.pop(keys.index(pygame.key.name(e.key)))

    delta_cam_pos = np.zeros(2, dtype=np.float32)
    if 'd' in keys:
        delta_cam_pos += (-1, 0)
    if 'a' in keys:
        delta_cam_pos += (1, 0)
    if 'w' in keys:
        delta_cam_pos += (0, 1)
    if 's' in keys:
        delta_cam_pos += (0, -1)

    if 'right' in keys:
        time_scale /= 0.95

    if 'left' in keys:
        time_scale *= 0.95

    cam_position += delta_cam_pos * cam_speed / cam_zoom

    # Physic step:

    # Gravity logic
    for ix in range(len(masses)):
        position_differences = positions - positions[ix]
        distances = np.sum(np.absolute(position_differences), axis=1)
        normals = position_differences / distances[:,None]
        F = (masses[ix] * masses) / (distances ** 2)
        deltas = normals * F[:,None]
        deltas[np.isnan(deltas)] = 0
        deltas[np.isinf(deltas)] = 0
        delta = np.sum(deltas, axis=0) * G
        momentum[ix] += delta * time_scale

    positions = positions + momentum / masses[:,None] * time_scale

    # Collision logic
    for ix in range(len(masses)):
        position = positions[ix]
        mass = masses[ix]
        
        min_ranges = (np.sqrt(masses) / math.pi) + np.sqrt(mass) / math.pi
        ranges = np.sum(np.absolute(positions - position), axis=1)
        collisions = ranges < min_ranges

        for coll_ix in np.nonzero(collisions)[0]:
            if coll_ix == ix:
                continue
            if masses[coll_ix] < ALMOST_ZERO:
                continue

            mass_ratio = masses[ix] / masses[coll_ix]
            
            masses[coll_ix] += masses[ix]
            momentum[coll_ix] += momentum[ix]
            positions[coll_ix] = np.average((positions[coll_ix], positions[ix]), weights=[1, mass_ratio], axis=0)

            masses[ix] = 0
            positions[ix] = (0, 0)
            momentum[ix] = (0, 0)
