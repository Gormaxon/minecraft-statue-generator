import requests
import json
import io
import base64
import math
from PIL import Image
from litemapy import Schematic, Region, BlockState
from collections import Counter

# Load fresh block colors
with open('block_colors.json', 'r') as f:
    BLOCK_COLORS = json.load(f)

print(f'Loaded {len(BLOCK_COLORS)} blocks')

# Build skin mapping
SKIN_MAPPING = {}

# Head: 8x8x8 at statue position x=4-11, y=25-32, z=0-7
for y in range(8):
    for x in range(8):
        for z in range(8):
            sx, sy, sz = 4 + x, 25 + y, z
            if z == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (8 + x, 8 + (7 - y))
            elif z == 7:
                SKIN_MAPPING[(sx, sy, sz)] = (24 + (7 - x), 8 + (7 - y))
            elif x == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (0 + z, 8 + (7 - y))
            elif x == 7:
                SKIN_MAPPING[(sx, sy, sz)] = (16 + (7 - z), 8 + (7 - y))
            elif y == 7:
                SKIN_MAPPING[(sx, sy, sz)] = (8 + x, 0 + z)
            elif y == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (16 + x, 0 + z)
            else:
                SKIN_MAPPING[(sx, sy, sz)] = (8 + x, 8 + (7 - y))

# Body: 8x12x4 at x=4-11, y=13-24, z=2-5
for y in range(12):
    for x in range(8):
        for z in range(4):
            sx, sy, sz = 4 + x, 13 + y, 2 + z
            if z == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (20 + x, 20 + (11 - y))
            elif z == 3:
                SKIN_MAPPING[(sx, sy, sz)] = (32 + (7 - x), 20 + (11 - y))
            elif x == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (16 + z, 20 + (11 - y))
            elif x == 7:
                SKIN_MAPPING[(sx, sy, sz)] = (28 + (3 - z), 20 + (11 - y))
            elif y == 11:
                SKIN_MAPPING[(sx, sy, sz)] = (20 + x, 16 + z)
            elif y == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (28 + x, 16 + z)
            else:
                SKIN_MAPPING[(sx, sy, sz)] = (20 + x, 20 + (11 - y))

# Right Arm: 4x12x4 at x=0-3, y=13-24, z=2-5
for y in range(12):
    for x in range(4):
        for z in range(4):
            sx, sy, sz = 0 + x, 13 + y, 2 + z
            if z == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (44 + x, 20 + (11 - y))
            elif z == 3:
                SKIN_MAPPING[(sx, sy, sz)] = (52 + (3 - x), 20 + (11 - y))
            elif x == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (40 + z, 20 + (11 - y))
            elif x == 3:
                SKIN_MAPPING[(sx, sy, sz)] = (48 + (3 - z), 20 + (11 - y))
            elif y == 11:
                SKIN_MAPPING[(sx, sy, sz)] = (44 + x, 16 + z)
            elif y == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (48 + x, 16 + z)
            else:
                SKIN_MAPPING[(sx, sy, sz)] = (44 + x, 20 + (11 - y))

# Left Arm: 4x12x4 at x=12-15, y=13-24, z=2-5 (mirrors right arm for old skins)
for y in range(12):
    for x in range(4):
        for z in range(4):
            sx, sy, sz = 12 + x, 13 + y, 2 + z
            if z == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (36 + x, 52 + (11 - y))
            elif z == 3:
                SKIN_MAPPING[(sx, sy, sz)] = (44 + (3 - x), 52 + (11 - y))
            elif x == 3:
                SKIN_MAPPING[(sx, sy, sz)] = (40 + (3 - z), 52 + (11 - y))
            elif x == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (32 + z, 52 + (11 - y))
            elif y == 11:
                SKIN_MAPPING[(sx, sy, sz)] = (36 + x, 48 + z)
            elif y == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (40 + x, 48 + z)
            else:
                SKIN_MAPPING[(sx, sy, sz)] = (36 + x, 52 + (11 - y))

# Right Leg: 4x12x4 at x=4-7, y=1-12, z=2-5
for y in range(12):
    for x in range(4):
        for z in range(4):
            sx, sy, sz = 4 + x, 1 + y, 2 + z
            if z == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (4 + x, 20 + (11 - y))
            elif z == 3:
                SKIN_MAPPING[(sx, sy, sz)] = (12 + (3 - x), 20 + (11 - y))
            elif x == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (0 + z, 20 + (11 - y))
            elif x == 3:
                SKIN_MAPPING[(sx, sy, sz)] = (8 + (3 - z), 20 + (11 - y))
            elif y == 11:
                SKIN_MAPPING[(sx, sy, sz)] = (4 + x, 16 + z)
            elif y == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (8 + x, 16 + z)
            else:
                SKIN_MAPPING[(sx, sy, sz)] = (4 + x, 20 + (11 - y))

# Left Leg: 4x12x4 at x=8-11, y=1-12, z=2-5 (mirrors right leg for old skins)
for y in range(12):
    for x in range(4):
        for z in range(4):
            sx, sy, sz = 8 + x, 1 + y, 2 + z
            if z == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (20 + x, 52 + (11 - y))
            elif z == 3:
                SKIN_MAPPING[(sx, sy, sz)] = (28 + (3 - x), 52 + (11 - y))
            elif x == 3:
                SKIN_MAPPING[(sx, sy, sz)] = (24 + (3 - z), 52 + (11 - y))
            elif x == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (16 + z, 52 + (11 - y))
            elif y == 11:
                SKIN_MAPPING[(sx, sy, sz)] = (20 + x, 48 + z)
            elif y == 0:
                SKIN_MAPPING[(sx, sy, sz)] = (24 + x, 48 + z)
            else:
                SKIN_MAPPING[(sx, sy, sz)] = (20 + x, 52 + (11 - y))

print(f'Skin mapping has {len(SKIN_MAPPING)} entries')

def color_distance(c1, c2):
    return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

def find_closest_block(r, g, b):
    min_dist = float('inf')
    closest = 'minecraft:white_wool'
    for block_id, data in BLOCK_COLORS.items():
        dist = color_distance((r, g, b), (data['r'], data['g'], data['b']))
        if dist < min_dist:
            min_dist = dist
            closest = block_id
    return closest

def get_skin_pixel(skin_img, u, v):
    width, height = skin_img.size

    # Handle 64x32 skins - mirror left limbs
    if height == 32 and v >= 32:
        # Left arm (32-47, 48-63) -> Right arm (40-55, 16-31)
        if 32 <= u < 48 and 48 <= v < 64:
            local_u = u - 32
            local_v = v - 48
            if local_u < 4:
                u = 48 + (3 - local_u)
            elif local_u < 8:
                u = 44 + (7 - local_u)
            elif local_u < 12:
                u = 40 + (11 - local_u)
            else:
                u = 52 + (15 - local_u)
            v = 16 + local_v
        # Left leg (16-31, 48-63) -> Right leg (0-15, 16-31)
        elif 16 <= u < 32 and 48 <= v < 64:
            local_u = u - 16
            local_v = v - 48
            if local_u < 4:
                u = 8 + (3 - local_u)
            elif local_u < 8:
                u = 4 + (7 - local_u)
            elif local_u < 12:
                u = 0 + (11 - local_u)
            else:
                u = 12 + (15 - local_u)
            v = 16 + local_v

    u = max(0, min(u, width - 1))
    v = max(0, min(v, height - 1))
    return skin_img.getpixel((u, v))

# Get Notch's skin
response = requests.get('https://api.mojang.com/users/profiles/minecraft/Notch')
uuid = response.json()['id']
response = requests.get(f'https://sessionserver.mojang.com/session/minecraft/profile/{uuid}')
props = response.json()['properties']
for p in props:
    if p['name'] == 'textures':
        tex_data = json.loads(base64.b64decode(p['value']))
        skin_url = tex_data['textures']['SKIN']['url']
        break

response = requests.get(skin_url)
skin_img = Image.open(io.BytesIO(response.content)).convert('RGBA')
print(f'Skin size: {skin_img.size}')

# Create schematic
region = Region(0, 0, 0, 16, 33, 8)

blocks_placed = 0
for (sx, sy, sz), (u, v) in SKIN_MAPPING.items():
    r, g, b, a = get_skin_pixel(skin_img, u, v)
    if a < 128:
        continue
    block_id = find_closest_block(r, g, b)
    region[sx, sy, sz] = BlockState(block_id)
    blocks_placed += 1

print(f'Blocks placed: {blocks_placed}')

schematic = Schematic(name='Notch_statue', author='StatueGen', regions={'statue': region})
schematic.save('C:/Users/gorma/AppData/Roaming/.minecraft/schematics/Notch_statue.litematic')
print('Saved to schematics folder!')

# Count blocks
blocks = []
for x in range(16):
    for y in range(33):
        for z in range(8):
            b = region[x, y, z]
            if b and str(b) != 'minecraft:air':
                blocks.append(str(b))
print(f'\nBlocks in region: {len(blocks)}')
print('Top 10 blocks used:')
for b, c in Counter(blocks).most_common(10):
    print(f'  {c}x {b}')
