from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import os
import io
import math
import base64
from PIL import Image
import nbtlib
from nbtlib import Int, Long, String, List, Compound
from litemapy.storage import LitematicaBitArray

# Serve React static files
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)


@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

# Load block colors database
with open(os.path.join(os.path.dirname(__file__), 'block_colors.json'), 'r') as f:
    BLOCK_COLORS = json.load(f)

# Variance weight for hybrid scoring
VARIANCE_WEIGHT = 15


def rgb_to_lab(r, g, b):
    """Convert RGB to LAB color space for perceptual color matching."""
    r, g, b = r/255.0, g/255.0, b/255.0
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
    r, g, b = r * 100, g * 100, b * 100
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y_val = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    x, y_val, z = x / 95.047, y_val / 100.0, z / 108.883
    x = x ** (1/3) if x > 0.008856 else (7.787 * x) + (16/116)
    y_val = y_val ** (1/3) if y_val > 0.008856 else (7.787 * y_val) + (16/116)
    z = z ** (1/3) if z > 0.008856 else (7.787 * z) + (16/116)
    return (116 * y_val) - 16, 500 * (x - y_val), 200 * (y_val - z)


def find_closest_block(r, g, b):
    """Find the Minecraft block with the closest perceptual color using hybrid scoring."""
    pixel_lab = rgb_to_lab(r, g, b)
    min_score = float('inf')
    closest = 'minecraft:white_wool'

    for block_id, data in BLOCK_COLORS.items():
        block_lab = (data['L'], data['a'], data['b_lab'])
        color_dist = math.sqrt(
            (pixel_lab[0]-block_lab[0])**2 +
            (pixel_lab[1]-block_lab[1])**2 +
            (pixel_lab[2]-block_lab[2])**2
        )

        # Hybrid score: color distance + variance penalty
        variance = data.get('variance', 0.5)
        score = color_dist + (variance * VARIANCE_WEIGHT)

        if score < min_score:
            min_score = score
            closest = block_id

    return closest


def get_uuid_from_username(username):
    """Get Minecraft UUID from username using Mojang API."""
    response = requests.get(f'https://api.mojang.com/users/profiles/minecraft/{username}')
    if response.status_code == 200:
        return response.json()['id']
    return None


def get_skin_url(uuid):
    """Get skin URL from UUID using Mojang session API."""
    response = requests.get(f'https://sessionserver.mojang.com/session/minecraft/profile/{uuid}')
    if response.status_code == 200:
        data = response.json()
        properties = data.get('properties', [])
        for prop in properties:
            if prop['name'] == 'textures':
                decoded = base64.b64decode(prop['value'])
                texture_data = json.loads(decoded)
                return texture_data['textures']['SKIN']['url']
    return None


def download_skin(url):
    """Download skin image from URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content)).convert('RGBA')
    return None


def get_skin_pixel(skin_img, u, v):
    """Get pixel color from skin, handling old format skins."""
    w, h = skin_img.size

    # Handle 64x32 (old format) skins - mirror left limbs from right limbs
    if h == 32 and v >= 32:
        # Left arm (32-47, 48-63) -> Right arm
        if 32 <= u < 48 and 48 <= v < 64:
            lu, lv = u - 32, v - 48
            if lu < 4: u = 48 + (3 - lu)
            elif lu < 8: u = 44 + (7 - lu)
            elif lu < 12: u = 40 + (11 - lu)
            else: u = 52 + (15 - lu)
            v = 16 + lv
        # Left leg (16-31, 48-63) -> Right leg
        elif 16 <= u < 32 and 48 <= v < 64:
            lu, lv = u - 16, v - 48
            if lu < 4: u = 8 + (3 - lu)
            elif lu < 8: u = 4 + (7 - lu)
            elif lu < 12: u = 0 + (11 - lu)
            else: u = 12 + (15 - lu)
            v = 16 + lv

    u = max(0, min(u, w-1))
    v = max(0, min(v, h-1))
    return skin_img.getpixel((u, v))


def create_statue_schematic(skin_img, username):
    """Create a Litematica schematic of the player statue."""
    width, height, length = 16, 33, 8
    total_volume = width * height * length
    blocks_array = [0] * total_volume

    palette = ['minecraft:air']
    block_to_idx = {'minecraft:air': 0}

    def get_or_add_palette(block_name):
        if block_name not in block_to_idx:
            block_to_idx[block_name] = len(palette)
            palette.append(block_name)
        return block_to_idx[block_name]

    def set_block(x, y, z, block_name):
        idx = (y * length + z) * width + x
        blocks_array[idx] = get_or_add_palette(block_name)

    def get_face_pixel(skin_img, part, sx, sy, sz, part_width, part_height, part_depth):
        """Get pixel from correct face based on block position within body part."""
        # Determine which face this block is on (prioritize front/back, then sides)
        # UV layouts for each body part (u_start, v_start for each face)
        uv_map = {
            'head': {
                'front': (8, 8),      # 8x8 face
                'back': (24, 8),
                'right': (0, 8),
                'left': (16, 8),
                'top': (8, 0),
                'bottom': (16, 0)
            },
            'body': {
                'front': (20, 20),    # 8x12 front/back, 4x12 sides
                'back': (32, 20),
                'right': (16, 20),
                'left': (28, 20),
                'top': (20, 16),
                'bottom': (28, 16)
            },
            'right_arm': {
                'front': (44, 20),    # 4x12 front/back, 4x12 sides
                'back': (52, 20),
                'right': (40, 20),    # outer
                'left': (48, 20),     # inner
                'top': (44, 16),
                'bottom': (48, 16)
            },
            'left_arm': {
                'front': (36, 52),
                'back': (44, 52),
                'right': (32, 52),    # inner
                'left': (40, 52),     # outer
                'top': (36, 48),
                'bottom': (40, 48)
            },
            'right_leg': {
                'front': (4, 20),     # 4x12 front/back, 4x12 sides
                'back': (12, 20),
                'right': (0, 20),     # outer
                'left': (8, 20),      # inner
                'top': (4, 16),
                'bottom': (8, 16)
            },
            'left_leg': {
                'front': (20, 52),
                'back': (28, 52),
                'right': (16, 52),    # inner
                'left': (24, 52),     # outer
                'top': (20, 48),
                'bottom': (24, 48)
            }
        }

        uv = uv_map[part]

        # Determine which face based on position
        # All faces render fully - top/bottom first so head/arm tops are complete

        # Top face: y at max (full top surface)
        if sy == part_height - 1:
            u, v = uv['top']
            return get_skin_pixel(skin_img, u + sx, v + sz)
        # Bottom face: y at min (full bottom surface)
        elif sy == 0:
            u, v = uv['bottom']
            return get_skin_pixel(skin_img, u + sx, v + (part_depth - 1 - sz))
        # Front face: z at max
        elif sz == part_depth - 1:
            u, v = uv['front']
            return get_skin_pixel(skin_img, u + sx, v + (part_height - 1 - sy))
        # Back face: z at min
        elif sz == 0:
            u, v = uv['back']
            # Back face is mirrored horizontally
            return get_skin_pixel(skin_img, u + (part_width - 1 - sx), v + (part_height - 1 - sy))
        # Right face: x at min
        elif sx == 0:
            u, v = uv['right']
            return get_skin_pixel(skin_img, u + (part_depth - 1 - sz), v + (part_height - 1 - sy))
        # Left face: x at max
        elif sx == part_width - 1:
            u, v = uv['left']
            return get_skin_pixel(skin_img, u + sz, v + (part_height - 1 - sy))
        # Interior block - use front face as default
        else:
            u, v = uv['front']
            return get_skin_pixel(skin_img, u + sx, v + (part_height - 1 - sy))

    # Fill all body parts
    # Head: x=4-11, y=25-32, z=0-7 (8x8x8)
    for sy in range(8):
        for sx in range(8):
            for sz in range(8):
                r, g, b, a = get_face_pixel(skin_img, 'head', sx, sy, sz, 8, 8, 8)
                set_block(4+sx, 25+sy, sz, find_closest_block(r, g, b))

    # Body: x=4-11, y=13-24, z=2-5 (8x12x4)
    for sy in range(12):
        for sx in range(8):
            for sz in range(4):
                r, g, b, a = get_face_pixel(skin_img, 'body', sx, sy, sz, 8, 12, 4)
                set_block(4+sx, 13+sy, 2+sz, find_closest_block(r, g, b))

    # Right Arm: x=0-3, y=13-24, z=2-5 (4x12x4)
    for sy in range(12):
        for sx in range(4):
            for sz in range(4):
                r, g, b, a = get_face_pixel(skin_img, 'right_arm', sx, sy, sz, 4, 12, 4)
                set_block(sx, 13+sy, 2+sz, find_closest_block(r, g, b))

    # Left Arm: x=12-15, y=13-24, z=2-5 (4x12x4)
    for sy in range(12):
        for sx in range(4):
            for sz in range(4):
                r, g, b, a = get_face_pixel(skin_img, 'left_arm', sx, sy, sz, 4, 12, 4)
                set_block(12+sx, 13+sy, 2+sz, find_closest_block(r, g, b))

    # Right Leg: x=4-7, y=1-12, z=2-5 (4x12x4)
    for sy in range(12):
        for sx in range(4):
            for sz in range(4):
                r, g, b, a = get_face_pixel(skin_img, 'right_leg', sx, sy, sz, 4, 12, 4)
                set_block(4+sx, 1+sy, 2+sz, find_closest_block(r, g, b))

    # Left Leg: x=8-11, y=1-12, z=2-5 (4x12x4)
    for sy in range(12):
        for sx in range(4):
            for sz in range(4):
                r, g, b, a = get_face_pixel(skin_img, 'left_leg', sx, sy, sz, 4, 12, 4)
                set_block(8+sx, 1+sy, 2+sz, find_closest_block(r, g, b))

    # Build NBT structure
    nbits = max(2, math.ceil(math.log2(len(palette))))
    bit_array = LitematicaBitArray(total_volume, nbits)
    for idx, val in enumerate(blocks_array):
        bit_array[idx] = val

    palette_nbt = List[Compound]([Compound({'Name': String(b)}) for b in palette])

    region = Compound({
        'Position': Compound({'x': Int(0), 'y': Int(0), 'z': Int(0)}),
        'Size': Compound({'x': Int(width), 'y': Int(height), 'z': Int(length)}),
        'BlockStatePalette': palette_nbt,
        'BlockStates': bit_array._to_nbt_long_array(),
        'Entities': List[Compound]([]),
        'TileEntities': List[Compound]([]),
        'PendingBlockTicks': List[Compound]([]),
        'PendingFluidTicks': List[Compound]([])
    })

    root = Compound({
        'Version': Int(6),
        'SubVersion': Int(1),
        'MinecraftDataVersion': Int(3465),
        'Metadata': Compound({
            'Name': String(f'{username}_statue'),
            'Author': String('Minecraft Statue Generator'),
            'Description': String(f'Statue of {username}'),
            'RegionCount': Int(1),
            'TotalBlocks': Int(1664),
            'TotalVolume': Int(total_volume),
            'TimeCreated': Long(0),
            'TimeModified': Long(0),
            'EnclosingSize': Compound({'x': Int(width), 'y': Int(height), 'z': Int(length)})
        }),
        'Regions': Compound({'statue': region})
    })

    return root


@app.route('/api/generate', methods=['POST'])
def generate_statue():
    """Generate a statue schematic from a Minecraft username."""
    data = request.json
    username = data.get('username', '').strip()

    if not username:
        return jsonify({'error': 'Username is required'}), 400

    # Get UUID
    uuid = get_uuid_from_username(username)
    if not uuid:
        return jsonify({'error': f'Player "{username}" not found'}), 404

    # Get skin URL
    skin_url = get_skin_url(uuid)
    if not skin_url:
        return jsonify({'error': 'Could not retrieve skin'}), 500

    # Download skin
    skin_img = download_skin(skin_url)
    if not skin_img:
        return jsonify({'error': 'Could not download skin'}), 500

    # Create schematic
    root = create_statue_schematic(skin_img, username)

    # Save to bytes
    output = io.BytesIO()
    nbt_file = nbtlib.File(root, gzipped=True)
    nbt_file.save(output)
    output.seek(0)

    return send_file(
        output,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=f'{username}_statue.litematic'
    )


@app.route('/api/preview', methods=['POST'])
def preview_skin():
    """Get skin preview URL for a username."""
    data = request.json
    username = data.get('username', '').strip()

    if not username:
        return jsonify({'error': 'Username is required'}), 400

    uuid = get_uuid_from_username(username)
    if not uuid:
        return jsonify({'error': f'Player "{username}" not found'}), 404

    skin_url = get_skin_url(uuid)
    if not skin_url:
        return jsonify({'error': 'Could not retrieve skin'}), 500

    return jsonify({
        'username': username,
        'uuid': uuid,
        'skin_url': skin_url,
        'render_url': f'https://crafatar.com/renders/body/{uuid}?overlay=true'
    })


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'blocks_loaded': len(BLOCK_COLORS)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
