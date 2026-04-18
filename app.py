from flask import Flask, request, send_file, jsonify, send_from_directory
from flask_cors import CORS
import requests
import json
import os
import io
import math
import base64
from itertools import product
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

# Pre-sort blocks by LAB lightness for faster searching
BLOCKS_BY_LIGHTNESS = sorted(BLOCK_COLORS.items(), key=lambda x: x[1]['L'])


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


def get_block_score(block_id, target_lab):
    """Calculate hybrid score for a block against target LAB color."""
    data = BLOCK_COLORS[block_id]
    block_lab = (data['L'], data['a'], data['b_lab'])
    color_dist = math.sqrt(
        (target_lab[0]-block_lab[0])**2 +
        (target_lab[1]-block_lab[1])**2 +
        (target_lab[2]-block_lab[2])**2
    )
    variance = data.get('variance', 0.5)
    return color_dist + (variance * VARIANCE_WEIGHT)


def find_closest_block(r, g, b):
    """Find the Minecraft block with the closest perceptual color using hybrid scoring."""
    pixel_lab = rgb_to_lab(r, g, b)
    min_score = float('inf')
    closest = 'minecraft:white_wool'

    for block_id, data in BLOCK_COLORS.items():
        score = get_block_score(block_id, pixel_lab)
        if score < min_score:
            min_score = score
            closest = block_id

    return closest


def find_best_block_combination(target_rgb, scale):
    """Find the best combination of blocks for a scaled pixel.

    For scale N, we need NxN blocks to represent one pixel.
    Uses fast greedy optimization to find blocks that average to target color.
    """
    num_blocks = scale * scale
    target_lab = rgb_to_lab(*target_rgb)

    if num_blocks == 1:
        return [find_closest_block(*target_rgb)]

    # Get top candidate blocks (closest to target)
    candidates = []
    for block_id, data in BLOCK_COLORS.items():
        score = get_block_score(block_id, target_lab)
        candidates.append((block_id, score, data))

    # Sort by score and take top candidates
    candidates.sort(key=lambda x: x[1])
    top_candidates = candidates[:min(15, len(candidates))]

    # Use greedy approach for all scales (fast)
    result = []
    remaining_target = [target_lab[0] * num_blocks,
                        target_lab[1] * num_blocks,
                        target_lab[2] * num_blocks]

    for i in range(num_blocks):
        blocks_left = num_blocks - i
        needed_avg = (remaining_target[0] / blocks_left,
                      remaining_target[1] / blocks_left,
                      remaining_target[2] / blocks_left)

        # Find block closest to needed average
        best_block = top_candidates[0][0]
        best_score = float('inf')

        for block_id, _, data in top_candidates:
            color_dist = math.sqrt(
                (needed_avg[0] - data['L'])**2 +
                (needed_avg[1] - data['a'])**2 +
                (needed_avg[2] - data['b_lab'])**2
            )
            variance = data.get('variance', 0.5)
            score = color_dist + (variance * VARIANCE_WEIGHT * 0.3)

            if score < best_score:
                best_score = score
                best_block = block_id

        result.append(best_block)
        block_data = BLOCK_COLORS[best_block]
        remaining_target[0] -= block_data['L']
        remaining_target[1] -= block_data['a']
        remaining_target[2] -= block_data['b_lab']

    return result


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


def get_overlay_offset(u, v):
    """Get the overlay layer offset for a given base UV coordinate.

    Returns (u_offset, v_offset) to add to base coords to get overlay coords.
    Returns None if no overlay exists for this region.
    """
    # Head region (v: 0-15)
    if 0 <= v < 16:
        if 0 <= u < 32:  # Head base region
            return (32, 0)  # Head overlay is at u+32

    # Body and arm base region (v: 16-31)
    elif 16 <= v < 32:
        if 16 <= u < 40:  # Body base
            return (0, 16)  # Body overlay is at v+16
        elif 40 <= u < 56:  # Right arm base
            return (0, 16)  # Right arm overlay is at v+16

    # Body and arm overlay region (v: 32-47) - no overlay for overlay
    elif 32 <= v < 48:
        return None

    # Left arm and left leg region (v: 48-63)
    elif 48 <= v < 64:
        if 16 <= u < 32:  # Left leg base
            return (-16, 0)  # Left leg overlay is at u-16 (0-15, 48-63)
        elif 32 <= u < 48:  # Left arm base
            return (16, 0)  # Left arm overlay is at u+16 (48-63, 48-63)

    return None


def get_skin_pixel(skin_img, u, v):
    """Get pixel color from skin, handling old format skins and overlay layers."""
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

    # Get base layer pixel
    base_pixel = skin_img.getpixel((u, v))

    # For 64x64 skins, check overlay layer
    if h == 64:
        overlay_offset = get_overlay_offset(u, v)
        if overlay_offset:
            ou = u + overlay_offset[0]
            ov = v + overlay_offset[1]
            if 0 <= ou < w and 0 <= ov < h:
                overlay_pixel = skin_img.getpixel((ou, ov))
                # If overlay is not transparent, use it
                if overlay_pixel[3] > 0:
                    return overlay_pixel
                # If base is transparent but overlay isn't, we already returned
                # If both are transparent or overlay is transparent, use base

    # If base pixel is transparent, return a fallback color
    if base_pixel[3] == 0:
        return (0, 0, 0, 255)  # Default to black if fully transparent

    return base_pixel


def create_statue_schematic(skin_img, username, hollow=False, scale=1):
    """Create a Litematica schematic of the player statue.

    Args:
        skin_img: PIL Image of the skin
        username: Player username
        hollow: If True, interior blocks are air
        scale: Scale factor (1 = 1 block per pixel, 2 = 2x2 blocks per pixel, etc.)
    """
    # Base dimensions (scale=1)
    base_width, base_height, base_length = 16, 33, 8

    # Scaled dimensions
    width = base_width * scale
    height = base_height * scale
    length = base_length * scale

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
        if 0 <= x < width and 0 <= y < height and 0 <= z < length:
            idx = (y * length + z) * width + x
            blocks_array[idx] = get_or_add_palette(block_name)

    # UV layouts for each body part
    UV_MAP = {
        'head': {
            'front': (8, 8), 'back': (24, 8), 'right': (0, 8),
            'left': (16, 8), 'top': (8, 0), 'bottom': (16, 0)
        },
        'body': {
            'front': (20, 20), 'back': (32, 20), 'right': (16, 20),
            'left': (28, 20), 'top': (20, 16), 'bottom': (28, 16)
        },
        'right_arm': {
            'front': (44, 20), 'back': (52, 20), 'right': (40, 20),
            'left': (48, 20), 'top': (44, 16), 'bottom': (48, 16)
        },
        'left_arm': {
            'front': (36, 52), 'back': (44, 52), 'right': (32, 52),
            'left': (40, 52), 'top': (36, 48), 'bottom': (40, 48)
        },
        'right_leg': {
            'front': (4, 20), 'back': (12, 20), 'right': (0, 20),
            'left': (8, 20), 'top': (4, 16), 'bottom': (8, 16)
        },
        'left_leg': {
            'front': (20, 52), 'back': (28, 52), 'right': (16, 52),
            'left': (24, 52), 'top': (20, 48), 'bottom': (24, 48)
        }
    }

    def get_face_pixel(part, sx, sy, sz, part_width, part_height, part_depth):
        """Get pixel from correct face based on block position within body part."""
        uv = UV_MAP[part]

        # Top face: y at max
        if sy == part_height - 1:
            u, v = uv['top']
            return get_skin_pixel(skin_img, u + sx, v + sz)
        # Bottom face: y at min
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
            return get_skin_pixel(skin_img, u + (part_width - 1 - sx), v + (part_height - 1 - sy))
        # Right face: x at min
        elif sx == 0:
            u, v = uv['right']
            return get_skin_pixel(skin_img, u + (part_depth - 1 - sz), v + (part_height - 1 - sy))
        # Left face: x at max
        elif sx == part_width - 1:
            u, v = uv['left']
            return get_skin_pixel(skin_img, u + sz, v + (part_height - 1 - sy))
        # Interior block
        else:
            return None  # Will be handled by hollow check

    def is_surface_block(sx, sy, sz, part_width, part_height, part_depth):
        """Check if block is on the surface of the body part."""
        return (sx == 0 or sx == part_width - 1 or
                sy == 0 or sy == part_height - 1 or
                sz == 0 or sz == part_depth - 1)

    def fill_body_part(part, base_x, base_y, base_z, part_width, part_height, part_depth):
        """Fill a body part with blocks, handling scale and hollow options."""
        for sy in range(part_height):
            for sx in range(part_width):
                for sz in range(part_depth):
                    # Check if this is an interior block
                    is_surface = is_surface_block(sx, sy, sz, part_width, part_height, part_depth)

                    if hollow and not is_surface:
                        # Skip interior blocks for hollow statue
                        continue

                    # Get pixel color for this position
                    pixel = get_face_pixel(part, sx, sy, sz, part_width, part_height, part_depth)

                    if pixel is None:
                        # Interior block in solid mode - use a fill color or front face
                        uv = UV_MAP[part]
                        u, v = uv['front']
                        pixel = get_skin_pixel(skin_img, u + sx, v + (part_height - 1 - sy))

                    r, g, b, a = pixel

                    # Calculate scaled block positions
                    scaled_x = base_x * scale + sx * scale
                    scaled_y = base_y * scale + sy * scale
                    scaled_z = base_z * scale + sz * scale

                    if scale == 1:
                        # Simple case: one block per pixel
                        set_block(scaled_x, scaled_y, scaled_z, find_closest_block(r, g, b))
                    else:
                        # Get best block combination for this pixel
                        blocks = find_best_block_combination((r, g, b), scale)

                        # Place blocks in a scale x scale pattern
                        block_idx = 0
                        for dy in range(scale):
                            for dx in range(scale):
                                for dz in range(scale):
                                    # For surface blocks, only place on the visible surface
                                    if is_surface:
                                        # Determine which sub-blocks are on the outer surface
                                        on_outer = False
                                        if sx == 0 and dx == 0:
                                            on_outer = True
                                        if sx == part_width - 1 and dx == scale - 1:
                                            on_outer = True
                                        if sy == 0 and dy == 0:
                                            on_outer = True
                                        if sy == part_height - 1 and dy == scale - 1:
                                            on_outer = True
                                        if sz == 0 and dz == 0:
                                            on_outer = True
                                        if sz == part_depth - 1 and dz == scale - 1:
                                            on_outer = True

                                        if hollow and not on_outer:
                                            continue

                                    # Use blocks from the combination (cycle through if needed)
                                    block = blocks[block_idx % len(blocks)]
                                    set_block(scaled_x + dx, scaled_y + dy, scaled_z + dz, block)
                                    block_idx += 1

    # Fill all body parts with base coordinates
    # Head: x=4-11, y=25-32, z=0-7 (8x8x8)
    fill_body_part('head', 4, 25, 0, 8, 8, 8)

    # Body: x=4-11, y=13-24, z=2-5 (8x12x4)
    fill_body_part('body', 4, 13, 2, 8, 12, 4)

    # Right Arm: x=0-3, y=13-24, z=2-5 (4x12x4)
    fill_body_part('right_arm', 0, 13, 2, 4, 12, 4)

    # Left Arm: x=12-15, y=13-24, z=2-5 (4x12x4)
    fill_body_part('left_arm', 12, 13, 2, 4, 12, 4)

    # Right Leg: x=4-7, y=1-12, z=2-5 (4x12x4)
    fill_body_part('right_leg', 4, 1, 2, 4, 12, 4)

    # Left Leg: x=8-11, y=1-12, z=2-5 (4x12x4)
    fill_body_part('left_leg', 8, 1, 2, 4, 12, 4)

    # Count non-air blocks
    total_blocks = sum(1 for b in blocks_array if b != 0)

    # Build NBT structure
    nbits = max(2, math.ceil(math.log2(len(palette)))) if len(palette) > 1 else 2
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

    desc = f'Statue of {username}'
    if scale > 1:
        desc += f' (scale {scale}x)'
    if hollow:
        desc += ' (hollow)'

    root = Compound({
        'Version': Int(6),
        'SubVersion': Int(1),
        'MinecraftDataVersion': Int(3465),
        'Metadata': Compound({
            'Name': String(f'{username}_statue'),
            'Author': String('Minecraft Statue Generator'),
            'Description': String(desc),
            'RegionCount': Int(1),
            'TotalBlocks': Int(total_blocks),
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
    hollow = data.get('hollow', False)
    scale = data.get('scale', 1)

    # Validate inputs
    if not username:
        return jsonify({'error': 'Username is required'}), 400

    try:
        scale = int(scale)
        if scale < 1 or scale > 10:
            return jsonify({'error': 'Scale must be between 1 and 10'}), 400
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid scale value'}), 400

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

    # Create schematic with options
    root = create_statue_schematic(skin_img, username, hollow=hollow, scale=scale)

    # Save to bytes
    output = io.BytesIO()
    nbt_file = nbtlib.File(root, gzipped=True)
    nbt_file.save(output)
    output.seek(0)

    # Generate filename with options
    filename = f'{username}_statue'
    if scale > 1:
        filename += f'_{scale}x'
    if hollow:
        filename += '_hollow'
    filename += '.litematic'

    return send_file(
        output,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name=filename
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
