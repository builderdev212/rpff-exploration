"""
RPFF File Format Layout:
    (32B) Header -
     ├─(4B)  Signature "RPFF"
     ├─(1B)  Version "1" (uint8)
     ├─(4B)  Width "640" (uint32)
     ├─(4B)  Height "480" (uint32)
     └─(19B) Undefined
"""

from typing import BinaryIO
import numpy as np
from scipy.fft import dctn, idctn, dstn, idstn
from PIL import Image, ImageFile

TF_BLOCK_SIZE = 16
QUANT_Y_MULT = 30
QUANT_CHROMA_MULT = 60
QUANT_Y_BITS = 3
QUANT_CHROMA_BITS = 2


def _accept(prefix: bytes) -> bool:
    return prefix.startswith(b"RPFF")


def zigzag_indices() -> tuple[np.ndarray, np.ndarray]:
    pos = []

    for row in range(TF_BLOCK_SIZE):
        for col in range(TF_BLOCK_SIZE):
            diag = row + col

            # for even, larger numbers come first
            if diag % 2 == 0:
                order = -row
            # for odd, smaller numbers come first
            else:
                order = row

            pos.append((diag, order, row, col))

    pos.sort()

    indicies = np.array(pos, dtype=np.intp)
    return indicies[:, 2], indicies[:, 3]


ZZ_ROWS, ZZ_COLS = zigzag_indices()

#RPFF: 0.615% - PSNR (22.2)
def transform_blocks(channel: np.ndarray, inverse: bool = False) -> np.ndarray:
    # Format into blocks
    if not inverse:
        height, width = channel.shape
        # pad to block size for tf
        channel = np.pad(
            channel,
            ((0, (-height) % TF_BLOCK_SIZE), (0, (-width) % TF_BLOCK_SIZE)),
            mode="edge",
        )

    height, width = channel.shape
    channel = channel.astype(np.float32)

    # Reshape into TF_BLOCK_SIZExTF_BLOCK_SIZE blocks to perform tf on
    channel = channel.reshape(
        height // TF_BLOCK_SIZE, TF_BLOCK_SIZE, width // TF_BLOCK_SIZE, TF_BLOCK_SIZE
    )

    # Perform tf
    tf = idctn if inverse else dctn
    output = tf(channel, type=2, norm="ortho", axes=(1, 3))

    return output.reshape(height, width)


def pack_signed(ac: list[tuple[np.ndarray, int]], bits_per_pixel: int) -> bytes:
    offset = 1 << (bits_per_pixel - 1)
    bit_shift = np.arange(bits_per_pixel)
    packed_output = bytearray()

    for channel, trailing_zero in ac:
        channel = channel.ravel().astype(np.int16)

        # Ensure values fit within given range
        if np.any((channel < -offset) | (channel >= offset)):
            raise (
                ValueError(
                    f"Data out of range for packed signed integer of {bits_per_pixel} bits."
                )
            )

        # Map int to uint
        codes = (channel + offset).astype(np.uint8)

        # Convert to bits
        bit_arr = ((codes[:, None] >> bit_shift) & 1).astype(np.uint8)
        bit_arr = np.packbits(bit_arr.ravel(), bitorder="little")

        # Store into bytearray with zero count and truncated data
        packed_output.extend(bytes([trailing_zero]))
        packed_output.extend(bit_arr.tobytes())

    return bytes(packed_output)


def unpack_signed(raw: bytes, bits_per_pixel: int, pixels: int) -> np.ndarray:
    bit_arr = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")

    # Get rid of bits added by tobytes()
    bit_arr = bit_arr[: pixels * bits_per_pixel].reshape(pixels, bits_per_pixel)

    # Convert to int8
    bit_shift = 1 << np.arange(bits_per_pixel)
    codes = (bit_arr * bit_shift).sum(axis=1)
    codes = codes.astype(np.int16) - (1 << (bits_per_pixel - 1))

    return codes.astype(np.int8)


class RPFFDecoder(ImageFile.PyDecoder):

    _pulls_fd = True
    name = "rpff_decoder"

    # Read helper function
    def _read(self, byte_count: int) -> bytes:
        data = self.fd.read(byte_count)
        if len(data) != byte_count:
            raise OSError(
                f"Truncated RPFF pixel data. Expected {byte_count}B, got {len(data)}B."
            )

        return data

    # Helper function to turn data into NP array
    def _read_channel(self, rows: int, columns: int, bits_per_pixel: int) -> np.ndarray:
        # Pad Image size to TF_BLOCK_SIZE blocks
        p_rows = rows + (-rows) % TF_BLOCK_SIZE
        p_columns = columns + (-columns) % TF_BLOCK_SIZE
        dc_count = (p_rows // TF_BLOCK_SIZE) * (p_columns // TF_BLOCK_SIZE)

        # Get DC Values
        dc = self._read(dc_count)  # DC values are 8 bits
        dc = np.frombuffer(dc, dtype=np.int8)

        # Get AC Values
        ac_per_block = TF_BLOCK_SIZE**2 - 1
        ac = np.zeros((dc_count, ac_per_block), dtype=np.int8)
        for block in range(dc_count):
            trailing_zeros = self._read(1)[0]
            remaining = ac_per_block - trailing_zeros
            block_byte_count = (remaining * bits_per_pixel + 7) // 8
            packed = self._read(block_byte_count)
            ac[block, :remaining] = unpack_signed(packed, bits_per_pixel, remaining)

        # Repack Values together
        zz_blocks = np.empty(
            (dc_count, TF_BLOCK_SIZE * TF_BLOCK_SIZE), dtype=np.float32
        )
        zz_blocks[:, 0] = dc
        zz_blocks[:, 1:] = ac.reshape(dc_count, TF_BLOCK_SIZE * TF_BLOCK_SIZE - 1)
        blocks = np.empty((dc_count, TF_BLOCK_SIZE, TF_BLOCK_SIZE), dtype=np.float32)
        blocks[:, ZZ_ROWS, ZZ_COLS] = zz_blocks
        blocks = blocks.reshape(
            p_rows // TF_BLOCK_SIZE,
            p_columns // TF_BLOCK_SIZE,
            TF_BLOCK_SIZE,
            TF_BLOCK_SIZE,
        ).transpose(0, 2, 1, 3)

        return blocks.reshape(p_rows, p_columns)

    def _upscale_chroma_array(
        self, channel: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        channel = channel.repeat(4, axis=0)  # Duplicate rows
        channel = channel.repeat(4, axis=1)  # Duplicate columns
        channel = channel[:height, :width]  # Truncate
        return channel

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        # Get image size
        width = self.state.xsize
        height = self.state.ysize

        # Get data from file
        chroma_width = (width + 3) // 4
        chroma_height = (height + 3) // 4

        y = self._read_channel(height, width, QUANT_Y_BITS)
        cb = self._read_channel(chroma_height, chroma_width, QUANT_CHROMA_BITS)
        cr = self._read_channel(chroma_height, chroma_width, QUANT_CHROMA_BITS)

        # Dequantize
        y = y * QUANT_Y_MULT
        cb = cb * QUANT_CHROMA_MULT
        cr = cr * QUANT_CHROMA_MULT

        # Inverse tf
        y = transform_blocks(y, True)[:height, :width]
        cb = transform_blocks(cb, True)[:chroma_height, :chroma_width]
        cr = transform_blocks(cr, True)[:chroma_height, :chroma_width]

        # Recenter the data
        y = y + 128
        cb = cb + 128
        cr = cr + 128

        # Decode chroma subsampling for 4:2:0
        cb = self._upscale_chroma_array(cb, width, height)
        cr = self._upscale_chroma_array(cr, width, height)

        # Pack separate Y, Cb, and Cr arrays into one array
        pixel_arr = np.stack((y, cb, cr), axis=-1)
        pixel_arr = np.clip(np.floor(pixel_arr + 0.5), 0, 255).astype(np.uint8)

        # Turn into 1D bytes in YCC format
        self.set_as_raw(pixel_arr.tobytes(), "YCbCr")

        # Return all good
        return -1, 0


class RPFFImageFile(ImageFile.ImageFile):

    format = "RPFF"
    format_description = "Experimental Ryan Photo File Format"

    def _open(self) -> None:
        header = self.fp.read(32)

        # Check Signature/Version
        if header[:4] != b"RPFF":
            raise SyntaxError("Invalid RPFF Header")
        if len(header) != 32:
            raise SyntaxError("Truncated RPFF Header")
        if header[4] != 1:
            raise SyntaxError("Unsupported RPFF Version")

        # Set image size in pixels
        width = int.from_bytes(header[5:9], byteorder="little")
        height = int.from_bytes(header[9:13], byteorder="little")
        if width == 0 or height == 0:
            raise SyntaxError("Invalid image dimensions")
        self._size = (width, height)

        # Set raw data type
        self._mode = "YCbCr"

        self.tile = [
            ImageFile._Tile(
                RPFFDecoder.name,  # Custom compression
                (0, 0) + self.size,  # Image size
                32,  # Header size to skip
                (),
            )
        ]


def chroma_subsample(channel: np.ndarray) -> np.ndarray:
    # this is for 4:2:0
    channel = channel.astype(np.uint16)  # 16 bits to avoid overflow
    height, width = channel.shape

    # Use np to pad size for averaging by copying edge data
    channel = np.pad(channel, ((0, (-height) % 4), (0, (-width) % 4)), mode="edge")

    # reshape the data into 2x2 blocks to average
    height, width = channel.shape
    channel = channel.reshape(height // 4, 4, width // 4, 4)

    # average the values to cut the size in half and convert back to 8 bit
    channel = np.floor(channel.mean(axis=(1, 3)) + 0.5).astype(np.uint8)

    return channel


def truncate_zero_runs(ac: np.ndarray) -> list[tuple[np.ndarray, int]]:
    trunc_ac = []

    for block in ac.reshape(-1, ac.shape[-1]):
        nonzero_ac = np.flatnonzero(block)

        if nonzero_ac.size == 0:
            trailing_zero = block.size
        else:
            trailing_zero = block.size - 1 - nonzero_ac[-1]

        trunc_ac.append((block[: block.size - trailing_zero], trailing_zero))

    return trunc_ac


def embed_dc_in_channel(channel: np.ndarray, quant_mult: int, quant_bits: int) -> bytes:
    # Quantize
    quant = np.floor(channel / quant_mult + 0.5).astype(np.int16)

    # Split into blocks and zigzag
    height, width = channel.shape
    blocks = quant.reshape(
        height // TF_BLOCK_SIZE, TF_BLOCK_SIZE, width // TF_BLOCK_SIZE, TF_BLOCK_SIZE
    ).transpose(0, 2, 1, 3)
    blocks = blocks[..., ZZ_ROWS, ZZ_COLS]

    # Find DC values (top left coefficent)
    dc = blocks[..., 0].astype(np.int8)

    # Clip AC values
    limit = 1 << (quant_bits - 1)
    ac = np.clip(blocks[..., 1:], -limit, limit - 1).astype(np.int8)
    ac[..., 8:] = 0 # Keep only the 8 largest AC values
    trunc_ac = truncate_zero_runs(ac)

    return dc.tobytes() + pack_signed(trunc_ac, quant_bits)


def _save(image: Image.Image, fp: BinaryIO, _filename: str | bytes) -> None:
    image = image.convert("YCbCr")
    pixel_arr = np.asarray(image)

    ## Encode data
    # Chroma subsample
    y = pixel_arr[:, :, 0]
    cb = chroma_subsample(pixel_arr[:, :, 1])
    cr = chroma_subsample(pixel_arr[:, :, 2])

    # Center the data
    y = y.astype(np.float32) - 128
    cb = cb.astype(np.float32) - 128
    cr = cr.astype(np.float32) - 128

    # Perform tf
    y = transform_blocks(y)
    cb = transform_blocks(cb)
    cr = transform_blocks(cr)

    ## Header
    fp.write(b"RPFF")  # Signature
    fp.write(bytes([1]))  # Version
    fp.write(image.width.to_bytes(4, "little"))  # Width
    fp.write(image.height.to_bytes(4, "little"))  # Height
    fp.write(bytes(19))  # Reserved

    ## Data
    fp.write(embed_dc_in_channel(y, QUANT_Y_MULT, QUANT_Y_BITS))
    fp.write(embed_dc_in_channel(cb, QUANT_CHROMA_MULT, QUANT_CHROMA_BITS))
    fp.write(embed_dc_in_channel(cr, QUANT_CHROMA_MULT, QUANT_CHROMA_BITS))


Image.register_decoder(RPFFDecoder.name, RPFFDecoder)
Image.register_open(RPFFImageFile.format, RPFFImageFile, _accept)
Image.register_save(RPFFImageFile.format, _save)
Image.register_extensions(RPFFImageFile.format, [".rpff"])
