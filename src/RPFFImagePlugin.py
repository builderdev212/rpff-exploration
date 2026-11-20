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

## Constants ##
CHROMA_FACTOR = 4
QUANT_Y_MULT = 30
QUANT_CHROMA_MULT = 60
QUANT_Y_BITS = 3
QUANT_CHROMA_BITS = 2
TF_BLOCK_SIZE = 16
TF_TYPE = ("DCT", 2)


## Array Helpers ##
def _2D_arr_pad(channel: np.ndarray, div_size: int) -> np.ndarray:
    """
    Helper function to pad a 2D array to a certain size to split into blocks.

    Args:
        channel: Two-dimensional array of values.
        div_size: Size of divisons that will be done.

    Returns:
        A two-dimensional padded array
    """

    height, width = channel.shape
    return np.pad(
        channel,
        ((0, (-height) % div_size), (0, (-width) % div_size)),
        mode="edge",
    )


def _2D_arr_block(channel: np.ndarray, block_size: int) -> np.ndarray:
    """
    Helper function to split a 2D array into blocks.

    Args:
        channel: Two-dimensional array of values.
        div_size: Size of blocks (div_size x div_size).

    Returns:
        A two-dimensional array split into blocks.
    """

    height, width = channel.shape
    return channel.reshape(
        height // block_size, block_size, width // block_size, block_size
    )


def _2D_zz_arr_mask(block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function to create a zig-zag array mask to do the zig-zag of a 2D array.

    Args:
        block_size: Size of blocks to make an array for (block_size x block size).

    Returns:
        Two arrays, one for the rows and a second for the columns.
    """

    rows = []
    cols = []

    # go through all diagonals
    for diag in range(2 * block_size - 1):
        # Get valid first/last positions for current diag
        first = max(0, diag - block_size + 1)
        last = min(diag, block_size - 1)

        # set the rows for the diagonal
        diag_rows = range(first, last + 1)
        if diag % 2 == 0:
            diag_rows = reversed(diag_rows)

        # store rows/cols
        for row in diag_rows:
            rows.append(row)
            cols.append(diag - row)

    return (np.array(rows, dtype=np.intp), np.array(cols, dtype=np.intp))


def _2D_zz_block(
    channel: np.ndarray, block_size: int, inverse: bool = False
) -> np.ndarray:
    """
    Perfom zig zag ordering on the 2D block array

    Args:
        channel: Two-dimensional array of values.
        block_size: Size of the blocks in the array (block_size x block_size)
        inverse: False if encoding, true if decoding.

    Returns:
        A two-dimensional zig zag organized array.
    """

    # Get the mask for zig-zag
    zz_rows, zz_cols = _2D_zz_arr_mask(block_size)

    if inverse:
        # array is rows, cols, and size of a block
        block_rows, block_cols, _ = channel.shape

        # create empty array to fill with compressed version
        blocks = np.empty(
            (block_rows, block_cols, block_size, block_size),
            dtype=channel.dtype,
        )
        blocks[..., zz_rows, zz_cols] = channel

        # transpose and return the unzig-zaged array
        return blocks.transpose(0, 2, 1, 3).reshape(
            block_rows * block_size,
            block_cols * block_size,
        )
    else:
        # split into blocks and transpose
        channel = _2D_arr_block(channel, block_size).transpose(0, 2, 1, 3)
        # order in zig-zag format
        return channel[..., zz_rows, zz_cols]


def _truncate_zero_runs(ac: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """
    Quantize a channel.

    Args:
        ac: Two-dimensional array of ac values.

    Returns:
        A list with a two-dimensional subsampled array and the numbers of zeros truncated.
    """

    trunc_ac = []

    for block in ac.reshape(-1, ac.shape[-1]):
        # flatten the current block
        nonzero_ac = np.flatnonzero(block)

        # set the number of trailing zeros
        if nonzero_ac.size == 0:
            trailing_zero = block.size
        else:
            trailing_zero = block.size - 1 - nonzero_ac[-1]

        # append the flattened block with the zero's removed
        trunc_ac.append((block[: block.size - trailing_zero], trailing_zero))

    return trunc_ac


## Chroma Subsampling ##
def chroma_size(width: int, height: int) -> tuple[int, int]:
    """
    Helper function to convert the height/width of an image to the chroma width.

    Args:
        width: Width of the image.
        height: Height of the image.

    Returns:
        Converted width and height values.
    """

    chroma_width = (width + CHROMA_FACTOR - 1) // CHROMA_FACTOR
    chroma_height = (height + CHROMA_FACTOR - 1) // CHROMA_FACTOR
    return (chroma_width, chroma_height)


def chroma_subsample(channel: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Chroma subsampling function, takes a given channel and subsamples it.

    Args:
        channel: Two-dimensional array of values.
        inverse: False if encoding, true if decoding.

    Returns:
        A two-dimensional subsampled array.
    """

    if inverse:
        # duplicate rows and columns to "undo" subsampling (note, this is why it's lossy)
        channel = channel.repeat(CHROMA_FACTOR, axis=0)
        channel = channel.repeat(CHROMA_FACTOR, axis=1)
    else:
        channel = channel.astype(np.uint16)  # 16 bits to avoid overflow

        # use np to pad size for averaging by copying edge data
        channel = _2D_arr_pad(channel, CHROMA_FACTOR)
        # reshape the data into CHROMA_FACTOR x CHROMA_FACTOR blocks to average
        channel = _2D_arr_block(channel, CHROMA_FACTOR)

        # average the values to cut the size in half and convert back to 8 bit
        channel = np.round(channel.mean(axis=(1, 3))).astype(np.uint8)

    return channel


## Transform Coding ##
def transform_blocks(channel: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Transform coding function.

    Args:
        channel: Two-dimensional array of values.
        inverse: False if encoding, true if decoding.

    Returns:
        A two-dimensional subsampled array.
    """

    # Format into blocks
    if not inverse:
        # pad to block size for tf
        channel = _2D_arr_pad(channel, TF_BLOCK_SIZE)

    og_height, og_width = channel.shape
    channel = channel.astype(np.float64)

    # reshape into TF_BLOCK_SIZE x TF_BLOCK_SIZE blocks
    channel = _2D_arr_block(channel, TF_BLOCK_SIZE)

    # perform tf
    if TF_TYPE[0] == "DCT":
        tf = idctn if inverse else dctn
    elif TF_TYPE[0] == "DST":
        tf = idstn if inverse else dstn
    output = tf(channel, type=TF_TYPE[1], norm="ortho", axes=(1, 3))

    return output.reshape(og_height, og_width)


## Quantization ##
def quantize(channel: np.ndarray, quant_mult: int, inverse: bool = False) -> np.ndarray:
    """
    Quantize a channel.

    Args:
        channel: Two-dimensional array of values.
        quant_mult: Factor to quantize by.
        inverse: False if encoding, true if decoding.

    Returns:
        A two-dimensional subsampled array.
    """

    if inverse:
        # multiply to decode the quantization
        return (channel * quant_mult).astype(np.float32)
    else:
        # divide and round to encode with quantization
        return np.round(channel / quant_mult).astype(np.int16)


## Data Packing/Unpacking ##
def embed_dc_in_channel(channel: np.ndarray, quant_bits: int) -> bytes:
    """
    Embed DC with AC and pack the data into a bytes array.

    Args:
        channel: Two-dimensional array of values.
        quant_bits: Number of bits to store for each AC value.

    Returns:
        Return a byte value to write to the RPFF file.
    """

    # Split into blocks and zigzag
    blocks = _2D_zz_block(channel, TF_BLOCK_SIZE)

    # Find DC values (top left coefficent)
    dc = blocks[..., 0].astype(np.int8)

    # Clip AC values
    limit = 1 << (quant_bits - 1)
    ac = np.clip(blocks[..., 1:], -limit, limit - 1).astype(np.int8)
    trunc_ac = _truncate_zero_runs(ac)

    # returned the packed channel
    return dc.tobytes() + pack_signed(trunc_ac, quant_bits)


def pack_signed(ac: list[tuple[np.ndarray, int]], bits_per_pixel: int) -> bytes:
    """
    Pack the ac bytes value into sizes smaller than a byte.

    Args:
        ac: Raw AC values and truncated zero value as formatted by the output of _truncate_zero_runs.
        bits_per_pixel: Number of bits to store for each AC value.

    Returns:
        Returned the packed AC values.
    """

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
    """
    Unpack ac data for decoding.

    Args:
        raw: Raw bytes of packed ac values.
        bits_per_pixel: number of bits per ac value.
        pixels: number of total pixels expected.

    Returns:
        Return a 2D array of unpacked AC values in zig zag order.
    """

    bit_arr = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")

    # Get rid of bits added by tobytes()
    bit_arr = bit_arr[: pixels * bits_per_pixel].reshape(pixels, bits_per_pixel)

    # Convert to int8
    bit_shift = 1 << np.arange(bits_per_pixel)
    codes = (bit_arr * bit_shift).sum(axis=1)
    codes = codes.astype(np.int16) - (1 << (bits_per_pixel - 1))

    return codes.astype(np.int8)


## Pillow Image Class Functions ##
def _accept(prefix: bytes) -> bool:
    """
    Check if the pillow library is getting an RPFF file

    Args:
        prefix: First 4 bytes of the file.

    Returns:
        Bool of whether or not prefix was expected
    """

    return prefix.startswith(b"RPFF")


class RPFFDecoder(ImageFile.PyDecoder):

    _pulls_fd = True
    name = "rpff_decoder"

    # Read helper function
    def _read(self, byte_count: int) -> bytes:
        """
        Helper function to read from file

        Args:
            byte_count: Number of bytes to read.

        Returns:
            Return the bytes read from the file.
        """

        data = self.fd.read(byte_count)
        if len(data) != byte_count:
            raise OSError(
                f"Truncated RPFF pixel data. Expected {byte_count}B, got {len(data)}B."
            )

        return data

    # Helper function to turn data into NP array
    def _read_channel(self, rows: int, columns: int, bits_per_pixel: int) -> np.ndarray:
        """
        Helper function to read a channel from the data encoded in the file.

        Args:
            rows: Number of rows in the channel.
            columns: Number of columns in the channel.
            bits_per_pixel: Number of bits used per pixel value.

        Returns:
            2D array of pixels, but not uncompressed, rather just properly ordered.
        """

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
        block_rows = p_rows // TF_BLOCK_SIZE
        block_cols = p_columns // TF_BLOCK_SIZE

        zz_blocks = np.concatenate(
            (dc.reshape(dc_count, 1), ac.reshape(dc_count, -1)),
            axis=1,
        ).astype(np.float32)

        zz_blocks = zz_blocks.reshape(block_rows, block_cols, TF_BLOCK_SIZE**2)

        # Do the inverse of the zig-zag ordering on the blocks, then return the result
        return _2D_zz_block(zz_blocks, TF_BLOCK_SIZE, inverse=True)

    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        """
        Decode function

        Args:
            buffer: image file buffer.

        Returns:
            Returns (-1, 0) to signal the image was decoded properly.
        """

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
        y = quantize(y, QUANT_Y_MULT, inverse=True)
        cb = quantize(cb, QUANT_CHROMA_MULT, inverse=True)
        cr = quantize(cr, QUANT_CHROMA_MULT, inverse=True)

        # Inverse tf
        y = transform_blocks(y, True)[:height, :width]
        cb = transform_blocks(cb, True)[:chroma_height, :chroma_width]
        cr = transform_blocks(cr, True)[:chroma_height, :chroma_width]

        # Shift the data back to uint8 range (from -127/128 to 0/255)
        y = y + 128
        cb = cb + 128
        cr = cr + 128

        # Inverse chroma subsampling
        cb = chroma_subsample(cb, inverse=True)[:height, :width]
        cr = chroma_subsample(cr, inverse=True)[:height, :width]

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
        """
        Helper function to open the RPFF file and parse the header information.
        """

        header = self.fp.read(12)

        # Check Signature/Version
        if header[:4] != b"RPFF":
            raise SyntaxError("Invalid RPFF Header")
        if len(header) != 12:
            raise SyntaxError("Truncated RPFF Header")

        # Set image size in pixels
        width = int.from_bytes(header[4:8], byteorder="little")
        height = int.from_bytes(header[8:12], byteorder="little")
        if width == 0 or height == 0:
            raise SyntaxError("Invalid image dimensions")
        self._size = (width, height)

        # Set raw data type
        self._mode = "YCbCr"

        # Setup decoder to use custom decoder and to start the image data after the 12 byte header.
        self.tile = [
            ImageFile._Tile(
                RPFFDecoder.name,  # Custom compression
                (0, 0) + self.size,  # Image size
                12,  # Header size to skip
                (),
            )
        ]


def _save(image: Image.Image, fp: BinaryIO, _filename: str | bytes) -> None:
    """
    Encode raw image data and save the data into an .rpff file.

    Args:
        image: Image class.
        fp: File IO class to write to the image file.
        _filename: Image file name.
    """

    # Ensure the data is in YCbCr not RGB
    image = image.convert("YCbCr")
    pixel_arr = np.asarray(image)

    ## Encode data ##

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

    # Quantize
    y = quantize(y, QUANT_Y_MULT)
    cb = quantize(cb, QUANT_CHROMA_MULT)
    cr = quantize(cr, QUANT_CHROMA_MULT)

    ## Header
    fp.write(b"RPFF")  # Signature
    fp.write(image.width.to_bytes(4, "little"))  # Width
    fp.write(image.height.to_bytes(4, "little"))  # Height

    ## Data
    fp.write(embed_dc_in_channel(y, QUANT_Y_BITS))
    fp.write(embed_dc_in_channel(cb, QUANT_CHROMA_BITS))
    fp.write(embed_dc_in_channel(cr, QUANT_CHROMA_BITS))


# Register the custom format with PIL to be used with their functions.
Image.register_decoder(RPFFDecoder.name, RPFFDecoder)
Image.register_open(RPFFImageFile.format, RPFFImageFile, _accept)
Image.register_save(RPFFImageFile.format, _save)
Image.register_extensions(RPFFImageFile.format, [".rpff"])
