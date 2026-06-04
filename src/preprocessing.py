# src/preprocessing.py
#
# PURPOSE: take a raw scanned absence sheet image and return two clean
# cropped regions — the header and the student table — ready for OCR
# and checkbox detection.
#
# PIPELINE:
#   raw BGR image
#   → deskew        (fix rotation)
#   → enhance       (fix lighting)
#   → binarize      (pure black/white)
#   → split         (separate header from table)


import cv2          # OpenCV — all image operations
import numpy as np  # NumPy — arrays and math


# =============================================================================
# PUBLIC FUNCTION — this is the only function the rest of the project calls
# =============================================================================

def preprocess(image_path):
    """
    INPUT : file path string pointing to a scanned absence sheet
    OUTPUT: dictionary with these keys:
              'original'      — the raw loaded image, untouched (BGR)
              'binary'        — the full cleaned binary image
              'header'        — the header region cropped out (BGR)
              'table'         — the table region cropped out (BGR)
              'header_binary' — the header region in binary black/white
              'table_binary'  — the table region in binary black/white
    """
    image = _load(image_path)
    deskewed = _deskew(image)
    enhanced = _enhance(deskewed)
    binary = _binarize(enhanced)
    header, table = _split(enhanced, binary)
    header_binary = _binarize(header)
    table_binary = _binarize(table)

    return {
        'original':      image,
        'binary':        binary,
        'header':        header,
        'table':         table,
        'header_binary': header_binary,
        'table_binary':  table_binary,
    }


# =============================================================================
# PRIVATE FUNCTIONS — internal steps, not called from outside this file
# =============================================================================
# Functions prefixed with _ are private by convention in Python.
# They exist as separate functions purely for readability — each one does
# exactly one thing, has a clear name, and can be tested individually.


def _load(path):
    """
    IN : file path string
    OUT: NumPy array of shape (h, w, 3), dtype uint8, BGR color order

    cv2.imread reads the file and decodes it into a NumPy array.
    It returns None silently on failure — no exception by default.
    We check manually and raise a clear error.
    """
    image = cv2.imread(path)

    if image is None:
        # None means the file does not exist, is unreadable,
        # or is in an unsupported format
        raise FileNotFoundError(f"Cannot load image: {path}")

    return image


def _deskew(image, max_angle=10.0):
    """
    IN : BGR image (h, w, 3) that may be slightly rotated
    OUT: BGR image (h, w, 3) with rotation corrected

    HOW:
      1. Convert to grayscale — Hough only needs intensity, not color
      2. Binarize inverted — Hough needs white objects on black background
      3. Close horizontally — merge letters on the same line into solid bars
      4. Hough Line Transform — detect those bars as line segments
      5. Compute the median angle of all near-horizontal lines
      6. Rotate the original image by that angle

    max_angle: ignore detected lines steeper than this.
               Protects against accidentally detecting vertical lines.
    """

    # --- Step 1: grayscale ---
    # IN : (h, w, 3) BGR
    # cv2.cvtColor converts between color spaces.
    # COLOR_BGR2GRAY uses the formula: gray = 0.114B + 0.587G + 0.299R
    # OUT: (h, w) single channel, values 0-255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Step 2: inverted binary ---
    # IN : (h, w) grayscale
    # cv2.threshold converts every pixel to either 0 or 255.
    # Parameters:
    #   gray       — input array
    #   0          — threshold value placeholder (Otsu ignores this)
    #   255        — value assigned to pixels that pass the threshold
    #   THRESH_BINARY_INV — pixels BELOW threshold → 255 (white)
    #                       pixels ABOVE threshold → 0   (black)
    #                       INV = inverted = text is white, background black
    #   THRESH_OTSU — automatically find the best threshold value
    #                 by analyzing the histogram
    # Returns a tuple: (computed_threshold, result_array)
    # The _ discards the computed threshold value, we only want the image
    # OUT: (h, w) binary, text=255, background=0
    _, binary_inv = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --- Step 3: merge letters into lines ---
    # IN : (h, w) binary — individual white letters on black background
    #
    # cv2.getStructuringElement creates a kernel — a small matrix
    # that defines the shape of a morphological operation.
    # MORPH_RECT = rectangular kernel
    # (40, 1) = 40 pixels wide, 1 pixel tall — a horizontal ruler
    #
    # cv2.morphologyEx applies a morphological operation.
    # MORPH_CLOSE = dilation then erosion:
    #   - dilation expands white pixels 40px horizontally
    #     → gaps between letters are filled
    #   - erosion shrinks back
    #     → individual letters on same line merge into one solid bar
    #
    # OUT: (h, w) binary — solid white horizontal bars where text lines were
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    closed = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, h_kernel)

    # --- Step 4: detect line segments ---
    # IN : (h, w) binary with solid horizontal bars
    #
    # cv2.HoughLinesP — Probabilistic Hough Line Transform
    # Scans the image and returns line segments as pairs of endpoints.
    #
    # Parameters:
    #   closed        — input binary image
    #   1             — distance resolution: search at 1-pixel precision
    #   np.pi / 180   — angle resolution: search every 1 degree
    #                   np.pi / 180 converts 1 degree to radians
    #   threshold=150 — a line needs at least 150 white pixels voting for it
    #   minLineLength=80  — discard segments shorter than 80 pixels
    #   maxLineGap=15     — connect segments with a gap smaller than 15px
    #
    # OUT: array of shape (N, 1, 4) — N detected lines
    #      each line = [[x1, y1, x2, y2]] — two endpoint coordinates
    #      returns None if no lines found
    lines = cv2.HoughLinesP(
        closed, 1, np.pi / 180,
        threshold=150,
        minLineLength=80,
        maxLineGap=15
    )

    if lines is None:
        # no lines detected — image is probably already straight
        return image

    # --- Step 5: compute median angle ---
    # IN : array of shape (N, 1, 4)
    angles = []
    for line in lines:
        # line has shape (1, 4) — the [0] unwraps it to just 4 values
        x1, y1, x2, y2 = line[0]

        # np.arctan2(vertical_component, horizontal_component)
        # returns the angle of this line segment in RADIANS
        # np.degrees() converts radians to degrees
        # a perfectly horizontal line = 0 degrees
        # a line tilted 2 degrees clockwise = -2 degrees
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # only keep near-horizontal lines
        # abs() makes -5 and +5 both pass the same check
        if abs(angle) < max_angle:
            angles.append(angle)

    if not angles:
        # all detected lines were too steep — no skew correction needed
        return image

    # np.median is more reliable than np.mean here —
    # a few badly detected lines won't distort the result
    # because median ignores outliers
    skew_angle = float(np.median(angles))

    # --- Step 6: rotate ---
    # IN : original BGR image (h, w, 3), skew_angle in degrees
    h, w = image.shape[:2]
    # image.shape returns (h, w, 3)
    # [:2] slices only the first two values — we don't need channel count here

    center = (w // 2, h // 2)
    # // is integer division — guarantees whole pixel coordinates
    # the center pixel is where the rotation pivots

    # cv2.getRotationMatrix2D builds a 2x3 transformation matrix M
    # that describes: rotate by skew_angle degrees around center, scale 1.0
    M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)

    # cv2.warpAffine applies matrix M to every pixel in the image
    # (w, h) — output size, same as input
    # INTER_CUBIC — bicubic interpolation, smoother result than linear
    # BORDER_REPLICATE — fills corners created by rotation by repeating
    #                    the nearest edge pixel, instead of leaving black
    # OUT: (h, w, 3) BGR — same shape, rotation corrected
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def _enhance(image, clip_limit=2.0, tile_size=(8, 8)):
    """
    IN : BGR image (h, w, 3) with potentially uneven lighting
    OUT: BGR image (h, w, 3) with improved local contrast

    HOW — CLAHE on the L channel of LAB color space:

    Why LAB?
      RGB and BGR mix color and brightness together — you cannot
      touch brightness without also affecting color.
      LAB separates them:
        L = lightness (brightness only)
        A = green-to-red color axis
        B = blue-to-yellow color axis
      We only enhance L. A and B stay untouched → colors are preserved.

    Why CLAHE instead of equalizeHist?
      equalizeHist applies one correction to the entire image globally.
      If one corner is dark and another is bright, a global correction
      helps one and hurts the other.
      CLAHE (Contrast Limited Adaptive Histogram Equalization) divides
      the image into small tiles and equalizes each tile independently.
      clip_limit prevents over-amplification of noise in uniform regions.
    """

    # --- convert BGR to LAB ---
    # IN : (h, w, 3) BGR
    # OUT: (h, w, 3) LAB — same shape, different meaning per channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # --- split into 3 separate channels ---
    # cv2.split takes a multi-channel array and returns
    # a list of single-channel arrays
    # IN : (h, w, 3)
    # OUT: three arrays each of shape (h, w)
    l, a, b = cv2.split(lab)

    # --- apply CLAHE to L only ---
    # cv2.createCLAHE creates the CLAHE object (not applied yet)
    # clip_limit=2.0 — limits contrast amplification to prevent noise boost
    # tileGridSize=(8,8) — divides image into 8x8=64 tiles
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    # clahe.apply() runs the equalization
    # IN : (h, w) single channel — the L channel
    # OUT: (h, w) single channel — L with enhanced local contrast
    l_enhanced = clahe.apply(l)

    # --- merge channels back ---
    # cv2.merge takes a list of single-channel arrays
    # and stacks them into one multi-channel array
    # IN : three arrays of (h, w)
    # OUT: (h, w, 3) LAB with enhanced L
    lab_enhanced = cv2.merge([l_enhanced, a, b])

    # --- convert back to BGR ---
    # IN : (h, w, 3) LAB
    # OUT: (h, w, 3) BGR
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def _binarize(image, block_size=31, C=15):
    """
    IN : BGR image (h, w, 3) OR grayscale image (h, w)
    OUT: binary image (h, w), dtype uint8, text=0 (black), background=255 (white)

    HOW — adaptive Gaussian thresholding:

    Why adaptive instead of Otsu?
      Otsu computes ONE threshold for the entire image.
      On a page with a shadow in one corner, the threshold that works
      for the bright area will make the dark area all black.
      Adaptive thresholding computes a DIFFERENT threshold for each
      small region of the image — it handles uneven lighting correctly.

    block_size=31 — each pixel's threshold is computed from its
                    31x31 pixel neighborhood. Must be odd.
    C=15          — a constant subtracted from the computed threshold.
                    Increasing C makes the result sparser (less black).
    """

    # handle both BGR and grayscale input
    # if image has 3 dimensions it is color — convert to gray first
    # if it already has 2 dimensions it is grayscale — use directly
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # GaussianBlur smooths the image slightly before thresholding
    # this reduces the effect of small noise dots on the threshold calculation
    # (5, 5) — kernel size, must be odd. Larger = more smoothing.
    # 0      — standard deviation, computed automatically from kernel size
    # IN : (h, w) grayscale
    # OUT: (h, w) grayscale, slightly smoothed
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # cv2.adaptiveThreshold computes local thresholds and binarizes
    # Parameters:
    #   blurred                         — input
    #   255                             — value for passing pixels
    #   ADAPTIVE_THRESH_GAUSSIAN_C      — threshold for each pixel =
    #                                     weighted mean of its block_size
    #                                     neighborhood (Gaussian weights)
    #                                     minus C
    #   THRESH_BINARY_INV               — pixels BELOW threshold → 255
    #                                     pixels ABOVE threshold → 0
    #                                     (text ends up white, bg black)
    #   block_size                      — neighborhood size (must be odd)
    #   C                               — constant subtracted from mean
    # OUT: (h, w) binary, text=255 (white), background=0 (black)
    binary_inv = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )

    # invert so text=0 (black) and background=255 (white)
    # this is the standard convention for document images
    # cv2.bitwise_not flips every bit: 0→255, 255→0
    # IN : (h, w) binary text=255 bg=0
    # OUT: (h, w) binary text=0   bg=255
    return cv2.bitwise_not(binary_inv)


def _split(image, binary):
    """
    IN : enhanced BGR image (h, w, 3)
         binary image (h, w) of the same image
    OUT: header (h1, w, 3) BGR — the region above the first long line
         table  (h2, w, 3) BGR — everything below that line

    HOW:
      The absence sheet has a printed horizontal line separating the
      header from the student table. We detect that line by:
        1. Finding all horizontal structures in the binary image
        2. Locating the first row where a long line spans >70% of page width
        3. Splitting the image at that row
      If no such line is found, we fall back to a 30% / 70% split.
    """

    # --- isolate horizontal lines ---
    # We need the binary image with white foreground (text=255, bg=0)
    # _binarize returns text=0 bg=255, so we invert it first
    # cv2.bitwise_not: 0→255, 255→0
    # IN : (h, w) text=0 bg=255
    # OUT: (h, w) text=255 bg=0
    foreground = cv2.bitwise_not(binary)

    # Build a wide horizontal kernel to keep only long horizontal structures
    # image.shape[1] is the image width
    # // 20 gives us a kernel that is 5% of the image width long
    # This is long enough to detect printed table lines but short enough
    # to ignore individual letters
    kernel_width = image.shape[1] // 20
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))

    # MORPH_OPEN = erosion then dilation
    # With a wide horizontal kernel:
    #   erosion removes everything that isn't at least kernel_width wide
    #   dilation restores what remains
    # Net result: only long horizontal lines survive
    # IN : (h, w) text=255 bg=0
    # OUT: (h, w) only horizontal lines remain as white
    h_lines = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, h_kernel)

    # --- find the split row ---
    # np.sum(h_lines > 0, axis=1):
    #   h_lines > 0 produces a boolean array — True where white, False where black
    #   np.sum(..., axis=1) sums across axis 1 (columns) for each row
    #   result: 1D array of length h, each value = number of white pixels in that row
    # IN : (h, w) binary
    # OUT: (h,) 1D array — white pixel count per row
    row_white_counts = np.sum(h_lines > 0, axis=1)

    # a real printed table line spans most of the page width
    # we require it to cover at least 70% of the image width
    long_line_threshold = image.shape[1] * 0.7

    # np.where returns indices where a condition is True
    # here: indices of rows where the white pixel count exceeds our threshold
    # IN : (h,) 1D array of counts
    # OUT: tuple containing one array of matching row indices
    candidate_rows = np.where(row_white_counts > long_line_threshold)[0]

    if len(candidate_rows) > 0:
        split_y = int(candidate_rows[0])
    else:
        # fallback: no long line found, assume header is top 30%
        split_y = int(image.shape[0] * 0.3)

    # ensure split_y is not at the very bottom edge
    split_y = min(split_y, image.shape[0] - 10)

    # --- crop ---
    # NumPy slice: array[row_start:row_end, col_start:col_end]
    # : alone means "all" — no start or end constraint
    # image[:split_y, :]  — all rows from 0 to split_y, all columns → header
    # image[split_y:, :]  — all rows from split_y to end, all columns → table
    # IN : (h, w, 3) full image
    # OUT: (split_y, w, 3) header
    #      (h - split_y, w, 3) table
    header = image[:split_y, :]
    table = image[split_y:, :]

    return header, table


# =============================================================================
# QUICK TEST — run this file directly to verify preprocessing works
# Usage: python src/preprocessing.py data/raw/real/as1.jpeg
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image_path>")
        sys.exit(1)

    result = preprocess(sys.argv[1])

    for key, img in result.items():
        filename = f"debug_{key}.jpg"
        cv2.imwrite(filename, img)
        print(f"saved {filename}  shape={img.shape}")