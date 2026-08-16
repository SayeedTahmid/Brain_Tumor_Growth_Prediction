"""Minimal NIfTI-1/2 reader that works on a non-seekable stream.

Why not nibabel: Stage 1 reads members out of `*.tar.bz2` without extracting
43 GB to disk (§4.1). Members arrive as forward-only streams, so the reader must
be able to parse a header and then continue reading the same stream. Only numpy
is required, which also removes one pinned dependency from bootstrap (§15.4).

Nothing here interprets the data. Shape, spacing, dtype and value range are
measured and reported; the descriptor's uint8 0-255 claim is checked against
them by G10 rather than assumed (§3.1.4, §9 G10).
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass, asdict
from typing import BinaryIO

import numpy as np

# NIfTI datatype code -> (numpy dtype string, bytes per voxel)
DTYPE_CODES = {
    2: ("u1", 1), 4: ("i2", 2), 8: ("i4", 4), 16: ("f4", 4),
    32: ("c8", 8), 64: ("f8", 8), 256: ("i1", 1), 512: ("u2", 2),
    768: ("u4", 4), 1024: ("i8", 8), 1280: ("u8", 8),
}


class NiftiParseError(RuntimeError):
    pass


@dataclass
class NiftiHeader:
    version: int          # 1 or 2
    byteorder: str        # '<' or '>'
    dim: list[int]        # spatial/temporal dims, dim[1:ndim+1]
    ndim: int
    pixdim: list[float]
    datatype_code: int
    dtype: str            # numpy dtype string incl. byte order
    bitpix: int
    vox_offset: int
    scl_slope: float
    scl_inter: float
    qform_code: int
    sform_code: int
    xyzt_units: int
    descrip: str
    # v0.20 — the spatial terms. Previously parsed past and discarded, which
    # left NOTHING on record saying where a volume sits in space. The dose maps
    # are 256x256x190 / 240x240x190 against a 193x229x193 derivative grid, and
    # without offsets there is no way to relate the two: matching by shape alone
    # would assume a shared origin that nothing supports, and GATE-1 would
    # produce a plausible-looking Dice from misplaced dose.
    quatern_b: float = 0.0
    quatern_c: float = 0.0
    quatern_d: float = 0.0
    qoffset_x: float = 0.0
    qoffset_y: float = 0.0
    qoffset_z: float = 0.0
    qfac: float = 1.0
    srow_x: tuple = (0.0, 0.0, 0.0, 0.0)
    srow_y: tuple = (0.0, 0.0, 0.0, 0.0)
    srow_z: tuple = (0.0, 0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def affine(self) -> "np.ndarray | None":
        """Voxel -> world (mm), by the NIfTI-1 precedence rule.

        sform first when sform_code > 0, else the qform when qform_code > 0.
        Returns None when BOTH codes are 0: that is not an identity affine, it
        is the file declining to state a position, and treating it as identity
        is exactly the silent misplacement this field exists to prevent.
        """
        if self.sform_code > 0:
            m = np.array([list(self.srow_x), list(self.srow_y),
                          list(self.srow_z), [0.0, 0.0, 0.0, 1.0]], dtype=float)
            # An sform_code > 0 with a singular 3x3 is a malformed header. It
            # would map every voxel onto the origin — a silent, total
            # misplacement that still returns a well-formed 4x4. Fall through to
            # the qform rather than hand it back.
            if abs(np.linalg.det(m[:3, :3])) > 1e-12:
                return m
        if self.qform_code > 0:
            b, c, d = self.quatern_b, self.quatern_c, self.quatern_d
            a2 = 1.0 - (b * b + c * c + d * d)
            a = float(np.sqrt(a2)) if a2 > 0 else 0.0
            R = np.array([
                [a*a + b*b - c*c - d*d, 2*(b*c - a*d),         2*(b*d + a*c)],
                [2*(b*c + a*d),         a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
                [2*(b*d - a*c),         2*(c*d + a*b),         a*a + d*d - b*b - c*c],
            ], dtype=float)
            sp = list(self.spacing) + [1.0, 1.0, 1.0]
            scale = np.diag([sp[0], sp[1], sp[2] * (self.qfac if self.qfac else 1.0)])
            out = np.eye(4)
            out[:3, :3] = R @ scale
            out[:3, 3] = [self.qoffset_x, self.qoffset_y, self.qoffset_z]
            return out
        return None

    @property
    def spatial_status(self) -> str:
        """Which spatial convention actually governs, after validation."""
        import numpy as _np
        if self.sform_code > 0:
            m = _np.array([list(self.srow_x)[:3], list(self.srow_y)[:3],
                           list(self.srow_z)[:3]], dtype=float)
            if abs(_np.linalg.det(m)) > 1e-12:
                return f"SFORM_code_{self.sform_code}"
            if self.qform_code > 0:
                return f"SFORM_SINGULAR_fell_back_to_QFORM_code_{self.qform_code}"
            return "SFORM_SINGULAR_and_no_QFORM"
        if self.qform_code > 0:
            return f"QFORM_code_{self.qform_code}"
        return "NO_SPATIAL_INFORMATION"

    @property
    def world_origin(self) -> tuple | None:
        """World coordinates (mm) of voxel (0,0,0). None when unknown."""
        a = self.affine
        return None if a is None else tuple(round(float(x), 4) for x in a[:3, 3])

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.dim[:self.ndim])

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(round(float(p), 6) for p in self.pixdim[:self.ndim])

    @property
    def n_voxels(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


def _open_maybe_gzip(fileobj: BinaryIO, name: str) -> BinaryIO:
    if name.endswith(".gz"):
        return gzip.GzipFile(fileobj=fileobj, mode="rb")
    return fileobj


def _read_exact(f: BinaryIO, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise NiftiParseError(f"stream ended after {len(buf)} of {n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def parse_header(stream: BinaryIO) -> tuple[NiftiHeader, BinaryIO]:
    """Parse a NIfTI header from the head of `stream`.

    Returns the header and the same stream positioned immediately after the
    header bytes that were consumed.
    """
    head = _read_exact(stream, 4)
    (n1,) = struct.unpack("<i", head)
    (n2,) = struct.unpack(">i", head)
    if n1 == 348:
        bo, version, hdr_size = "<", 1, 348
    elif n2 == 348:
        bo, version, hdr_size = ">", 1, 348
    elif n1 == 540:
        bo, version, hdr_size = "<", 2, 540
    elif n2 == 540:
        bo, version, hdr_size = ">", 2, 540
    else:
        raise NiftiParseError(f"not a NIfTI header (sizeof_hdr={n1}/{n2})")

    raw = head + _read_exact(stream, hdr_size - 4)

    if version == 1:
        dim = list(struct.unpack(bo + "8h", raw[40:56]))
        datatype = struct.unpack(bo + "h", raw[70:72])[0]
        bitpix = struct.unpack(bo + "h", raw[72:74])[0]
        pixdim = list(struct.unpack(bo + "8f", raw[76:108]))
        vox_offset = int(struct.unpack(bo + "f", raw[108:112])[0])
        scl_slope = struct.unpack(bo + "f", raw[112:116])[0]
        scl_inter = struct.unpack(bo + "f", raw[116:120])[0]
        xyzt_units = raw[123]
        descrip = raw[148:228].split(b"\x00")[0].decode("latin-1", "replace")
        qform = struct.unpack(bo + "h", raw[252:254])[0]
        sform = struct.unpack(bo + "h", raw[254:256])[0]
        qb, qc, qd = struct.unpack(bo + "3f", raw[256:268])
        qx, qy, qz = struct.unpack(bo + "3f", raw[268:280])
        sx = struct.unpack(bo + "4f", raw[280:296])
        sy = struct.unpack(bo + "4f", raw[296:312])
        sz = struct.unpack(bo + "4f", raw[312:328])
        # pixdim[0] is the qfac sign term: -1 means a left-handed voxel-to-world
        # mapping and flips the k axis. Ignoring it mirrors the volume.
        qfac = -1.0 if float(pixdim[0]) < 0 else 1.0
    else:  # NIfTI-2
        datatype = struct.unpack(bo + "h", raw[12:14])[0]
        bitpix = struct.unpack(bo + "h", raw[14:16])[0]
        dim = list(struct.unpack(bo + "8q", raw[16:80]))
        pixdim = list(struct.unpack(bo + "8d", raw[104:168]))
        vox_offset = int(struct.unpack(bo + "q", raw[168:176])[0])
        scl_slope = struct.unpack(bo + "d", raw[176:184])[0]
        scl_inter = struct.unpack(bo + "d", raw[184:192])[0]
        qform = struct.unpack(bo + "i", raw[344:348])[0]
        sform = struct.unpack(bo + "i", raw[348:352])[0]
        xyzt_units = struct.unpack(bo + "i", raw[500:504])[0]
        descrip = raw[120:200].split(b"\x00")[0].decode("latin-1", "replace")
        qb, qc, qd = struct.unpack(bo + "3d", raw[192:216])
        qx, qy, qz = struct.unpack(bo + "3d", raw[216:240])
        sx = struct.unpack(bo + "4d", raw[240:272])
        sy = struct.unpack(bo + "4d", raw[272:304])
        sz = struct.unpack(bo + "4d", raw[304:336])
        qfac = -1.0 if float(pixdim[0]) < 0 else 1.0

    ndim = int(dim[0])
    if not 1 <= ndim <= 7:
        raise NiftiParseError(f"implausible ndim={ndim}")
    if datatype not in DTYPE_CODES:
        raise NiftiParseError(f"unsupported datatype code {datatype}")

    np_dtype = bo + DTYPE_CODES[datatype][0]
    hdr = NiftiHeader(
        version=version, byteorder=bo, dim=[int(d) for d in dim[1:]], ndim=ndim,
        pixdim=[float(p) for p in pixdim[1:]], datatype_code=int(datatype),
        dtype=np_dtype, bitpix=int(bitpix), vox_offset=int(vox_offset),
        scl_slope=float(scl_slope), scl_inter=float(scl_inter),
        qform_code=int(qform), sform_code=int(sform), xyzt_units=int(xyzt_units),
        descrip=descrip,
        quatern_b=float(qb), quatern_c=float(qc), quatern_d=float(qd),
        qoffset_x=float(qx), qoffset_y=float(qy), qoffset_z=float(qz),
        qfac=float(qfac),
        srow_x=tuple(float(v) for v in sx),
        srow_y=tuple(float(v) for v in sy),
        srow_z=tuple(float(v) for v in sz),
    )
    return hdr, stream


def read_header(fileobj: BinaryIO, name: str) -> NiftiHeader:
    """Header only. Cheap: reads at most a few KB of the member."""
    stream = _open_maybe_gzip(fileobj, name)
    hdr, _ = parse_header(stream)
    return hdr


def read_array(fileobj: BinaryIO, name: str,
               max_voxels: int = 64_000_000) -> tuple[NiftiHeader, np.ndarray]:
    """Header plus the full voxel array, read sequentially.

    `max_voxels` is a guard against loading something unexpectedly large on a
    CPU-only Stage 1 runtime; exceeding it raises rather than swapping.
    """
    stream = _open_maybe_gzip(fileobj, name)
    hdr, stream = parse_header(stream)
    if hdr.n_voxels > max_voxels:
        raise NiftiParseError(
            f"{name}: {hdr.n_voxels} voxels exceeds max_voxels={max_voxels}")
    hdr_size = 348 if hdr.version == 1 else 540
    skip = max(0, hdr.vox_offset - hdr_size)
    while skip > 0:
        chunk = stream.read(min(skip, 1 << 20))
        if not chunk:
            raise NiftiParseError(f"{name}: stream ended inside vox_offset pad")
        skip -= len(chunk)
    itemsize = DTYPE_CODES[hdr.datatype_code][1]
    raw = _read_exact(stream, hdr.n_voxels * itemsize)
    arr = np.frombuffer(raw, dtype=np.dtype(hdr.dtype))
    # NIfTI is Fortran-ordered (fastest axis first).
    arr = arr.reshape(hdr.shape, order="F")
    return hdr, arr


def scaled(arr: np.ndarray, hdr: NiftiHeader) -> np.ndarray:
    """Apply scl_slope/scl_inter if the header declares a real scaling."""
    if hdr.scl_slope not in (0.0, 1.0) or hdr.scl_inter != 0.0:
        return arr.astype(np.float64) * hdr.scl_slope + hdr.scl_inter
    return arr
