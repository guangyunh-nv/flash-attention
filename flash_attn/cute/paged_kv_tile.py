"""Page-coordinate iteration independent of attention layout or GPU architecture."""
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute import FastDivmodDivisorV2


class PagedKVTileIterator:
    """Random-access tile iterator, suitable for forward or reverse traversal.

    A caller supplies a logical token interval and a single sequence's page
    table. The returned (physical_page, offset, in_page) location is reusable
    across K/V and CTA partitions. Only complete, valid, single-page intervals
    select the in-page route. Crossings and tails select a predicated fallback;
    their physical_page is a sentinel and must not be dereferenced.

    There are no MMA, TMA, shared-memory, barrier, CTA-count, K/V-axis, or SM100
    dependencies. A backend adapter decides how to load the returned interval.
    Explicit arguments avoid binding dynamic CuTe IR values into Python state.
    Precondition: nonnegative tile_start and positive page/tile sizes; page table
    entries cover sequence_length. page_size_divmod must describe page_size.
    """

    @staticmethod
    @cute.jit
    def locate(page_table: cute.Tensor, page_size_divmod: FastDivmodDivisorV2,
               page_size: Int32, sequence_length: Int32, tile_start: Int32,
               tile_size: cutlass.Constexpr[int]):
        logical_page, offset = divmod(tile_start, page_size_divmod)
        in_page = (offset + tile_size <= page_size
                   and tile_start + tile_size <= sequence_length)
        physical_page = Int32(0)
        if in_page:
            physical_page = page_table[logical_page]
        return physical_page, offset, in_page
