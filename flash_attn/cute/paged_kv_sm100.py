"""SM100 adapter for page-local TMA and predicated boundary copies."""
import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass._mlir.dialects import nvvm, llvm
from cutlass.cutlass_dsl import dsl_user_op

from flash_attn.cute.paged_kv import PagedKVManager


@dsl_user_op
def complete_fallback_tx(ptr, nbytes, cluster, *, loc=None, ip=None):
    # cp.async.wait_group + sync_warp precede this elected-lane operation.
    # complete_tx is relaxed: publish the completed copies with a release fence
    # before decrementing the leader's byte count. Never use this for TMA copies,
    # whose byte completions are delivered by hardware.
    addr = ptr.to_llvm_ptr(loc=loc, ip=ip)
    scope = nvvm.MemScopeKind.CTA
    if cluster:
        cute.arch.fence_acq_rel_cluster(loc=loc, ip=ip)
        addr = nvvm.mapa(llvm.PointerType.get(cute.AddressSpace.dsmem), addr,
                         Int32(0).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
        scope = nvvm.MemScopeKind.CLUSTER
    else:
        cute.arch.fence_acq_rel_cta(loc=loc, ip=ip)
    nvvm.mbarrier_complete_tx(addr, Int32(nbytes).ir_value(loc=loc, ip=ip),
                             scope=scope, loc=loc, ip=ip)


class PagedKVTileLoaderSm100:
    """Consume a portable page location using the FA4 SM100 operand layouts.

    All participating CTAs take the same route, decided for the complete logical
    tile before partitioning B. The usual TMA full barrier has one arrival and
    counts bytes from both CTAs. Boundary copies complete those same bytes in
    software, leaving the common TMA path free of extra cluster arrivals.
    """

    @staticmethod
    @cute.jit
    def load(fallback: PagedKVManager, mK: cute.Tensor, mV: cute.Tensor,
             thr_mma_qk: cute.ThrMma, thr_mma_pv: cute.ThrMma,
             cta_group_size: cutlass.Constexpr[int],
             tma_atom, sX, pipeline_kv, producer_state,
             block: Int32, K_or_V: cutlass.Constexpr[str], page_idx=None):
        physical_page, page_offset, in_page = page_idx
        stage = producer_state.index
        pipeline_kv.producer_acquire(producer_state)
        if in_page:
            if const_expr(K_or_V == "K"):
                mX = cute.domain_offset((page_offset, 0), mK[None, None, physical_page])
                gX = cute.local_tile(mX, (fallback.n_block_size, fallback.head_dim_padded), (0, 0))
                tXgX = thr_mma_qk.partition_B(gX)
            else:
                mX = cute.domain_offset((0, page_offset), mV[None, None, physical_page])
                gX = cute.local_tile(mX, (fallback.head_dim_v_padded, fallback.n_block_size), (0, 0))
                tXgX = thr_mma_pv.partition_B(gX)
            tXsX, tXgX = cpasync.tma_partition(
                tma_atom, 0, cute.make_layout(1),
                cute.group_modes(sX, 0, 3), cute.group_modes(tXgX, 0, 3),
            )
            cute.copy(tma_atom, tXgX, tXsX[None, stage],
                      tma_bar_ptr=pipeline_kv.producer_get_barrier(producer_state))
        else:
            fallback.load_page_table(block)
            fallback.load_KV(block, sX[None, None, None, stage], K_or_V)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                complete_fallback_tx(
                    pipeline_kv.producer_get_barrier(producer_state),
                    pipeline_kv.sync_object_full.tx_count // cta_group_size,
                    cta_group_size == 2,
                )
