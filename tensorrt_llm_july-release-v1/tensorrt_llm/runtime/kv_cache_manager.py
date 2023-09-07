import math
from collections import defaultdict
from typing import List

import torch


class Block(object):

    def __init__(self, block_idx, k_ptrs, v_ptrs):
        self.idx = block_idx
        self.k_ptrs = k_ptrs
        self.v_ptrs = v_ptrs
        self.ref_count = 0

    def add_link(self):
        self.ref_count += 1

    def remove_link(self):
        self.ref_count -= 1

    def has_link(self) -> bool:
        return self.ref_count > 0

    def get_k_ptr(self, idx) -> int:
        return self.k_ptrs[idx]

    def get_v_ptr(self, idx) -> int:
        return self.v_ptrs[idx]


class GenerationSequence(object):

    def __init__(self, seq_idx, batch_idx):
        self.seq_idx = seq_idx
        self.batch_idx = batch_idx

    def get_batch_idx(self) -> int:
        """
        Returns idx of sequence in batch
        """
        return self.batch_idx

    def get_seq_idx(self) -> int:
        """
        Returns sequence idx
        """
        return self.seq_idx

    def __eq__(self, another):
        return hasattr(another, 'seq_idx') and self.seq_idx == another.seq_idx and \
            hasattr(another, 'batch_idx') and self.batch_idx == another.batch_idx

    def __hash__(self):
        return self.seq_idx


class BlocksManager(object):
    _sizeof = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1
    }

    def __init__(self,
                 memory_pools: List[torch.Tensor],
                 blocks: int,
                 max_blocks_per_seq: int = 128,
                 beam_width: int = 1):
        self.max_blocks_per_seq = max_blocks_per_seq

        self.pointer_array = None
        self.memory_pools = memory_pools
        self.blocks = blocks
        self.beam_width = beam_width

        self.elts_per_blocks = []
        for pool in memory_pools:
            # Pool consists of memory for K and V caches
            self.elts_per_blocks.append(pool.nelement() // (2 * blocks))

        self.free_blocks = []
        for bi in range(blocks):
            k_ptrs = []
            v_ptrs = []
            for pool, elts_per_block in zip(memory_pools, self.elts_per_blocks):
                k_ptrs.append(self.get_mempool_pointer(bi, pool,
                                                       elts_per_block))
                v_ptrs.append(
                    self.get_mempool_pointer(bi, pool, elts_per_block) +
                    self.blocks * elts_per_block * self._sizeof[pool.dtype])
            self.free_blocks.append(Block(bi, k_ptrs, v_ptrs))

        self.allocated_blocks = defaultdict(
            lambda: [[] for _ in range(self.beam_width)])

    def has_free_block(self) -> bool:
        """
        Returns True if we have at least 1 free block
        """
        return len(self.free_blocks) > 0

    def allocate(self,
                 owner: GenerationSequence,
                 share_across_beam: bool = False):
        """
        Add block to owner and increase ref count
        """
        # Add blocks for whole beam width
        block = None
        for bi in range(self.beam_width):
            if not self.has_free_block():
                raise RuntimeError("Can't allocate new block for KV cache")

            # Use the same block for all seqs in beam if share_across_beam
            if block is None or share_across_beam == False:
                block = self.free_blocks.pop(0)
            # Add one reference to the block
            block.add_link()
            self.allocated_blocks[owner][bi].append(block)

    def free(self, owner: GenerationSequence):
        """
        Unlink all blocks of given owner.
        Moves blocks with ref_count == 0 to free.
        Removes owner from allocated blocks.
        """
        for bi in range(self.beam_width):
            for block in self.allocated_blocks[owner][bi]:
                # Move block to free if no one refers to it
                block.remove_link()

                # Move block to free if no one refers to it
                if not block.has_link():
                    self.free_blocks.append(block)
        # Remove owner from allocated blocks
        self.allocated_blocks.pop(owner)

    def get_number_blocks(self, owner: GenerationSequence) -> int:
        """
        Returns number of blocks allocated to the sequence owner
        """
        return len(self.allocated_blocks[owner][0])

    def get_mempool_pointer(self, block_idx: int, pool: torch.Tensor,
                            elts_per_block: int) -> int:
        """
        Computes linear pointer
        """
        return pool.data_ptr(
        ) + block_idx * elts_per_block * self._sizeof[pool.dtype]

    def get_pointer_array(self, pool_idx: int) -> torch.Tensor:
        """
        Returns array of [batch size, beam_width, 2, max_blocks_per_seq] of poitners
        to the allocated blocks in memory pool
        """
        self.pointer_array = torch.zeros(len(self.allocated_blocks),
                                         self.beam_width,
                                         2,
                                         self.max_blocks_per_seq,
                                         dtype=torch.int64)

        for owner, beams_blocks in self.allocated_blocks.items():
            for bi in range(self.beam_width):
                for block_linear_idx, block in enumerate(beams_blocks[bi]):
                    # K cache pointers
                    self.pointer_array[owner.get_batch_idx(
                    )][bi][0][block_linear_idx] = block.get_k_ptr(pool_idx)
                    # V cache pointers
                    self.pointer_array[owner.get_batch_idx(
                    )][bi][1][block_linear_idx] = block.get_v_ptr(pool_idx)

        return self.pointer_array

    def get_continous_caches(self, pool_idx: int) -> torch.Tensor:
        """
        Returns countinous KV caches.
        Used only for debug purposes.
        """
        assert self.beam_width == 1

        elts_per_block = self.elts_per_blocks[pool_idx]
        pool = self.memory_pools[pool_idx].flatten()
        continous_kv_cache = torch.zeros(len(self.allocated_blocks),
                                         2,
                                         self.max_blocks_per_seq *
                                         elts_per_block,
                                         dtype=pool.dtype,
                                         device="cuda")
        for owner, beam_blocks in self.allocated_blocks.items():
            for bi in range(self.beam_width):
                for block_linear_idx, block in enumerate(beam_blocks[bi]):
                    # The batch index.
                    batch_idx = owner.get_batch_idx()
                    # The first index in the sequence.
                    block_offset = block_linear_idx * elts_per_block
                    # The first index in the pool for K.
                    k_start = block.idx * elts_per_block
                    # The first index in the pool for V.
                    v_start = k_start + self.blocks * elts_per_block

                    continous_kv_cache[batch_idx][0][
                        block_offset:block_offset +
                        elts_per_block] = pool[k_start:k_start + elts_per_block]
                    continous_kv_cache[batch_idx][1][
                        block_offset:block_offset +
                        elts_per_block] = pool[v_start:v_start + elts_per_block]

        return continous_kv_cache


class KVCacheManager(object):

    def __init__(self,
                 memory_pools: List[torch.Tensor],
                 blocks: int,
                 tokens_per_block: int,
                 max_blocks_per_seq: int,
                 beam_width: int = 1):
        self.blocks_manager = BlocksManager(
            memory_pools=memory_pools,
            blocks=blocks,
            max_blocks_per_seq=max_blocks_per_seq,
            beam_width=beam_width)
        self.num_pools = len(memory_pools)
        self.tokens_per_block = tokens_per_block
        self.beam_width = beam_width

        self.lens = []
        self.sequences = []

    def step(self, finished: List[bool]):
        """
        Iterate to the next generation step.
        Add new blocks where needed and clear finished sequences.
        """
        for seq in self.sequences:
            batch_idx = seq.get_batch_idx()
            if not finished[batch_idx] and self.lens[
                    batch_idx] % self.tokens_per_block == self.tokens_per_block - 1:
                self.blocks_manager.allocate(seq)

            self.lens[batch_idx] += 1

        # Remove finished sequences
        for fi in range(len(finished)):
            if finished[fi]:
                self.blocks_manager.free(self.sequences[fi])
        self.lens = [l for l, f in zip(self.lens, finished) if not f]

        # Remap sequence ids
        new_sequences = []
        batch_idx = 0
        for seq, finish in zip(self.sequences, finished):
            if not finish:
                seq.batch_idx = batch_idx
                new_sequences.append(seq)
                batch_idx += 1
        self.sequences = new_sequences

    def add_sequence(self, sequence: GenerationSequence, context_len: int):
        """
        Add sequence to the manager and allocate minimum amount of blocks for context
        """
        self.lens.append(context_len)
        self.sequences.append(sequence)

        # With beam_width > 1 we share context blocks between beams. If context len is not
        # a multiple of tokens per block, several beams will try to write to the shared block during the generation phase
        # FIXME(nkorobov) copy the last block to the individual beams after the context phase
        assert self.beam_width == 1 or (self.beam_width > 1 and context_len %
                                        self.tokens_per_block == 0)

        # Allocate one more token for the 1st generation step
        context_blocks = math.ceil((context_len + 1) / self.tokens_per_block)
        for _ in range(context_blocks):
            # Share context stage blocks within beam
            self.blocks_manager.allocate(sequence, share_across_beam=True)

    def get_pointer_arrays(self) -> List[torch.Tensor]:
        """
        Returns arrays of pointers for all memory pools copied to GPU
        """
        pointer_arrays = []
        for pool in range(self.num_pools):
            # View to int32 is hack to overcome absence of int64 dtype in TRT
            pointer_arrays.append(
                self.blocks_manager.get_pointer_array(pool).to('cuda').view(
                    dtype=torch.int32))
        return pointer_arrays
