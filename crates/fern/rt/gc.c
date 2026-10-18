/* The fern runtime: a conservative mark-sweep collector over one static
   arena, plus the bump/free-list allocator that drives it.

   The runtime is lowered into the *program's* symbol table, so every
   name here is `fern_`-prefixed and the prefix is reserved in the
   language: a fern program cannot collide with the collector's
   internals. The only unprefixed names are the C library's `write` and
   `abort`.

   The heap is `fern_heap`, a zero-initialized arena that lands in .bss.
   A block is a 16-byte header -- `next` in the live list, then
   `size | mark` -- followed by its payload, which is what allocation
   returns. Blocks are 8-byte aligned and size-rounded, and their starts
   are marked in `fern_objs`, one bit per 8-byte granule, so a candidate
   pointer can be validated before any header is touched.

   Roots come from generated code: the lowering pins every pointer that
   may live across an allocation through `fern_pin`, and `fern_depth` is
   the high-water mark of pinned slots over the frames on the stack. A
   frame is a cookie -- the depth at entry -- and `fern_leave` restores
   it before the frame returns.

   Collection is conservative: every 8-byte word of a payload is a
   candidate pointer, validated against the start bitmap. Words that are
   not pointers can only retain, never corrupt; that is the price of not
   carrying type descriptors in cells.

   `fern_alloc` prefers recycled blocks, falls back to the bump pointer,
   runs a collection when it crosses `fern_threshold`, and aborts when
   the arena is genuinely full. */

/* LP64: pointers, words, and `unsigned long` are all 64 bits. */
typedef unsigned long u64;

#define HEAP_BYTES (1UL << 26)
#define HEADER 16UL
#define MIN_BLOCK 32UL
/* Bytes of heap in use before the first collection. */
#define INITIAL_THRESHOLD (1UL << 22)
/* Bump headroom granted past the live set after each collection. */
#define GROW_STEP (1UL << 20)
/* Root-array capacity, in slots; pinning past it aborts. */
#define ROOTS (1UL << 18)
/* Worklist capacity, in slots; overflow falls back to a re-scan. */
#define MARK_STACK (1UL << 19)

#define ALIGN8(n) (((n) + 7UL) & ~7UL)
#define MARKED(w) ((w) & 1UL)
#define SIZE_OF(w) ((w) & ~7UL)

typedef struct block {
	struct block *next;
	u64 size;
} block;

static u64 fern_heap[HEAP_BYTES / 8];
/* One start bit per 8-byte granule. */
static u64 fern_objs[(HEAP_BYTES / 8 + 63) / 64];
static u64 fern_roots[ROOTS];
static u64 fern_work[MARK_STACK];

static block *fern_live;
static block *fern_free;
static u64 fern_depth;
static u64 fern_bump;
static u64 fern_threshold = INITIAL_THRESHOLD;
static u64 fern_live_bytes;
static u64 fern_marks;
static u64 fern_work_sp;
static u64 fern_overflowed;
static u64 fern_collections;

long write(int fd, const void *buf, unsigned long count);
void abort(void);

static const char fern_oom_msg[] = "fern: heap full\n";
static const char fern_root_msg[] = "fern: root overflow\n";

static void fern_die(const char *msg, u64 len) {
	write(2, msg, len);
	abort();
}

/* ------------------------------------------------------------------ */
/* block starts                                                        */

static void fern_mark_start(u64 addr) {
	u64 i = (addr - (u64)fern_heap) >> 3;
	fern_objs[i >> 6] |= 1UL << (i & 63);
}

static void fern_clear_start(u64 addr) {
	u64 i = (addr - (u64)fern_heap) >> 3;
	fern_objs[i >> 6] &= ~(1UL << (i & 63));
}

static int fern_is_start(u64 addr) {
	u64 i = (addr - (u64)fern_heap) >> 3;
	return (int)((fern_objs[i >> 6] >> (i & 63)) & 1UL);
}

/* ------------------------------------------------------------------ */
/* marking                                                             */

/* Mark and queue `p`'s block when `p` is a payload pointer to a
   current block start. */
static void fern_try_mark(u64 p) {
	block *b;
	if (p < (u64)fern_heap + HEADER || p >= (u64)fern_heap + HEAP_BYTES) return;
	b = (block *)(p - HEADER);
	if (!fern_is_start((u64)b)) return;
	if (MARKED(b->size)) return;
	b->size |= 1UL;
	fern_marks++;
	if (fern_work_sp < MARK_STACK) fern_work[fern_work_sp++] = p;
	else fern_overflowed = 1;
}

/* Try to mark every word of one marked block. */
static void fern_scan(u64 p) {
	block *b = (block *)(p - HEADER);
	u64 *at = (u64 *)p;
	u64 *end = (u64 *)((u64)b + SIZE_OF(b->size));
	while (at < end) fern_try_mark(*at++);
}

static void fern_drain(void) {
	while (fern_work_sp > 0) fern_scan(fern_work[--fern_work_sp]);
}

/* Unlink unmarked blocks into the free list, clearing marks on the
   survivors so the next collection starts clean. */
static void fern_sweep(void) {
	block **link = &fern_live;
	block *b = fern_live;
	while (b != 0) {
		block *next = b->next;
		if (MARKED(b->size)) {
			b->size &= ~1UL;
			link = &b->next;
		} else {
			*link = next;
			fern_clear_start((u64)b);
			fern_live_bytes -= SIZE_OF(b->size);
			b->next = fern_free;
			fern_free = b;
		}
		b = next;
	}
}

static void fern_gc(void) {
	u64 i;
	fern_collections++;
	fern_work_sp = 0;
	fern_overflowed = 0;
	for (i = 0; i < fern_depth; i++) fern_try_mark(fern_roots[i]);
	fern_drain();
	/* A full worklist left marked blocks unqueued: scan the whole live
	   list until a pass through it marks nothing new. */
	if (fern_overflowed) {
		for (;;) {
			block *b;
			fern_overflowed = 0;
			fern_marks = 0;
			for (b = fern_live; b != 0; b = b->next)
				if (MARKED(b->size)) fern_scan((u64)b + HEADER);
			fern_drain();
			if (fern_marks == 0) break;
		}
	}
	fern_sweep();
	fern_threshold = fern_live_bytes * 2 + GROW_STEP;
	if (fern_threshold < fern_bump + GROW_STEP) fern_threshold = fern_bump + GROW_STEP;
}

/* ------------------------------------------------------------------ */
/* allocation                                                          */

/* First fit; the tail of a split block goes back to the free list. */
static u64 fern_find_free(u64 total) {
	block **link = &fern_free;
	block *b = *link;
	while (b != 0) {
		u64 size = b->size;
		if (size >= total) {
			block *rest;
			*link = b->next;
			if (size - total >= MIN_BLOCK) {
				rest = (block *)((u64)b + total);
				rest->size = size - total;
				rest->next = fern_free;
				fern_free = rest;
				fern_mark_start((u64)rest);
			}
			b->size = total;
			b->next = fern_live;
			fern_live = b;
			fern_mark_start((u64)b);
			return (u64)b + HEADER;
		}
		link = &b->next;
		b = *link;
	}
	return 0;
}

void *fern_alloc(u64 size) {
	u64 total = ALIGN8(size) + HEADER;
	u64 got;
	int collected = 0;
	if (total < MIN_BLOCK) total = MIN_BLOCK;
	for (;;) {
		got = fern_find_free(total);
		if (got != 0) break;
		if (fern_bump + total <= HEAP_BYTES && fern_bump < fern_threshold) {
			block *b = (block *)((u64)fern_heap + fern_bump);
			fern_bump += total;
			b->next = fern_live;
			fern_live = b;
			b->size = total;
			fern_mark_start((u64)b);
			fern_live_bytes += total;
			return (void *)((u64)b + HEADER);
		}
		if (collected) fern_die(fern_oom_msg, sizeof fern_oom_msg - 1);
		fern_gc();
		collected = 1;
	}
	fern_live_bytes += total;
	return (void *)got;
}

/* ------------------------------------------------------------------ */
/* generated-code interface                                            */

/* The frame cookie: where this body's root slots start. */
u64 fern_enter(void) {
	return fern_depth;
}

/* Publish `p` as root `slot` of `frame`; a null pins a slot dark. */
void fern_pin(u64 frame, u64 slot, void *p) {
	u64 at = frame + slot;
	if (at >= ROOTS) fern_die(fern_root_msg, sizeof fern_root_msg - 1);
	fern_roots[at] = (u64)p;
	if (at >= fern_depth) fern_depth = at + 1;
}

void fern_leave(u64 frame) {
	fern_depth = frame;
}

/* ------------------------------------------------------------------ */
/* hooks for the runtime's own tests                                   */

void fern_collect(void) {
	fern_gc();
}

u64 fern_stats_live(void) {
	return fern_live_bytes;
}

u64 fern_stats_used(void) {
	return fern_bump;
}

u64 fern_stats_roots(void) {
	return fern_depth;
}

u64 fern_stats_collections(void) {
	return fern_collections;
}
