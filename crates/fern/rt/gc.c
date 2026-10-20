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

/* ------------------------------------------------------------------ */
/* strings                                                             */

static u64 fern_strlen(const char *s) {
	u64 n = 0;
	while (s[n] != 0) n++;
	return n;
}

long fern_print(const char *s) {
	write(1, s, fern_strlen(s));
	return 0;
}

char *fern_str_concat(const char *a, const char *b) {
	u64 na = fern_strlen(a);
	u64 nb = fern_strlen(b);
	char *out = (char *)fern_alloc(na + nb + 1);
	u64 i;
	for (i = 0; i < na; i++) out[i] = a[i];
	for (i = 0; i < nb; i++) out[na + i] = b[i];
	out[na + nb] = 0;
	return out;
}

/* A typed hole in a shell word: `shell_arg` constrains `${expr}` to a
 * string without copying it. */
char *fern_str_identity(char *s) {
	return s;
}

char fern_str_eq(const char *a, const char *b) {
	u64 i = 0;
	while (a[i] != 0 && b[i] != 0) {
		if (a[i] != b[i]) return 0;
		i++;
	}
	return a[i] == b[i];
}

long fern_str_len(const char *s) {
	return (long)fern_strlen(s);
}

/* Byte offset of `needle` in `s`, or -1. */
long fern_str_find(const char *s, const char *needle) {
	u64 n = fern_strlen(needle);
	if (n == 0) return 0;
	u64 i = 0;
	while (s[i] != 0) {
		u64 j = 0;
		while (j < n && s[i + j] != 0 && s[i + j] == needle[j]) j++;
		if (j == n) return (long)i;
		i++;
	}
	return -1;
}

/* A fresh GC string holding up to `len` bytes from `start`. */
char *fern_str_slice(const char *s, long start, long len) {
	u64 n = fern_strlen(s);
	u64 begin = start < 0 ? 0 : (u64)start;
	u64 take = len < 0 ? 0 : (u64)len;
	if (begin > n) begin = n;
	if (take > n - begin) take = n - begin;
	char *out = (char *)fern_alloc(take + 1);
	u64 i;
	for (i = 0; i < take; i++) out[i] = s[begin + i];
	out[take] = 0;
	return out;
}

/* A fresh GC string holding one byte. Byte zero is the empty string
   because fern strings are NUL-terminated; callers that need to display
   a zero byte should render it separately. */
char *fern_str_of_byte(long b) {
	char *out = (char *)fern_alloc(2);
	out[0] = (char)(b & 0xff);
	out[1] = 0;
	return out;
}

/* Decimal rendering of a signed 64-bit value into a GC string. */
char *fern_str_of_int(long v) {
	char *out = (char *)fern_alloc(32);
	char tmp[32];
	int n = 0;
	int neg = v < 0;
	unsigned long u;
	if (neg) u = (unsigned long)(-(v + 1)) + 1;
	else u = (unsigned long)v;
	if (u == 0) tmp[n++] = '0';
	while (u != 0) {
		tmp[n++] = (char)('0' + (int)(u % 10));
		u = u / 10;
	}
	if (neg) tmp[n++] = '-';
	int i = 0;
	while (n > 0) out[i++] = tmp[--n];
	out[i] = 0;
	return out;
}

char *getenv(const char *name);

char *fern_getenv(const char *name) {
	char *v = getenv(name);
	if (v == 0) return "";
	return v;
}

int chdir(const char *path);
int setenv(const char *name, const char *value, int overwrite);

long fern_chdir(const char *path) {
	return (long)chdir(path);
}

long fern_setenv(const char *name, const char *value) {
	return (long)setenv(name, value, 1);
}

void _exit(int code);

long fern_exit(long code) {
	_exit((int)code);
	return 0;
}

/* ------------------------------------------------------------------ */
/* processes                                                           */

int fork(void);
int execvp(const char *file, char **argv);
int waitpid(int pid, int *status, int options);
int pipe(int *fds);
int dup2(int oldfd, int newfd);
int open(const char *path, int flags);
int close(int fd);
long read(int fd, void *buf, unsigned long count);
int kill(int pid, int sig);
void *malloc(unsigned long size);
void free(void *p);

/* A linked argv list built by the prelude from `List String`. */
typedef struct fargv {
	char *s;
	struct fargv *next;
} fargv;

fargv *fern_argv_nil(long unused) {
	(void)unused;
	return 0;
}

fargv *fern_argv_cons(char *s, fargv *next) {
	fargv *node = (fargv *)fern_alloc(sizeof(fargv));
	node->s = s;
	node->next = next;
	return node;
}

static u64 fern_argv_count(fargv *a) {
	u64 n = 0;
	while (a != 0) {
		n++;
		a = a->next;
	}
	return n;
}

/* A NULL-terminated `char **` with the program name at index 0. */
static char **fern_argv_array(char *name, fargv *args) {
	u64 n = fern_argv_count(args);
	char **out = (char **)fern_alloc((n + 2) * sizeof(char *));
	u64 i = 0;
	out[0] = name;
	while (args != 0) {
		i++;
		out[i] = args->s;
		args = args->next;
	}
	out[n + 1] = 0;
	return out;
}

/* Wait status as a shell would report it: exit code, or 128 + signal. */
static long fern_status_of(int status) {
	if ((status & 0x7f) == 0) return (status >> 8) & 0xff;
	return 128 + (status & 0x7f);
}

long fern_spawn_status(char *name, fargv *args) {
	char **argv = fern_argv_array(name, args);
	int pid = fork();
	if (pid == 0) {
		execvp(name, argv);
		_exit(127);
	}
	if (pid < 0) return -1;
	int status = 0;
	if (waitpid(pid, &status, 0) < 0) return -1;
	return fern_status_of(status);
}

/* A regular file's contents, truncated at 64 KiB. A missing file reads
   as the empty string. */
char *fern_read_file(const char *path) {
	int fd = open(path, 0);
	if (fd < 0) return "";
	u64 cap = 65536;
	char *out = (char *)fern_alloc(cap);
	u64 len = 0;
	for (;;) {
		long got = read(fd, out + len, cap - 1 - len);
		if (got <= 0) break;
		len += (u64)got;
		if (len >= cap - 1) break;
	}
	close(fd);
	out[len] = 0;
	return out;
}

/* Captured stdout, truncated at 64 KiB. */
char *fern_spawn_capture(char *name, fargv *args) {
	char **argv = fern_argv_array(name, args);
	int fds[2];
	if (pipe(fds) != 0) return "";
	int pid = fork();
	if (pid == 0) {
		close(fds[0]);
		dup2(fds[1], 1);
		close(fds[1]);
		execvp(name, argv);
		_exit(127);
	}
	if (pid < 0) {
		close(fds[0]);
		close(fds[1]);
		return "";
	}
	close(fds[1]);
	u64 cap = 65536;
	char *out = (char *)fern_alloc(cap);
	u64 len = 0;
	for (;;) {
		long got = read(fds[0], out + len, cap - 1 - len);
		if (got <= 0) break;
		len += (u64)got;
		if (len >= cap - 1) break;
	}
	close(fds[0]);
	out[len] = 0;
	int status = 0;
	waitpid(pid, &status, 0);
	return out;
}

/* ------------------------------------------------------------------ */
/* pipelines                                                           */

typedef struct fpipe {
	fargv *argv;
	struct fpipe *next;
} fpipe;

fpipe *fern_pipe_nil(long unused) {
	(void)unused;
	return 0;
}

fpipe *fern_pipe_cons(fargv *argv, fpipe *next) {
	fpipe *node = (fpipe *)fern_alloc(sizeof(fpipe));
	node->argv = argv;
	node->next = next;
	return node;
}

static char **fern_argv_vec(fargv *args) {
	u64 n = fern_argv_count(args);
	char **out = (char **)malloc((n + 1) * sizeof(char *));
	u64 i = 0;
	if (out == 0) return 0;
	while (args != 0) {
		out[i] = args->s;
		i++;
		args = args->next;
	}
	out[n] = 0;
	return out;
}

static void fern_argv_vecs_free(char ***argvs, u64 n) {
	u64 i;
	if (argvs == 0) return;
	for (i = 0; i < n; i++) free(argvs[i]);
	free(argvs);
}

static long fern_pipeline_wait(int *pids, u64 n) {
	long last = -1;
	u64 i;
	for (i = 0; i < n; i++) {
		int status = 0;
		if (waitpid(pids[i], &status, 0) >= 0) last = fern_status_of(status);
	}
	return last;
}

static void fern_pipeline_close(int *fds, u64 nfds, int *out_fds) {
	u64 i;
	for (i = 0; i < nfds; i++) {
		if (fds[i] >= 0) close(fds[i]);
	}
	if (out_fds[0] >= 0) close(out_fds[0]);
	if (out_fds[1] >= 0) close(out_fds[1]);
}

/* Run the stages with real OS pipes. When `out` is non-NULL the last
 * stage's stdout is captured (truncated at `cap` bytes). */
static long fern_pipeline_run(fpipe *stages, char *out, u64 cap) {
	u64 n = 0;
	u64 connectors;
	u64 nfds;
	u64 i;
	u64 k;
	fpipe *p;
	int *fds = 0;
	int *pids;
	char ***argvs;
	int out_fds[2];
	u64 started = 0;

	p = stages;
	while (p != 0) {
		n++;
		p = p->next;
	}
	if (n == 0) {
		if (out != 0 && cap > 0) out[0] = 0;
		return -1;
	}
	connectors = n - 1;
	nfds = connectors * 2;
	if (nfds > 0) {
		fds = (int *)malloc(nfds * sizeof(int));
		if (fds == 0) return -1;
		for (i = 0; i < nfds; i++) fds[i] = -1;
	}
	pids = (int *)malloc(n * sizeof(int));
	if (pids == 0) {
		free(fds);
		return -1;
	}
	argvs = (char ***)malloc(n * sizeof(char **));
	if (argvs == 0) {
		free(pids);
		free(fds);
		return -1;
	}
	p = stages;
	for (i = 0; i < n; i++) {
		argvs[i] = p == 0 ? 0 : fern_argv_vec(p->argv);
		if (argvs[i] == 0 || argvs[i][0] == 0) {
			fern_argv_vecs_free(argvs, i + 1);
			free(pids);
			free(fds);
			return -1;
		}
		p = p->next;
	}
	out_fds[0] = -1;
	out_fds[1] = -1;
	if (out != 0 && pipe(out_fds) != 0) {
		fern_argv_vecs_free(argvs, n);
		free(pids);
		free(fds);
		return -1;
	}

	for (i = 0; i < n; i++) {
		int pid;
		if (i + 1 < n) {
			int pair[2];
			if (pipe(pair) != 0) {
				fern_pipeline_close(fds, nfds, out_fds);
				for (k = 0; k < started; k++) kill(pids[k], 15);
				fern_pipeline_wait(pids, started);
				fern_argv_vecs_free(argvs, n);
				free(pids);
				free(fds);
				if (out != 0 && cap > 0) out[0] = 0;
				return -1;
			}
			fds[2 * i] = pair[0];
			fds[2 * i + 1] = pair[1];
		}
		pid = fork();
		if (pid == 0) {
			char **argv;
			if (i > 0) dup2(fds[2 * (i - 1)], 0);
			if (i + 1 < n) {
				dup2(fds[2 * i + 1], 1);
			} else if (out_fds[1] >= 0) {
				dup2(out_fds[1], 1);
			}
			fern_pipeline_close(fds, nfds, out_fds);
			argv = argvs[i];
			if (argv == 0 || argv[0] == 0) _exit(127);
			execvp(argv[0], argv);
			_exit(127);
		}
		if (pid < 0) {
			fern_pipeline_close(fds, nfds, out_fds);
			for (k = 0; k < started; k++) kill(pids[k], 15);
			fern_pipeline_wait(pids, started);
			fern_argv_vecs_free(argvs, n);
			free(pids);
			free(fds);
			if (out != 0 && cap > 0) out[0] = 0;
			return -1;
		}
		pids[i] = pid;
		started = i + 1;
		if (i > 0) close(fds[2 * (i - 1)]);
		if (i + 1 < n) close(fds[2 * i + 1]);
	}

	if (out != 0) {
		u64 len = 0;
		if (out_fds[1] >= 0) {
			close(out_fds[1]);
			out_fds[1] = -1;
		}
		for (;;) {
			long got = read(out_fds[0], out + len, cap - 1 - len);
			if (got <= 0) break;
			len += (u64)got;
			if (len >= cap - 1) break;
		}
		close(out_fds[0]);
		out_fds[0] = -1;
		out[len] = 0;
	}

	long status = fern_pipeline_wait(pids, n);
	fern_argv_vecs_free(argvs, n);
	free(pids);
	free(fds);
	return status;
}

long fern_pipeline_status(fpipe *stages) {
	return fern_pipeline_run(stages, 0, 0);
}

char *fern_pipeline_capture(fpipe *stages) {
	u64 cap = 65536;
	u64 i = 0;
	char *buf = (char *)malloc(cap);
	char *out;
	if (buf == 0) return "";
	buf[0] = 0;
	fern_pipeline_run(stages, buf, cap);
	out = (char *)fern_alloc(fern_strlen(buf) + 1);
	while (buf[i] != 0) {
		out[i] = buf[i];
		i++;
	}
	out[i] = 0;
	free(buf);
	return out;
}
