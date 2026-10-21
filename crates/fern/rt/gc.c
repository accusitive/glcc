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

/* Fern Int and pointer-sized values are always 64 bits. `long` is
   32-bit on Windows (LLP64), so spell the width explicitly. */
typedef unsigned long long u64;
typedef long long i64;

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

/* Coroutine frames are malloc'd, so the collector owns their lifetime
   separately from heap cells. The payload is scanned conservatively;
   spilled pointers survive suspension that way. */
typedef struct frame {
	struct frame *next;
	void *raw;
	void *payload;
	u64 size;
} frame;

typedef struct {
	void *frame;
	int (*poll)(void *, void *);
} fern_future;

typedef struct {
	int woken;
} fern_ctx;

static u64 fern_heap[HEAP_BYTES / 8];
/* One start bit per 8-byte granule. */
static u64 fern_objs[(HEAP_BYTES / 8 + 63) / 64];
static u64 fern_roots[ROOTS];
static u64 fern_work[MARK_STACK];

static block *fern_live;
static block *fern_free;
static frame *fern_frames;
static u64 fern_depth;
static u64 fern_bump;
static u64 fern_threshold = INITIAL_THRESHOLD;
static u64 fern_live_bytes;
static u64 fern_marks;
static u64 fern_work_sp;
static u64 fern_overflowed;
static u64 fern_collections;

#ifdef _WIN32
int _write(int fd, const void *buf, unsigned int count);
#else
i64 write(int fd, const void *buf, u64 count);
#endif
void abort(void);
void *malloc(u64 size);
void free(void *p);
void *realloc(void *p, u64 size);
void *fopen(const char *path, const char *mode);
u64 fread(void *ptr, u64 size, u64 count, void *stream);
u64 fwrite(const void *ptr, u64 size, u64 count, void *stream);
int fclose(void *stream);

static const char fern_oom_msg[] = "fern: heap full\n";
static const char fern_root_msg[] = "fern: root overflow\n";

static i64 fern_os_write(int fd, const void *buf, u64 count) {
#ifdef _WIN32
	return (i64)_write(fd, buf, (unsigned int)count);
#else
	return write(fd, buf, count);
#endif
}

static void fern_die(const char *msg, u64 len) {
	fern_os_write(2, msg, len);
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

static void fern_scan_frames(void) {
	frame *f = fern_frames;
	while (f != 0) {
		u64 *at = (u64 *)f->payload;
		u64 n = (f->size + 7) / 8;
		while (n > 0) {
			fern_try_mark(*at);
			at++;
			n--;
		}
		f = f->next;
	}
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
	fern_scan_frames();
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
/* coroutine frames and the blocking driver                            */

void *__glcc_frame_alloc(u64 size, u64 align) {
	u64 a = align < 8 ? 8 : align;
	u64 total = size + sizeof(frame) + a + 8;
	char *raw = (char *)malloc(total);
	frame *f;
	u64 start;
	if (raw == 0) abort();
	f = (frame *)raw;
	start = ((u64)raw + sizeof(frame) + a - 1) & ~(a - 1);
	f->raw = raw;
	f->payload = (void *)start;
	f->size = size;
	f->next = fern_frames;
	fern_frames = f;
	return f->payload;
}

void __glcc_frame_free(void *payload) {
	frame **link = &fern_frames;
	while (*link != 0) {
		frame *f = *link;
		if (f->payload == payload) {
			*link = f->next;
			free(f->raw);
			return;
		}
		link = &f->next;
	}
}

static void fern_idle(void);

static void fern_wait(fern_future f) {
	fern_ctx ctx;
	int status;
	ctx.woken = 0;
	status = f.poll(f.frame, &ctx);
	while (status == 0) {
		fern_idle();
		ctx.woken = 0;
		status = f.poll(f.frame, &ctx);
	}
}

i64 fern_block_on_int(fern_future f) {
	i64 out;
	fern_wait(f);
	out = *(i64 *)f.frame;
	__glcc_frame_free(f.frame);
	return out;
}

double fern_block_on_float(fern_future f) {
	double out;
	fern_wait(f);
	out = *(double *)f.frame;
	__glcc_frame_free(f.frame);
	return out;
}

char fern_block_on_bool(fern_future f) {
	char out;
	fern_wait(f);
	out = *(char *)f.frame;
	__glcc_frame_free(f.frame);
	return out;
}

char *fern_block_on_string(fern_future f) {
	char *out;
	fern_wait(f);
	out = *(char **)f.frame;
	__glcc_frame_free(f.frame);
	return out;
}

char fern_block_on_unit(fern_future f) {
	fern_wait(f);
	__glcc_frame_free(f.frame);
	return 0;
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

i64 fern_print(const char *s) {
	fern_os_write(1, s, fern_strlen(s));
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

i64 fern_str_len(const char *s) {
	return (i64)fern_strlen(s);
}

/* Byte offset of `needle` in `s`, or -1. */
i64 fern_str_find(const char *s, const char *needle) {
	u64 n = fern_strlen(needle);
	if (n == 0) return 0;
	u64 i = 0;
	while (s[i] != 0) {
		u64 j = 0;
		while (j < n && s[i + j] != 0 && s[i + j] == needle[j]) j++;
		if (j == n) return (i64)i;
		i++;
	}
	return -1;
}

/* A fresh GC string holding up to `len` bytes from `start`. */
char *fern_str_slice(const char *s, i64 start, i64 len) {
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
char *fern_str_of_byte(i64 b) {
	char *out = (char *)fern_alloc(2);
	out[0] = (char)(b & 0xff);
	out[1] = 0;
	return out;
}

/* Decimal rendering of a signed 64-bit value into a GC string. */
char *fern_str_of_int(i64 v) {
	char *out = (char *)fern_alloc(32);
	char tmp[32];
	int n = 0;
	int neg = v < 0;
	u64 u;
	if (neg) u = (u64)(-(v + 1)) + 1;
	else u = (u64)v;
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

#ifdef _WIN32
int _chdir(const char *path);
int _putenv_s(const char *name, const char *value);
int _write(int fd, const void *buf, unsigned int count);
int _read(int fd, void *buf, unsigned int count);
int _open(const char *path, int flags, ...);
int _close(int fd);
int _dup(int fd);
int _dup2(int oldfd, int newfd);
int _pipe(int *pfds, unsigned int psize, int textmode);
long long _lseeki64(int fd, long long offset, int origin);
char *_tempnam(const char *dir, const char *prefix);
int remove(const char *path);
typedef long long fern_intptr;
fern_intptr _spawnvp(int mode, const char *cmdname, char *const *argv);
int _cwait(int *status, fern_intptr process, int action);
int system(const char *command);
void *_popen(const char *command, const char *mode);
int _pclose(void *stream);
#else
int chdir(const char *path);
int setenv(const char *name, const char *value, int overwrite);
i64 write(int fd, const void *buf, u64 count);
int open(const char *path, int flags);
int close(int fd);
i64 read(int fd, void *buf, u64 count);
int fork(void);
int execvp(const char *file, char **argv);
int waitpid(int pid, int *status, int options);
int pipe(int *fds);
int dup2(int oldfd, int newfd);
int kill(int pid, int sig);
#endif

char *fern_getenv(const char *name) {
	char *v = getenv(name);
	if (v == 0) return "";
	return v;
}

i64 fern_chdir(const char *path) {
#ifdef _WIN32
	return (i64)_chdir(path);
#else
	return (i64)chdir(path);
#endif
}

i64 fern_setenv(const char *name, const char *value) {
#ifdef _WIN32
	return (i64)_putenv_s(name, value);
#else
	return (i64)setenv(name, value, 1);
#endif
}

void _exit(int code);

i64 fern_exit(i64 code) {
	_exit((int)code);
	return 0;
}

#ifdef _WIN32
int _access(const char *path, int mode);
int GetFileAttributesExA(const char *path, int level, void *data);
typedef struct {
	unsigned int attributes;
	unsigned int creation_low, creation_high;
	unsigned int access_low, access_high;
	unsigned int write_low, write_high;
	unsigned int size_low, size_high;
} fern_file_attr;
#else
int access(const char *path, int mode);
#endif

char fern_path_exists(const char *path) {
#ifdef _WIN32
	return _access(path, 0) == 0;
#else
	return access(path, 0) == 0;
#endif
}

/* Whether `prereq` is strictly newer than `target`. A missing target
   reports true so callers rebuild; a missing prerequisite reports false. */
char fern_path_newer(const char *target, const char *prereq) {
#ifdef _WIN32
	fern_file_attr t;
	fern_file_attr p;
	if (!GetFileAttributesExA(target, 0, &t)) return 1;
	if (!GetFileAttributesExA(prereq, 0, &p)) return 0;
	u64 tw = ((u64)t.write_high << 32) | (u64)t.write_low;
	u64 pw = ((u64)p.write_high << 32) | (u64)p.write_low;
	return pw > tw;
#else
	int pid = fork();
	if (pid == 0) {
		char *argv[5];
		argv[0] = (char *)"test";
		argv[1] = (char *)prereq;
		argv[2] = (char *)"-nt";
		argv[3] = (char *)target;
		argv[4] = 0;
		execvp(argv[0], argv);
		_exit(127);
	}
	if (pid < 0) return 0;
	int status = 0;
	if (waitpid(pid, &status, 0) != pid) return 0;
	return status == 0;
#endif
}

/* ------------------------------------------------------------------ */
/* processes                                                           */

/* A linked argv list built by the prelude from `List String`. */
typedef struct fargv {
	char *s;
	struct fargv *next;
} fargv;

fargv *fern_argv_nil(i64 unused) {
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

#ifdef _WIN32
/* _spawnvp joins argv with spaces without quoting, so a recipe reaches
   the shell as separate words. Quote each argument for CreateProcess's
   command-line parser before handing it to _spawnvp. */
static char *fern_win_quote(const char *arg) {
	u64 n = fern_strlen(arg);
	char *out = (char *)fern_alloc(2 * n + 3);
	u64 i = 0;
	int quote = n == 0;
	for (u64 j = 0; j < n; j++) {
		if (arg[j] == ' ' || arg[j] == '\t' || arg[j] == '"') quote = 1;
	}
	if (!quote) {
		for (u64 j = 0; j < n; j++) out[i++] = arg[j];
		out[i] = 0;
		return out;
	}
	out[i++] = '"';
	u64 j = 0;
	while (j < n) {
		u64 backslashes = 0;
		while (j < n && arg[j] == '\\') {
			backslashes++;
			j++;
		}
		if (j == n) {
			for (u64 k = 0; k < backslashes * 2; k++) out[i++] = '\\';
		} else if (arg[j] == '"') {
			for (u64 k = 0; k < backslashes * 2 + 1; k++) out[i++] = '\\';
			out[i++] = '"';
			j++;
		} else {
			for (u64 k = 0; k < backslashes; k++) out[i++] = '\\';
			out[i++] = arg[j++];
		}
	}
	out[i++] = '"';
	out[i] = 0;
	return out;
}

static char **fern_win_argv_array(char *name, fargv *args) {
	u64 n = fern_argv_count(args);
	char **out = (char **)fern_alloc((n + 2) * sizeof(char *));
	u64 i = 0;
	out[0] = name;
	while (args != 0) {
		i++;
		out[i] = fern_win_quote(args->s);
		args = args->next;
	}
	out[n + 1] = 0;
	return out;
}

/* Pipeline stages become one shell command line. Quote only arguments
   that need it: cmd.exe treats an all-quoted line as a single token. */
static int fern_arg_is_simple(const char *s) {
	if (*s == 0) return 0;
	while (*s != 0) {
		char c = *s++;
		if (c == ' ' || c == '\t' || c == '"' || c == '&' || c == '|' || c == '<' || c == '>' || c == '^' || c == '(' || c == ')' || c == '%' || c == '!') return 0;
	}
	return 1;
}

static void fern_append_char(char *out, u64 *len, u64 cap, char c) {
	if (*len + 1 < cap) out[(*len)++] = c;
}

static void fern_append_arg(char *out, u64 *len, u64 cap, const char *arg) {
	if (fern_arg_is_simple(arg)) {
		while (*arg != 0) fern_append_char(out, len, cap, *arg++);
		return;
	}
	fern_append_char(out, len, cap, '"');
	while (*arg != 0) {
		if (*arg == '"') fern_append_char(out, len, cap, '\\');
		fern_append_char(out, len, cap, *arg++);
	}
	fern_append_char(out, len, cap, '"');
}

static void fern_append_argv(char *out, u64 *len, u64 cap, char *name, fargv *args) {
	fern_append_arg(out, len, cap, name);
	while (args != 0) {
		fern_append_char(out, len, cap, ' ');
		fern_append_arg(out, len, cap, args->s);
		args = args->next;
	}
}

i64 fern_spawn_status(char *name, fargv *args) {
	return (i64)_spawnvp(0, name, fern_win_argv_array(name, args));
}

/* A regular file's contents, truncated at 64 KiB. A missing file reads
   as the empty string. */
char *fern_read_file(const char *path) {
	int fd = _open(path, 0);
	if (fd < 0) return "";
	u64 cap = 65536;
	char *out = (char *)fern_alloc(cap);
	u64 len = 0;
	for (;;) {
		int got = _read(fd, out + len, (unsigned int)(cap - 1 - len));
		if (got <= 0) break;
		len += (u64)got;
		if (len >= cap - 1) break;
	}
	_close(fd);
	out[len] = 0;
	return out;
}

/* Captured stdout, truncated at 64 KiB. MSVCRT has no captured spawn
   that passes an argv array, so point fd 1 at a temporary file while
   the child runs and read it back afterwards. */
char *fern_spawn_capture(char *name, fargv *args) {
	u64 cap = 65536;
	char *out = (char *)fern_alloc(cap);
	out[0] = 0;
	char *tmp = _tempnam(0, "fern");
	if (tmp == 0) return out;
	int fd = _open(tmp, 0x0002 | 0x0100 | 0x0200 | 0x8000, 0x0100 | 0x0080);
	if (fd < 0) {
		free(tmp);
		return out;
	}
	int saved = _dup(1);
	if (saved < 0 || _dup2(fd, 1) < 0) {
		if (saved >= 0) _close(saved);
		_close(fd);
		remove(tmp);
		free(tmp);
		return out;
	}
	_spawnvp(0, name, fern_win_argv_array(name, args));
	_dup2(saved, 1);
	_close(saved);
	_lseeki64(fd, 0, 0);
	u64 len = 0;
	for (;;) {
		int got = _read(fd, out + len, (unsigned int)(cap - 1 - len));
		if (got <= 0) break;
		len += (u64)got;
		if (len >= cap - 1) break;
	}
	_close(fd);
	remove(tmp);
	free(tmp);
	out[len] = 0;
	return out;
}
#else
/* Wait status as a shell would report it: exit code, or 128 + signal. */
static i64 fern_status_of(int status) {
	if ((status & 0x7f) == 0) return (status >> 8) & 0xff;
	return 128 + (status & 0x7f);
}

i64 fern_spawn_status(char *name, fargv *args) {
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
		i64 got = read(fd, out + len, cap - 1 - len);
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
		i64 got = read(fds[0], out + len, cap - 1 - len);
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
#endif

/* ------------------------------------------------------------------ */
/* pipelines                                                           */

typedef struct fpipe {
	fargv *argv;
	struct fpipe *next;
} fpipe;

fpipe *fern_pipe_nil(i64 unused) {
	(void)unused;
	return 0;
}

fpipe *fern_pipe_cons(fargv *argv, fpipe *next) {
	fpipe *node = (fpipe *)fern_alloc(sizeof(fpipe));
	node->argv = argv;
	node->next = next;
	return node;
}

#ifdef _WIN32
static char *fern_pipeline_line(fpipe *stages) {
	u64 cap = 65536;
	char *out = (char *)fern_alloc(cap);
	u64 len = 0;
	while (stages != 0) {
		if (stages->argv == 0) break;
		if (len > 0) {
			fern_append_char(out, &len, cap, ' ');
			fern_append_char(out, &len, cap, '|');
			fern_append_char(out, &len, cap, ' ');
		}
		fern_append_argv(out, &len, cap, stages->argv->s, stages->argv->next);
		stages = stages->next;
	}
	out[len] = 0;
	return out;
}

static i64 fern_pipeline_run(fpipe *stages, char *out, u64 cap) {
	char *line = fern_pipeline_line(stages);
	if (out == 0) return (i64)system(line);
	void *pipe = _popen(line, "r");
	if (pipe == 0) {
		if (cap > 0) out[0] = 0;
		return -1;
	}
	u64 len = 0;
	for (;;) {
		u64 got = fread(out + len, 1, cap - 1 - len, pipe);
		if (got == 0) break;
		len += got;
		if (len >= cap - 1) break;
	}
	int status = _pclose(pipe);
	out[len] = 0;
	return (i64)status;
}
#else
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

static i64 fern_pipeline_wait(int *pids, u64 n) {
	i64 last = -1;
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
static i64 fern_pipeline_run(fpipe *stages, char *out, u64 cap) {
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
			i64 got = read(out_fds[0], out + len, cap - 1 - len);
			if (got <= 0) break;
			len += (u64)got;
			if (len >= cap - 1) break;
		}
		close(out_fds[0]);
		out_fds[0] = -1;
		out[len] = 0;
	}

	i64 status = fern_pipeline_wait(pids, n);
	fern_argv_vecs_free(argvs, n);
	free(pids);
	free(fds);
	return status;
}
#endif

i64 fern_pipeline_status(fpipe *stages) {
	return fern_pipeline_run(stages, 0, 0);
}

char *fern_pipeline_capture(fpipe *stages) {
	u64 cap = 65536;
	char *out = (char *)fern_alloc(cap);
	out[0] = 0;
	fern_pipeline_run(stages, out, cap);
	return out;
}

/* ------------------------------------------------------------------ */
/* async IO leaves                                                     */

/* File and stdin reads run on a detached OS thread and fill a malloc'd
   buffer; only the polling thread allocates GC strings. The leaf frame
   is registered with the collector, so its raw path and data pointers
   stay reachable while the worker runs. */

typedef struct {
	i64 sec;
	i64 usec;
} fern_timeval;

#ifdef _WIN32
u64 GetTickCount64(void);
void Sleep(unsigned int ms);
u64 _beginthreadex(void *security, unsigned int stack, unsigned (*start)(void *), void *arg, unsigned int flags, unsigned int *tid);
int CloseHandle(u64 handle);
#else
int gettimeofday(void *tv, void *tz);
int usleep(unsigned int usec);
int pthread_create(u64 *thread, void *attr, void *(*start)(void *), void *arg);
int pthread_detach(u64 thread);
#endif

static i64 fern_now_ms(void) {
#ifdef _WIN32
	return (i64)GetTickCount64();
#else
	fern_timeval tv;
	gettimeofday(&tv, 0);
	return tv.sec * 1000 + tv.usec / 1000;
#endif
}

static void fern_idle(void) {
#ifdef _WIN32
	Sleep(1);
#else
	usleep(1000);
#endif
}

typedef struct {
	i64 deadline;
} fern_sleep;

static int fern_sleep_poll(void *frame, void *ctx) {
	fern_sleep *s = (fern_sleep *)frame;
	(void)ctx;
	return fern_now_ms() >= s->deadline;
}

fern_future fern_io_sleep(i64 ms) {
	fern_future f;
	fern_sleep *s = (fern_sleep *)__glcc_frame_alloc(sizeof(fern_sleep), 8);
	if (ms < 0) ms = 0;
	s->deadline = fern_now_ms() + ms;
	f.frame = s;
	f.poll = fern_sleep_poll;
	return f;
}

typedef struct {
	char *result; /* Future String result lives at frame offset 0 */
	i64 done;
	i64 copied;
	i64 mode; /* 0 read file, 1 write file, 2 stdin bytes, 3 stdin line */
	char *path;
	char *data;
	char *raw;
	u64 len;
	i64 max;
} fern_io;

static i64 fern_io_fd_read(i64 fd, char *buf, u64 n) {
#ifdef _WIN32
	return (i64)_read((int)fd, buf, (unsigned int)n);
#else
	return read((int)fd, buf, n);
#endif
}

static void fern_io_read_file_run(fern_io *io) {
	void *stream = fopen(io->path, "rb");
	u64 cap = 65536;
	char *buf;
	u64 len = 0;
	if (stream == 0) return;
	buf = (char *)malloc(cap + 1);
	if (buf == 0) {
		fclose(stream);
		return;
	}
	for (;;) {
		u64 got;
		if (len == cap) {
			char *grown = (char *)realloc(buf, cap * 2 + 1);
			if (grown == 0) break;
			buf = grown;
			cap = cap * 2;
		}
		got = fread(buf + len, 1, cap - len, stream);
		if (got == 0) break;
		len += got;
	}
	fclose(stream);
	buf[len] = 0;
	io->raw = buf;
	io->len = len;
}

static void fern_io_write_file_run(fern_io *io) {
	void *stream = fopen(io->path, "wb");
	u64 len = fern_strlen(io->data);
	u64 at = 0;
	if (stream == 0) return;
	while (at < len) {
		u64 put = fwrite(io->data + at, 1, len - at, stream);
		if (put == 0) break;
		at += put;
	}
	fclose(stream);
}

static void fern_io_stdin_run(fern_io *io) {
	u64 cap = io->max > 0 ? (u64)io->max : 65536;
	char *buf = (char *)malloc(cap + 1);
	u64 len = 0;
	if (buf == 0) return;
	while (len < cap) {
		i64 got = fern_io_fd_read(0, buf + len, cap - len);
		if (got <= 0) break;
		len += (u64)got;
	}
	buf[len] = 0;
	io->raw = buf;
	io->len = len;
}

static void fern_io_line_run(fern_io *io) {
	u64 cap = 65536;
	char *buf = (char *)malloc(cap + 1);
	u64 len = 0;
	if (buf == 0) return;
	while (len < cap) {
		char c;
		i64 got = fern_io_fd_read(0, &c, 1);
		if (got <= 0) break;
		if (c == '\n') break;
		buf[len] = c;
		len++;
	}
	if (len > 0 && buf[len - 1] == '\r') len--;
	buf[len] = 0;
	io->raw = buf;
	io->len = len;
}

static void fern_io_run(fern_io *io) {
	if (io->mode == 0) fern_io_read_file_run(io);
	else if (io->mode == 1) fern_io_write_file_run(io);
	else if (io->mode == 2) fern_io_stdin_run(io);
	else fern_io_line_run(io);
	__atomic_store_n(&io->done, 1, 5);
}

#ifdef _WIN32
static unsigned fern_io_thread(void *arg) {
	fern_io_run((fern_io *)arg);
	return 0;
}
#else
static void *fern_io_thread(void *arg) {
	fern_io_run((fern_io *)arg);
	return 0;
}
#endif

static void fern_io_start(fern_io *io) {
#ifdef _WIN32
	u64 handle = _beginthreadex(0, 0, fern_io_thread, io, 0, 0);
	if (handle != 0) CloseHandle(handle);
	else fern_io_run(io);
#else
	u64 thread;
	if (pthread_create(&thread, 0, fern_io_thread, io) == 0) pthread_detach(thread);
	else fern_io_run(io);
#endif
}

static int fern_io_poll(void *frame, void *ctx) {
	fern_io *io = (fern_io *)frame;
	(void)ctx;
	if (__atomic_load_n(&io->done, 5) == 0) return 0;
	if (io->mode != 1 && io->copied == 0) {
		char *out = (char *)fern_alloc(io->len + 1);
		u64 i;
		for (i = 0; i < io->len; i++) out[i] = io->raw[i];
		out[io->len] = 0;
		if (io->raw != 0) free(io->raw);
		io->raw = 0;
		io->result = out;
		io->copied = 1;
	}
	return 1;
}

static fern_future fern_io_future(fern_io *io) {
	fern_future f;
	f.frame = io;
	f.poll = fern_io_poll;
	return f;
}

static fern_io *fern_io_new(i64 mode) {
	fern_io *io = (fern_io *)__glcc_frame_alloc(sizeof(fern_io), 8);
	io->result = 0;
	io->done = 0;
	io->copied = 0;
	io->mode = mode;
	io->path = 0;
	io->data = 0;
	io->raw = 0;
	io->len = 0;
	io->max = 0;
	return io;
}

fern_future fern_io_read_file(char *path) {
	fern_io *io = fern_io_new(0);
	io->path = path;
	fern_io_start(io);
	return fern_io_future(io);
}

fern_future fern_io_write_file(char *path, char *data) {
	fern_io *io = fern_io_new(1);
	io->path = path;
	io->data = data;
	fern_io_start(io);
	return fern_io_future(io);
}

fern_future fern_io_read_stdin(i64 max) {
	fern_io *io = fern_io_new(2);
	io->max = max;
	fern_io_start(io);
	return fern_io_future(io);
}

fern_future fern_io_read_line(i64 unused) {
	fern_io *io = fern_io_new(3);
	(void)unused;
	fern_io_start(io);
	return fern_io_future(io);
}
