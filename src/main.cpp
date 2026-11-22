// Shopping Sessions pipeline: a native, multithreaded implementation of rules.md.
//
// Reads  $DATA_DIR    (default /data)
// Writes $OUTPUT_DIR  (default /output): user_daily/ and campaign_daily/ as Parquet
//
// Two implementations share the parsing rules:
//   fast::run     compact records, AVX2 line scanning, partitioned dedup and
//                 grouping. Needs a window under ~40 days and at most 256
//                 categories, which every generated dataset satisfies.
//   generic::run  the straightforward version (64-bit everything) used when
//                 those limits do not hold.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <dirent.h>
#include <sched.h>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>

using namespace std;
typedef int64_t i64;
typedef uint64_t u64;
typedef uint32_t u32;
typedef int32_t i32;
typedef uint16_t u16;
typedef int16_t i16;
typedef uint8_t u8;

static const i64 DAY_MS = 86400000LL;
static const i64 GAP_MS = 1800000LL;
static const i64 WEEK_MS = 7 * DAY_MS;

static int NT = 8;
static bool DBG = false;
static chrono::steady_clock::time_point T0;
static void tick(const char* what) {
    if (DBG) {
        rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        fprintf(stderr, "%9.3f ms  %s  [minflt %ld]\n",
                chrono::duration<double, milli>(chrono::steady_clock::now() - T0).count(), what, (long)ru.ru_minflt);
    }
}
static void die(const char* msg) {
    fprintf(stderr, "fatal: %s\n", msg);
    exit(1);
}

// One multiply: the high half is well mixed (partitions take the top bits),
// and folding it into the low half mixes the low bits (tables mask them).
static inline u64 mix64(u64 x) {
    x *= 0x9E3779B97F4A7C15ULL;
    return x ^ (x >> 32);
}

// Thread count for par_for; a thread may override it (the links thread runs
// its own par_for calls serially while the parse workers run).
static thread_local int PAR_NT = 0;
template <class F>
static void par_for(int n, F&& f) {
    const int NT = PAR_NT ? PAR_NT : ::NT;
    atomic<int> next(0);
    auto work = [&](int tid) {
        for (;;) {
            int i = next.fetch_add(1, memory_order_relaxed);
            if (i >= n) break;
            f(i, tid);
        }
    };
    int nt = min(NT, max(n, 1));
    vector<thread> th;
    for (int t = 1; t < nt; t++) th.emplace_back(work, t);
    work(0);
    for (auto& x : th) x.join();
}

// ---------------------------------------------------------------------------
// Keys: an exact, order-preserving 64-bit code for short IDs.
// Strings of 1..9 bytes, each byte in 1..127, pack 7 bits per byte, first byte
// most significant and zero padded, so numeric order equals byte-wise order.
// Anything else is interned in a pool and flagged with the top bit. Key 0 is
// never produced (null).
// ---------------------------------------------------------------------------
static void init_codes() {}
static const u64 POOLBIT = 1ULL << 63;
static mutex pool_mu;
static unordered_map<string, u64> pool_map;
static vector<string> pool_strs;

static u64 pool_intern(const char* p, size_t n) {
    lock_guard<mutex> g(pool_mu);
    string s(p, n);
    auto it = pool_map.find(s);
    if (it != pool_map.end()) return it->second;
    u64 k = POOLBIT | (u64)pool_strs.size();
    pool_strs.push_back(s);
    pool_map.emplace(std::move(s), k);
    return k;
}
static u64 pool_find(const char* p, size_t n) {
    lock_guard<mutex> g(pool_mu);
    auto it = pool_map.find(string(p, n));
    return it == pool_map.end() ? 0 : it->second;
}
static inline bool short_code(const char* p, size_t n, u64& k) {
    if (n > 9 || n == 0) return false;
    u64 r = 0;
    for (size_t i = 0; i < n; i++) {
        u64 c = (u8)p[i];
        if (c == 0 || c >= 128) return false;
        r |= c << (56 - 7 * i);
    }
    k = r;
    return true;
}
static inline u64 key_of(const char* p, size_t n) {
    u64 k;
    if (__builtin_expect(short_code(p, n, k), 1)) return k;
    return pool_intern(p, n);
}
// Branch-light short_code using pext; needs 9 readable bytes at p.
static inline bool short_code_simd(const char* p, u32 n, u64& k) {
    if (n > 9 || n == 0) return false;
    u64 w;
    memcpy(&w, p, 8);
    const u32 n8 = n < 8 ? n : 8;
    const u64 m = n8 == 8 ? ~0ULL : ((1ULL << (8 * n8)) - 1);
    const u64 x = w & m;
    const u64 y = x | (0x0101010101010101ULL & ~m);  // bytes past the end count as non-zero
    u64 bad = (x & 0x8080808080808080ULL) | ((y - 0x0101010101010101ULL) & ~y & 0x8080808080808080ULL);
    u64 c8 = 0;
    if (n == 9) {
        c8 = (u8)p[8];
        bad |= (c8 == 0) | (c8 >= 128);
    }
    if (bad) return false;
    k = (_pext_u64(__builtin_bswap64(x), 0x7F7F7F7F7F7F7F7FULL) << 7) | c8;
    return true;
}
static inline u64 key_of_fast(const char* p, u32 n, const char* bufend) {
    u64 k;
    if (__builtin_expect(p + 16 <= bufend, 1)) {
        if (__builtin_expect(short_code_simd(p, n, k), 1)) return k;
        return pool_intern(p, n);
    }
    return key_of(p, n);
}
static inline u64 key_find(const char* p, size_t n) {
    u64 k;
    if (short_code(p, n, k)) return k;
    return pool_find(p, n);
}
static string key_str(u64 k) {
    if (k & POOLBIT) {
        lock_guard<mutex> g(pool_mu);
        return pool_strs[k & ~POOLBIT];
    }
    string s;
    for (int i = 0; i < 9; i++) {
        int c = (int)((k >> (56 - 7 * i)) & 127);
        if (!c) break;
        s.push_back((char)c);
    }
    return s;
}
static inline bool key_less(u64 a, u64 b) {
    if (!((a | b) & POOLBIT)) return a < b;
    return key_str(a) < key_str(b);
}
// Appends the 4-byte length and the bytes of key k (Parquet PLAIN BYTE_ARRAY).
static inline void put_key(string& page, u64 k) {
    if (k & POOLBIT) {
        string s = key_str(k);
        u32 n = (u32)s.size();
        page.append((const char*)&n, 4);
        page += s;
        return;
    }
    char buf[13];
    u32 n = 0;
    for (int i = 0; i < 9; i++) {
        int c = (int)((k >> (56 - 7 * i)) & 127);
        if (!c) break;
        buf[4 + n++] = (char)c;
    }
    memcpy(buf, &n, 4);
    page.append(buf, 4 + n);
}

// ---------------------------------------------------------------------------
// Open-addressing hash map u64 -> u64 (key 0 = empty).
// ---------------------------------------------------------------------------
struct KV {
    u64 k, v;
};
struct FMap {
    vector<KV> t;
    u64 mask = 0;
    void init(size_t n) {
        size_t c = 16;
        while (c < n * 2) c <<= 1;
        t.assign(c, KV{0, 0});
        mask = c - 1;
    }
    KV* slot(u64 k) {
        u64 i = mix64(k) & mask;
        while (t[i].k && t[i].k != k) i = (i + 1) & mask;
        return &t[i];
    }
    const KV* get(u64 k) const {
        u64 i = mix64(k) & mask;
        for (;;) {
            const KV& e = t[i];
            if (e.k == k) return &e;
            if (!e.k) return nullptr;
            i = (i + 1) & mask;
        }
    }
};

// ---------------------------------------------------------------------------
// Timestamps, prices, CSV dialect
// ---------------------------------------------------------------------------
static inline bool D(char c) { return (u8)(c - '0') < 10; }
static inline int d2(const char* p) { return (p[0] - '0') * 10 + (p[1] - '0'); }
static inline i64 days_from_civil(i64 y, i64 m, i64 d) {
    y -= m <= 2;
    i64 era = (y >= 0 ? y : y - 399) / 400;
    i64 yoe = y - era * 400;
    i64 doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    i64 doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + doe - 719468;
}
static inline bool is_leap(i64 y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }
static inline i64 floordiv(i64 a, i64 b) { return a / b - ((a % b != 0) && ((a < 0) != (b < 0))); }

// true = a valid timestamp (F1, F2 or F3); the caller checks the window.
// An F3 value too large for 64 bits is outside every window, so it is false too.
static inline bool parse_ts(const char* p, size_t n, i64& out) {
    if ((n == 20 || n == 25) && p[4] == '-') {
        u32 bad = (u8)(p[0] - '0') > 9;
        bad |= (u8)(p[1] - '0') > 9;
        bad |= (u8)(p[2] - '0') > 9;
        bad |= (u8)(p[3] - '0') > 9;
        bad |= (u8)(p[5] - '0') > 9;
        bad |= (u8)(p[6] - '0') > 9;
        bad |= p[7] != '-';
        bad |= (u8)(p[8] - '0') > 9;
        bad |= (u8)(p[9] - '0') > 9;
        bad |= p[10] != 'T';
        bad |= (u8)(p[11] - '0') > 9;
        bad |= (u8)(p[12] - '0') > 9;
        bad |= p[13] != ':';
        bad |= (u8)(p[14] - '0') > 9;
        bad |= (u8)(p[15] - '0') > 9;
        bad |= p[16] != ':';
        bad |= (u8)(p[17] - '0') > 9;
        bad |= (u8)(p[18] - '0') > 9;
        if (bad) return false;
        int y = d2(p) * 100 + d2(p + 2), mo = d2(p + 5), d = d2(p + 8), h = d2(p + 11),
            mi = d2(p + 14), s = d2(p + 17);
        if (mo < 1 || mo > 12 || d < 1 || h > 23 || mi > 59 || s > 59) return false;
        static const int DIM[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        if (d > DIM[mo - 1] + (mo == 2 && is_leap(y))) return false;
        i64 off = 0;
        if (n == 20) {
            if (p[19] != 'Z') return false;
        } else {
            char sg = p[19];
            if (sg != '+' && sg != '-') return false;
            if (!(D(p[20]) && D(p[21]) && p[22] == ':' && D(p[23]) && D(p[24]))) return false;
            int oh = d2(p + 20), om = d2(p + 23);
            if (oh > 23 || om > 59) return false;
            off = (oh * 60 + om) * 60000LL;
            if (sg == '-') off = -off;
        }
        out = (days_from_civil(y, mo, d) * 86400LL + h * 3600 + mi * 60 + s) * 1000LL - off;
        return true;
    }
    if (n == 0) return false;
    size_t i = 0;
    while (i < n && p[i] == '0') i++;
    if (n - i > 19) return false;
    u64 v = 0;
    u32 bad = 0;
    for (; i < n; i++) {
        u32 dg = (u8)(p[i] - '0');
        bad |= dg > 9;
        v = v * 10 + dg;
    }
    if (bad || v > (u64)INT64_MAX) return false;
    out = (i64)v;
    return true;
}

// ---- fast timestamp paths ----------------------------------------------------------
// A table of every valid "YYYY-MM-DD" from window_start - 70 days to window_end + 70
// days (keyed by the exact 10 bytes) gives the day number without re-validating.
struct DateEnt {
    u64 k8;  // bytes 0..7  "YYYY-MM-"
    u32 k2;  // bytes 8..9  "DD" (+1 in bit 16 to mark a used slot)
    i32 days;
};
// Direct-mapped by (year % 4, month digits, day digits); a slot holds at most one
// date of the +-70-day range, and the full 10-byte key is always compared.
static const int DT_N = 4 * 26 * 46;
static DateEnt DATES[DT_N];
static inline u32 date_index(const char* p) {
    u32 y = (u8)p[3] & 3;
    u32 mo = ((u8)p[5] & 1) * 10 + ((u8)p[6] & 15);
    u32 dd = ((u8)p[8] & 3) * 10 + ((u8)p[9] & 15);
    return (y * 26 + mo) * 46 + dd;
}
static inline u64 ld8(const char* p) {
    u64 v;
    memcpy(&v, p, 8);
    return v;
}
static inline u32 ld2(const char* p) {
    u16 v;
    memcpy(&v, p, 2);
    return v;
}
static void init_dates(i64 ws, i64 we) {
    memset(DATES, 0, sizeof(DATES));
    i64 d0 = floordiv(ws, DAY_MS) - 70, d1 = floordiv(we, DAY_MS) + 70;
    for (i64 d = d0; d <= d1; d++) {
        // civil from days (Howard Hinnant)
        i64 z = d + 719468;
        i64 era = (z >= 0 ? z : z - 146096) / 146097;
        i64 doe = z - era * 146097;
        i64 yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        i64 y = yoe + era * 400;
        i64 doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        i64 mp = (5 * doy + 2) / 153;
        i64 dd = doy - (153 * mp + 2) / 5 + 1;
        i64 m = mp < 10 ? mp + 3 : mp - 9;
        y += m <= 2;
        if (y < 0 || y > 9999) continue;
        char s[11];
        snprintf(s, sizeof s, "%04d-%02d-%02d", (int)y, (int)m, (int)dd);
        u64 k8 = ld8(s);
        u32 k2 = ld2(s + 8) | 0x10000u;
        u32 i = date_index(s);
        if (DATES[i].k2) continue;  // cannot happen within +-70 days; the slow path covers it
        DATES[i] = DateEnt{k8, k2, (i32)d};
    }
}
// Days for the 10 bytes at p if they are a date in the table; false otherwise.
static inline bool date_lookup(const char* p, i64& days) {
    const DateEnt& e = DATES[date_index(p)];
    if (e.k8 == ld8(p) && e.k2 == (ld2(p + 8) | 0x10000u)) {
        days = e.days;
        return true;
    }
    return false;
}
static inline bool all_digits8(u64 w) {
    return ((w & 0xF0F0F0F0F0F0F0F0ULL) == 0x3030303030303030ULL) &&
           (((w + 0x0606060606060606ULL) & 0xF0F0F0F0F0F0F0F0ULL) == 0x3030303030303030ULL);
}
// 8 ASCII digits (already validated), first digit most significant.
static inline u64 parse8(u64 w) {
    w -= 0x3030303030303030ULL;
    w = (w * 10) + (w >> 8);
    w = (((w & 0x000000FF000000FFULL) * (100 + (1000000ULL << 32))) +
         (((w >> 16) & 0x000000FF000000FFULL) * (1 + (10000ULL << 32)))) >>
        32;
    return w;
}
// "HH:MM:SS" at p -> seconds of day; false if invalid.
__attribute__((always_inline)) static inline bool hms(const char* p, i64& sec) {
    u64 t = ld8(p);
    // digits at bytes 0,1,3,4,6,7 and ':' at 2,5
    const u64 DM = 0xFFFF00FFFF00FFFFULL;
    u64 x = (t & DM) | (0x3030303030303030ULL & ~DM);
    if (!all_digits8(x) || (t & ~DM) != 0x00003A00003A0000ULL) return false;
    // bytes are now 0..9 (digits) or 10 (colons); byte i becomes 10*b[i] + b[i+1], no carries
    u64 d = t - 0x3030303030303030ULL;
    u64 v = d * 10 + (d >> 8);
    u32 h = (u32)(v & 0xFF), m = (u32)((v >> 24) & 0xFF), s = (u32)((v >> 48) & 0xFF);
    if (h > 23 || m > 59 || s > 59) return false;
    sec = h * 3600 + m * 60 + s;
    return true;
}
// Same contract as parse_ts, with fast paths for the common shapes.
__attribute__((always_inline)) static inline bool parse_ts_fast(const char* p, size_t n, i64& out) {
    if (n == 13) {
        u64 a = ld8(p), b = ld8(p + 5);
        if (all_digits8(a) && all_digits8(b)) {
            out = (i64)(parse8(a) * 100000 + parse8(b) % 100000);
            return true;
        }
        return parse_ts(p, n, out);
    }
    if ((n == 20 || n == 25) && p[10] == 'T') {
        i64 days, sec;
        if (date_lookup(p, days) && hms(p + 11, sec)) {
            if (n == 20) {
                if (p[19] != 'Z') return false;
                out = (days * 86400 + sec) * 1000;
                return true;
            }
            char sg = p[19];
            if ((sg == '+' || sg == '-') && D(p[20]) && D(p[21]) && p[22] == ':' && D(p[23]) && D(p[24])) {
                int oh = d2(p + 20), om = d2(p + 23);
                if (oh > 23 || om > 59) return false;
                i64 off = (oh * 60 + om) * 60000LL;
                out = (days * 86400 + sec) * 1000 - (sg == '-' ? -off : off);
                return true;
            }
            return false;
        }
    }
    return parse_ts(p, n, out);
}

static inline bool parse_price(const char* p, size_t n, u32& out) {
    if (n == 0) return false;
    size_t i = 0;
    while (i < n && p[i] == '0') i++;
    if (n - i > 8) return false;
    u32 v = 0, bad = 0;
    for (; i < n; i++) {
        u32 dg = (u8)(p[i] - '0');
        bad |= dg > 9;
        v = v * 10 + dg;
    }
    if (bad || v < 1 || v > 10000000) return false;
    out = v;
    return true;
}
// parse_price must also reject non-digits among leading zeros: they are skipped
// only while they are '0', so any other byte reaches the digit check.

// Splits [p, e) into exactly NF fields. Every comma separates fields; a quote
// may only wrap a whole field.
template <int NF>
static inline bool split(const char* p, const char* e, const char** fs, u32* fl) {
    int f = 0;
    const char* s = p;
    for (const char* q = p;; q++) {
        if (q == e || *q == ',') {
            if (f >= NF) return false;
            const char* a = s;
            const char* b = q;
            size_t len = (size_t)(b - a);
            if (len && (a[0] == '"' || b[-1] == '"')) {
                if (len < 2 || a[0] != '"' || b[-1] != '"') return false;
                a++, b--;
            }
            if (b > a && memchr(a, '"', (size_t)(b - a))) return false;
            fs[f] = a;
            fl[f] = (u32)(b - a);
            f++;
            if (q == e) break;
            s = q + 1;
        }
    }
    return f == NF;
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------
struct Mapped {
    const char* p = nullptr;
    size_t n = 0;
    int fd = -1;  // kept open for the big inputs, which are read with pread
};
static Mapped map_file(const string& path) {
    Mapped m;
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return m;
    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_size > 0) {
        void* a = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (a != MAP_FAILED) {
            m.p = (const char*)a;
            m.n = (size_t)st.st_size;
            m.fd = fd;
            return m;
        }
    }
    close(fd);
    return m;
}
static vector<string> list_csv(const string& dir) {
    vector<string> out;
    DIR* d = opendir(dir.c_str());
    if (!d) return out;
    while (dirent* e = readdir(d)) {
        string name = e->d_name;
        if (name.size() < 4 || name[0] == '.' || name.compare(name.size() - 4, 4, ".csv") != 0) continue;
        string path = dir + "/" + name;
        struct stat st;
        if (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) out.push_back(path);
    }
    closedir(d);
    sort(out.begin(), out.end());
    return out;
}

struct Task {
    const char* b;
    size_t size, begin, end;
    int kind;
    int fd = -1;
};
static void add_tasks(vector<Task>& tasks, const vector<Mapped>& files, int kind, size_t chunk) {
    for (auto& m : files) {
        if (!m.n) continue;
        for (size_t s = 0; s < m.n; s += chunk) tasks.push_back(Task{m.p, m.n, s, min(m.n, s + chunk), kind, m.fd});
    }
}
// Position of the first data line starting in [t.begin, ...), skipping the header.
static inline size_t first_line(const Task& t) {
    const char* b = t.b;
    size_t pos = t.begin;
    if (pos == 0 || b[pos - 1] != '\n') {
        const char* nl = (const char*)memchr(b + pos, '\n', t.size - pos);
        if (!nl) return t.size;
        pos = (size_t)(nl - b) + 1;
    }
    return pos;
}
// Calls f(lineStart, lineEnd) for every data line that starts in [begin, end).
template <class F>
static void for_lines(const Task& t, F&& f) {
    const char* b = t.b;
    size_t pos = first_line(t);
    while (pos < t.end) {
        const char* nl = (const char*)memchr(b + pos, '\n', t.size - pos);
        size_t le = nl ? (size_t)(nl - b) : t.size;
        f(b + pos, b + le);
        pos = le + 1;
    }
}
static string read_small(const string& path) {
    Mapped m = map_file(path);
    string s(m.p ? m.p : "", m.n);
    if (m.p) munmap((void*)m.p, m.n);
    return s;
}
template <class F>
static void for_small_lines(const string& text, F&& f) {
    size_t pos = text.find('\n');
    if (pos == string::npos) return;
    pos++;
    while (pos < text.size()) {
        size_t nl = text.find('\n', pos);
        size_t le = nl == string::npos ? text.size() : nl;
        f(text.data() + pos, text.data() + le);
        pos = le + 1;
    }
}

// ---------------------------------------------------------------------------
// Parquet writer (uncompressed, PLAIN, one row group)
// ---------------------------------------------------------------------------
struct TW {
    string b;
    vector<int> last{0};
    void uv(u64 v) {
        while (v >= 0x80) {
            b.push_back((char)(v | 0x80));
            v >>= 7;
        }
        b.push_back((char)v);
    }
    void zz(i64 v) { uv(((u64)v << 1) ^ (u64)(v >> 63)); }
    void fh(int id, int ty) {
        int d = id - last.back();
        if (d > 0 && d <= 15)
            b.push_back((char)((d << 4) | ty));
        else {
            b.push_back((char)ty);
            zz(id);
        }
        last.back() = id;
    }
    void fi32(int id, i64 v) { fh(id, 5), zz(v); }
    void fi64(int id, i64 v) { fh(id, 6), zz(v); }
    void fstr(int id, const string& s) { fh(id, 8), uv(s.size()), b += s; }
    void fstruct(int id) { fh(id, 12), last.push_back(0); }
    void sbegin() { last.push_back(0); }
    void send() { b.push_back(0), last.pop_back(); }
    void flist(int id, int ety, int n) {
        fh(id, 9);
        if (n < 15)
            b.push_back((char)((n << 4) | ety));
        else {
            b.push_back((char)(0xF0 | ety));
            uv((u64)n);
        }
    }
};
enum { PT_INT32 = 1, PT_INT64 = 2, PT_BYTES = 6 };
struct ColData {
    string name;
    int ptype;
    bool optional;
    int conv;     // -1 none, 0 UTF8, 6 DATE
    int logical;  // 0 none, 1 STRING, 6 DATE
    string page;
};
static void put32(string& s, u32 v) { s.append((const char*)&v, 4); }

static void write_parquet(const string& path, i64 nrows, const vector<ColData>& cols) {
    // The file is "PAR1", per column a page header and its page, then the
    // footer; written with writev straight from the page buffers.
    vector<string> hdr;
    i64 pos = 4;
    vector<i64> offs, sizes;
    i64 total = 0;
    if (nrows > 0) {
        for (auto& c : cols) {
            TW h;
            h.fi32(1, 0);
            h.fi32(2, (i64)c.page.size());
            h.fi32(3, (i64)c.page.size());
            h.fstruct(5);
            h.fi32(1, nrows);
            h.fi32(2, 0);
            h.fi32(3, 3);
            h.fi32(4, 3);
            h.send();
            h.b.push_back(0);
            offs.push_back(pos);
            sizes.push_back((i64)(h.b.size() + c.page.size()));
            pos += sizes.back();
            total += sizes.back();
            hdr.push_back(h.b);
        }
    }
    TW f;
    f.fi32(1, 1);
    f.flist(2, 12, (int)cols.size() + 1);
    f.sbegin();
    f.fstr(4, "schema");
    f.fi32(5, (i64)cols.size());
    f.send();
    for (auto& c : cols) {
        f.sbegin();
        f.fi32(1, c.ptype);
        f.fi32(3, c.optional ? 1 : 0);
        f.fstr(4, c.name);
        if (c.conv >= 0) f.fi32(6, c.conv);
        if (c.logical) {
            f.fstruct(10);
            f.fstruct(c.logical);
            f.send();
            f.send();
        }
        f.send();
    }
    f.fi64(3, nrows);
    if (nrows > 0) {
        f.flist(4, 12, 1);
        f.sbegin();
        f.flist(1, 12, (int)cols.size());
        for (size_t i = 0; i < cols.size(); i++) {
            auto& c = cols[i];
            f.sbegin();
            f.fi64(2, offs[i]);
            f.fstruct(3);
            f.fi32(1, c.ptype);
            if (c.optional) {
                f.flist(2, 5, 2);
                f.zz(0);
                f.zz(3);
            } else {
                f.flist(2, 5, 1);
                f.zz(0);
            }
            f.flist(3, 8, 1);
            f.uv(c.name.size());
            f.b += c.name;
            f.fi32(4, 0);
            f.fi64(5, nrows);
            f.fi64(6, sizes[i]);
            f.fi64(7, sizes[i]);
            f.fi64(9, offs[i]);
            f.send();
            f.send();
        }
        f.fi64(2, total);
        f.fi64(3, nrows);
        f.send();
    } else {
        f.flist(4, 12, 0);
    }
    f.fstr(6, "shopping-sessions");
    f.b.push_back(0);
    string tail = f.b;
    put32(tail, (u32)f.b.size());
    tail += "PAR1";
    vector<iovec> iov;
    static const char MAGIC[4] = {'P', 'A', 'R', '1'};
    iov.push_back({(void*)MAGIC, 4});
    for (size_t i = 0; i < hdr.size(); i++) {
        iov.push_back({(void*)hdr[i].data(), hdr[i].size()});
        if (!cols[i].page.empty()) iov.push_back({(void*)cols[i].page.data(), cols[i].page.size()});
    }
    iov.push_back({(void*)tail.data(), tail.size()});
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) die("cannot write output");
    size_t k = 0;
    while (k < iov.size()) {
        ssize_t w = writev(fd, iov.data() + k, (int)min<size_t>(iov.size() - k, 1024));
        if (w <= 0) die("write failed");
        size_t left = (size_t)w;
        while (k < iov.size() && left >= iov[k].iov_len) left -= iov[k].iov_len, k++;
        if (left) iov[k].iov_base = (char*)iov[k].iov_base + left, iov[k].iov_len -= left;
    }
    close(fd);
}
static void rle_levels(string& page, const vector<u8>& def) {
    string rle;
    TW t;
    size_t i = 0, n = def.size();
    while (i < n) {
        u8 d = def[i];
        size_t j = i;
        while (j < n && def[j] == d) j++;
        t.b.clear();
        t.uv((u64)(j - i) << 1);
        rle += t.b;
        rle.push_back((char)d);
        i = j;
    }
    put32(page, (u32)rle.size());
    page += rle;
}
static void page_strings(string& page, const vector<string>& v) {
    for (auto& s : v) put32(page, (u32)s.size()), page += s;
}

// ---------------------------------------------------------------------------
// Shared inputs
// ---------------------------------------------------------------------------
static i64 WS, WE, DAY0;
static int NDAYS;
static FMap PRODUCTS;            // product key -> category index
static vector<string> CATNAMES;  // sorted byte-wise; index = rank
static int UNKNOWN_CAT;
static FMap CAMPAIGNS;                   // campaign key -> 1..N
static vector<string> CAMP_ID, CAMP_CH;  // index 0 = organic
static vector<Mapped> EVF, CLF, LKF;
// IDs of the files' own shape (a short prefix and a canonical decimal number,
// like p123 or c_45) are looked up by the number in a small array. The map
// from such a string to (prefix, number) is one-to-one, so a hit or a miss
// there is exact; any other string falls back to the hash map.
struct NumIdx {
    char pre[4] = {0, 0, 0, 0};
    u32 plen = 0;
    vector<u32> v;  // number -> value + 1, 0 = no such ID
    bool on = false;
    void build(const FMap& m, u32 add) {
        string first;
        for (auto& e : m.t)
            if (e.k) { first = key_str(e.k); break; }
        size_t d = 0;
        while (d < first.size() && !D(first[d])) d++;
        if (first.empty() || d == 0 || d > 4 || d == first.size()) return;
        plen = (u32)d;
        memcpy(pre, first.data(), d);
        for (auto& e : m.t) {
            if (!e.k) continue;
            string sk = key_str(e.k);
            u64 num;
            if (!parse(sk.data(), (u32)sk.size(), num)) continue;
            if (num >= (1u << 20)) continue;  // such IDs are never found by number: see find
            if (v.size() <= num) v.resize(max<size_t>(num + 1, v.size() * 2), 0);
            v[num] = (u32)e.v + add;
        }
        on = true;
    }
    inline bool parse(const char* p, u32 n, u64& num) const {
        if (n <= plen || n > plen + 6 || memcmp(p, pre, plen) != 0) return false;
        const char* q = p + plen;
        const u32 nd = n - plen;
        if (q[0] == '0' && nd > 1) return false;
        u64 x = 0;
        for (u32 i = 0; i < nd; i++) {
            const u32 c = (u8)(q[i] - '0');
            if (c > 9) return false;
            x = x * 10 + c;
        }
        num = x;
        return true;
    }
    // value + add for an ID of the shape, 0 if no such ID, -1 if not of the shape
    inline i64 find(const char* p, u32 n) const {
        u64 num;
        if (!on || !parse(p, n, num) || num >= (1u << 20)) return -1;
        return num < v.size() ? (i64)v[num] : 0;
    }
};
static NumIdx PIDX, CIDX;

static size_t EV_BYTES, CL_BYTES, LK_BYTES;
static string OUT_DIR;

static string env_or(const char* k, const char* d) {
    const char* v = getenv(k);
    return v && *v ? string(v) : string(d);
}
static bool find_json_ts(const string& js, const char* key, i64& out) {
    size_t k = js.find(string("\"") + key + "\"");
    if (k == string::npos) return false;
    size_t c = js.find(':', k);
    if (c == string::npos) return false;
    size_t q1 = js.find('"', c + 1);
    if (q1 == string::npos) return false;
    size_t q2 = js.find('"', q1 + 1);
    if (q2 == string::npos) return false;
    return parse_ts(js.data() + q1 + 1, q2 - q1 - 1, out);
}

static void load_inputs(const string& DATA) {
    {
        string js = read_small(DATA + "/manifest.json");
        if (!find_json_ts(js, "window_start", WS) || !find_json_ts(js, "window_end", WE)) die("bad manifest");
        DAY0 = floordiv(WS, DAY_MS);
        i64 last = WE > WS ? floordiv(WE - 1, DAY_MS) : DAY0;
        NDAYS = (int)max<i64>(1, last - DAY0 + 1);
        init_dates(WS, WE);
    }
    {
        string text = read_small(DATA + "/products.csv");
        // distinct category names, found by a short scan (few categories) or a map
        vector<string> distinct;
        unordered_map<string, int> big;
        auto cat_id = [&](const char* s, u32 n) -> int {
            if (distinct.size() <= 64) {
                for (size_t i = 0; i < distinct.size(); i++)
                    if (distinct[i].size() == n && memcmp(distinct[i].data(), s, n) == 0) return (int)i;
                distinct.emplace_back(s, n);
                if (distinct.size() > 64)
                    for (size_t i = 0; i < distinct.size(); i++) big[distinct[i]] = (int)i;
                return (int)distinct.size() - 1;
            }
            string k(s, n);
            auto it = big.find(k);
            if (it != big.end()) return it->second;
            distinct.push_back(k);
            big[k] = (int)distinct.size() - 1;
            return (int)distinct.size() - 1;
        };
        vector<pair<u64, int>> rows;
        rows.reserve(text.size() / 12);
        for_small_lines(text, [&](const char* p, const char* e) {
            const char* f[2];
            u32 l[2];
            if (!split<2>(p, e, f, l) || !l[0]) return;
            rows.push_back({key_of(f[0], l[0]), cat_id(f[1], l[1])});
        });
        int unk = cat_id("unknown", 7);
        vector<int> order(distinct.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = (int)i;
        sort(order.begin(), order.end(), [&](int a, int b) { return distinct[a] < distinct[b]; });
        vector<int> rank(distinct.size());
        CATNAMES.clear();
        for (size_t r = 0; r < order.size(); r++) rank[order[r]] = (int)r, CATNAMES.push_back(distinct[order[r]]);
        UNKNOWN_CAT = rank[unk];
        PRODUCTS.init(rows.size());
        for (auto& r : rows) {
            KV* s = PRODUCTS.slot(r.first);
            if (!s->k) s->k = r.first, s->v = (u64)rank[r.second];
        }
        PIDX.build(PRODUCTS, 1);
    }
    {
        string text = read_small(DATA + "/campaigns.csv");
        CAMP_ID.push_back("organic");
        CAMP_CH.push_back("organic");
        vector<pair<u64, string>> rows;
        for_small_lines(text, [&](const char* p, const char* e) {
            const char* f[2];
            u32 l[2];
            if (!split<2>(p, e, f, l) || !l[0]) return;
            rows.push_back({key_of(f[0], l[0]), string(f[1], l[1])});
        });
        CAMPAIGNS.init(rows.size());
        for (auto& r : rows) {
            KV* s = CAMPAIGNS.slot(r.first);
            if (s->k) continue;
            s->k = r.first;
            s->v = CAMP_ID.size();
            CAMP_ID.push_back(key_str(r.first));
            CAMP_CH.push_back(r.second);
        }
        CIDX.build(CAMPAIGNS, 0);
    }
    auto map_all = [&](const string& dir, size_t& bytes) {
        vector<Mapped> v;
        bytes = 0;
        for (auto& p : list_csv(dir)) {
            v.push_back(map_file(p));
            bytes += v.back().n;
        }
        return v;
    };
    EVF = map_all(DATA + "/events", EV_BYTES);
    CLF = map_all(DATA + "/clicks", CL_BYTES);
    LKF = map_all(DATA + "/links", LK_BYTES);
}

static void write_campaign_daily(const vector<i64>& cs, const vector<i64>& cps, const vector<i64>& crev,
                                 const vector<i64>& cas) {
    const int NCAMP = (int)CAMP_ID.size();
    vector<string> id, ch;
    vector<i32> day;
    vector<i64> a, b, c, d;
    for (int k = 0; k < NCAMP; k++)
        for (int di = 0; di < NDAYS; di++) {
            size_t x = (size_t)k * NDAYS + di;
            if (!cs[x] && !cas[x]) continue;
            id.push_back(CAMP_ID[k]);
            ch.push_back(CAMP_CH[k]);
            day.push_back((i32)(DAY0 + di));
            a.push_back(cs[x]);
            b.push_back(cps[x]);
            c.push_back(crev[x]);
            d.push_back(cas[x]);
        }
    size_t n = id.size();
    vector<ColData> cols(7);
    cols[0] = {"campaign_id", PT_BYTES, false, 0, 1, ""};
    page_strings(cols[0].page, id);
    cols[1] = {"channel", PT_BYTES, false, 0, 1, ""};
    page_strings(cols[1].page, ch);
    cols[2] = {"day", PT_INT32, false, 6, 6, ""};
    cols[2].page.append((const char*)day.data(), n * 4);
    cols[3] = {"sessions", PT_INT64, false, -1, 0, ""};
    cols[3].page.append((const char*)a.data(), n * 8);
    cols[4] = {"purchasing_sessions", PT_INT64, false, -1, 0, ""};
    cols[4].page.append((const char*)b.data(), n * 8);
    cols[5] = {"revenue_cents", PT_INT64, false, -1, 0, ""};
    cols[5].page.append((const char*)c.data(), n * 8);
    cols[6] = {"assisted_sessions", PT_INT64, false, -1, 0, ""};
    cols[6].page.append((const char*)d.data(), n * 8);
    write_parquet(OUT_DIR + "/campaign_daily/part-0.parquet", (i64)n, cols);
}


// Huge-page backed, zero-filled array (fresh anonymous memory is already zero).
template <class T>
static T* huge_zeroed(size_t n) {
    size_t bytes = max<size_t>(n * sizeof(T), 4096);
    size_t big = bytes + ((size_t)2 << 20);
    void* m = mmap(nullptr, big, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (m == MAP_FAILED) die("mmap failed");
    char* a = (char*)(((uintptr_t)m + ((2 << 20) - 1)) & ~(uintptr_t)((2 << 20) - 1));
    madvise(a, bytes, MADV_HUGEPAGE);
    return (T*)a;
}
struct KVArr {
    KV* p = nullptr;
    size_t n = 0;
    KV* data() const { return p; }
};
// LMF: every linked ID -> its person, one open-addressing table (linear
// probing from mix64(key) & LMASK, at most about 2/3 full). An ID that is not
// in it is its own person.
static KVArr LMF;
static u64 LMASK;
static inline u64 person_of(u64 k) {
    u64 s = mix64(k) & LMASK;
    const KV* t = LMF.data();
    for (;;) {
        const KV& e = t[s];
        if (e.k == k) return e.v;
        if (!e.k) return k;
        s = (s + 1) & LMASK;
    }
}
static inline void person_prefetch(u64 k) { __builtin_prefetch(LMF.data() + (mix64(k) & LMASK)); }

#ifdef SS_PROF
static atomic<u64> PROFC[8];
#define PROF(i) do { u64 _t = __rdtsc(); PROFC[i] += _t - _pl; _pl = _t; } while (0)
#else
#define PROF(i) do {} while (0)
#endif
static void build_links();  // defined after namespace fast (uses its line scanner)

// ===========================================================================
// generic: the straightforward implementation (64-bit times, std::sort).
// ===========================================================================
namespace generic {

static const int PB = 10;
static const int NP = 1 << PB;
static inline int part_of(u64 k) { return (int)(mix64(k) >> (64 - PB)); }

struct EvRec {
    u64 eid;
    i64 ingest;
    u64 user;
    i64 ts;
    u32 price;
    u16 cat;
    u8 flags;  // 1 = valid (kept if this copy wins), 2 = purchase
    u8 pad;
};
struct PRec {
    u64 person;
    i64 ts;
    u32 price;
    u32 cat;
    u8 purch;
};
struct CRec {
    u64 person;
    i64 ts;
    u64 cid;
    u32 camp;
};
struct Out {
    vector<u64> uid;
    vector<i32> day;
    vector<i64> ses, evs, rev;
    vector<i32> top;
    vector<i64> cs, cps, crev, cas;
    vector<int> cnt, pos, active;
    vector<i64> catsum;
    vector<int> touched;
};

static void run() {
    const int NCAMP = (int)CAMP_ID.size();
    const int NCAT = (int)CATNAMES.size();
    size_t total_bytes = EV_BYTES + CL_BYTES;
    size_t chunk = max<size_t>(256 << 10, min<size_t>(8 << 20, total_bytes / (NT * 24) + 1));

    vector<vector<EvRec>> EB((size_t)NT * NP);
    vector<vector<CRec>> CB((size_t)NT * NP);
    {
        vector<Task> tasks;
        add_tasks(tasks, EVF, 0, chunk);
        add_tasks(tasks, CLF, 1, chunk);
        const i64 CWS = WS - WEEK_MS;
        par_for((int)tasks.size(), [&](int i, int tid) {
            const Task& t = tasks[i];
            vector<EvRec>* eb = &EB[(size_t)tid * NP];
            vector<CRec>* cb = &CB[(size_t)tid * NP];
            if (t.kind == 0) {
                for_lines(t, [&](const char* p, const char* e) {
                    const char* f[7];
                    u32 l[7];
                    if (!split<7>(p, e, f, l) || !l[0]) return;
                    EvRec r;
                    r.eid = key_of(f[0], l[0]);
                    i64 ing;
                    if (!parse_ts(f[3], l[3], ing)) ing = INT64_MAX;
                    r.ingest = ing;
                    r.user = 0;
                    r.ts = 0;
                    r.price = 0;
                    r.cat = 0;
                    r.flags = 0;
                    r.pad = 0;
                    do {
                        if (!l[1]) break;
                        i64 ts;
                        if (!parse_ts(f[2], l[2], ts) || ts < WS || ts >= WE) break;
                        int type;
                        if (l[4] == 4 && memcmp(f[4], "view", 4) == 0)
                            type = 0;
                        else if (l[4] == 4 && memcmp(f[4], "cart", 4) == 0)
                            type = 1;
                        else if (l[4] == 8 && memcmp(f[4], "purchase", 8) == 0)
                            type = 2;
                        else
                            break;
                        if (!l[5]) break;
                        u32 price = 0;
                        if (type == 2 && !parse_price(f[6], l[6], price)) break;
                        if (ing - ts > DAY_MS) break;
                        r.user = key_of(f[1], l[1]);
                        r.ts = ts;
                        if (type == 2) {
                            r.price = price;
                            u64 pk = key_find(f[5], l[5]);
                            const KV* pe = pk ? PRODUCTS.get(pk) : nullptr;
                            r.cat = (u16)(pe ? pe->v : (u64)UNKNOWN_CAT);
                            r.flags = 3;
                        } else {
                            r.flags = 1;
                        }
                    } while (0);
                    eb[part_of(r.eid)].push_back(r);
                });
            } else {
                for_lines(t, [&](const char* p, const char* e) {
                    const char* f[4];
                    u32 l[4];
                    if (!split<4>(p, e, f, l) || !l[0] || !l[1] || !l[2]) return;
                    i64 ts;
                    if (!parse_ts(f[3], l[3], ts) || ts < CWS || ts >= WE) return;
                    u64 ck = key_find(f[2], l[2]);
                    const KV* ce = ck ? CAMPAIGNS.get(ck) : nullptr;
                    if (!ce) return;
                    CRec c;
                    c.person = person_of(key_of(f[1], l[1]));
                    c.ts = ts;
                    c.cid = key_of(f[0], l[0]);
                    c.camp = (u32)ce->v;
                    cb[part_of(c.person)].push_back(c);
                });
            }
        });
        tick("generic: parse");
    }

    vector<vector<PRec>> PBK((size_t)NT * NP);
    {
        struct DE {
            u64 eid;
            i64 ing;
            const EvRec* r;
        };
        vector<vector<DE>> tabs(NT);
        par_for(NP, [&](int p, int tid) {
            size_t n = 0;
            for (int t = 0; t < NT; t++) n += EB[(size_t)t * NP + p].size();
            if (!n) return;
            size_t cap = 16;
            while (cap < n * 2) cap <<= 1;
            auto& tab = tabs[tid];
            tab.assign(cap, DE{0, 0, nullptr});
            u64 mask = cap - 1;
            for (int t = 0; t < NT; t++)
                for (const EvRec& r : EB[(size_t)t * NP + p]) {
                    u64 i = mix64(r.eid) & mask;
                    while (tab[i].eid && tab[i].eid != r.eid) i = (i + 1) & mask;
                    DE& d = tab[i];
                    if (!d.eid)
                        d = DE{r.eid, r.ingest, &r};
                    else if (r.ingest < d.ing)
                        d.ing = r.ingest, d.r = &r;
                }
            vector<PRec>* pb = &PBK[(size_t)tid * NP];
            for (auto& d : tab) {
                if (!d.eid || !(d.r->flags & 1)) continue;
                const EvRec& r = *d.r;
                PRec q;
                q.person = person_of(r.user);
                q.ts = r.ts;
                q.price = r.price;
                q.cat = r.cat;
                q.purch = (r.flags >> 1) & 1;
                pb[part_of(q.person)].push_back(q);
            }
        });
        tabs.clear();
        for (auto& v : EB) vector<EvRec>().swap(v);
        tick("generic: dedup");
    }

    vector<Out> outs(NT);
    for (auto& o : outs) {
        size_t cells = (size_t)NCAMP * NDAYS;
        o.cs.assign(cells, 0);
        o.cps.assign(cells, 0);
        o.crev.assign(cells, 0);
        o.cas.assign(cells, 0);
        o.cnt.assign(NCAMP, 0);
        o.pos.assign(NCAMP, 0);
        o.catsum.assign(NCAT, 0);
    }
    {
        vector<size_t> psize(NP, 0);
        for (int p = 0; p < NP; p++)
            for (int t = 0; t < NT; t++) psize[p] += PBK[(size_t)t * NP + p].size();
        vector<int> order(NP);
        for (int p = 0; p < NP; p++) order[p] = p;
        sort(order.begin(), order.end(), [&](int a, int b) { return psize[a] > psize[b]; });
        par_for(NP, [&](int oi, int tid) {
            int p = order[oi];
            Out& o = outs[tid];
            vector<PRec> ev;
            ev.reserve(psize[p]);
            for (int t = 0; t < NT; t++) {
                auto& v = PBK[(size_t)t * NP + p];
                ev.insert(ev.end(), v.begin(), v.end());
                vector<PRec>().swap(v);
            }
            if (ev.empty()) return;
            sort(ev.begin(), ev.end(), [](const PRec& a, const PRec& b) {
                return a.person != b.person ? a.person < b.person : a.ts < b.ts;
            });
            vector<CRec> cl;
            for (int t = 0; t < NT; t++) {
                auto& v = CB[(size_t)t * NP + p];
                cl.insert(cl.end(), v.begin(), v.end());
            }
            sort(cl.begin(), cl.end(), [](const CRec& a, const CRec& b) {
                if (a.person != b.person) return a.person < b.person;
                if (a.ts != b.ts) return a.ts < b.ts;
                if (a.cid == b.cid) return false;
                return key_less(b.cid, a.cid);  // smallest click_id last
            });
            size_t ne = ev.size(), nc = cl.size(), cp = 0;
            size_t i = 0;
            while (i < ne) {
                u64 person = ev[i].person;
                size_t j = i;
                while (j < ne && ev[j].person == person) j++;
                while (cp < nc && cl[cp].person < person) cp++;
                size_t ca = cp;
                while (cp < nc && cl[cp].person == person) cp++;
                size_t cb = cp, lo = ca, hi = ca;
                i64 cur_day = INT64_MIN;
                i64 d_ses = 0, d_evs = 0, d_rev = 0;
                auto flush_day = [&]() {
                    if (cur_day == INT64_MIN) return;
                    int best = -1;
                    i64 bestv = -1;
                    for (int c : o.touched) {
                        i64 v = o.catsum[c];
                        if (v > bestv || (v == bestv && c < best)) bestv = v, best = c;
                        o.catsum[c] = 0;
                    }
                    o.touched.clear();
                    o.uid.push_back(person);
                    o.day.push_back((i32)cur_day);
                    o.ses.push_back(d_ses);
                    o.evs.push_back(d_evs);
                    o.rev.push_back(d_rev);
                    o.top.push_back(best);
                    d_ses = d_evs = d_rev = 0;
                };
                size_t s = i;
                while (s < j) {
                    const i64 start = ev[s].ts;
                    size_t e = s + 1;
                    while (e < j && ev[e].ts - ev[e - 1].ts <= GAP_MS) e++;
                    const i64 day = floordiv(start, DAY_MS);
                    if (day != cur_day) {
                        flush_day();
                        cur_day = day;
                    }
                    i64 rev = 0;
                    bool purch = false;
                    for (size_t k = s; k < e; k++)
                        if (ev[k].purch) {
                            purch = true;
                            rev += ev[k].price;
                            u32 c = ev[k].cat;
                            if (o.catsum[c] == 0) o.touched.push_back((int)c);
                            o.catsum[c] += ev[k].price;
                        }
                    d_ses += 1;
                    d_evs += (i64)(e - s);
                    d_rev += rev;
                    while (hi < cb && cl[hi].ts <= start) {
                        u32 c = cl[hi].camp;
                        if (o.cnt[c]++ == 0) o.pos[c] = (int)o.active.size(), o.active.push_back((int)c);
                        hi++;
                    }
                    while (lo < hi && cl[lo].ts < start - WEEK_MS) {
                        u32 c = cl[lo].camp;
                        if (--o.cnt[c] == 0) {
                            int at = o.pos[c], last = o.active.back();
                            o.active[at] = last;
                            o.pos[last] = at;
                            o.active.pop_back();
                        }
                        lo++;
                    }
                    u32 credited = hi > lo ? cl[hi - 1].camp : 0;
                    size_t di = (size_t)(day - DAY0);
                    size_t cell = (size_t)credited * NDAYS + di;
                    o.cs[cell] += 1;
                    o.cps[cell] += purch;
                    o.crev[cell] += rev;
                    for (int c : o.active) o.cas[(size_t)c * NDAYS + di] += 1;
                    s = e;
                }
                flush_day();
                for (int c : o.active) o.cnt[c] = 0;
                o.active.clear();
                i = j;
            }
        });
        tick("generic: sessions");
    }

    {
        size_t total_rows = 0;
        for (auto& o : outs) total_rows += o.uid.size();
        par_for(NT, [&](int t, int) {
            Out& o = outs[t];
            size_t n = o.uid.size();
            if (n == 0 && !(t == 0 && total_rows == 0)) return;
            vector<ColData> cols(6);
            cols[0] = {"user_id", PT_BYTES, false, 0, 1, ""};
            for (size_t r = 0; r < n; r++) put_key(cols[0].page, o.uid[r]);
            cols[1] = {"day", PT_INT32, false, 6, 6, ""};
            cols[1].page.append((const char*)o.day.data(), n * 4);
            cols[2] = {"sessions", PT_INT64, false, -1, 0, ""};
            cols[2].page.append((const char*)o.ses.data(), n * 8);
            cols[3] = {"events", PT_INT64, false, -1, 0, ""};
            cols[3].page.append((const char*)o.evs.data(), n * 8);
            cols[4] = {"revenue_cents", PT_INT64, false, -1, 0, ""};
            cols[4].page.append((const char*)o.rev.data(), n * 8);
            cols[5] = {"top_category", PT_BYTES, true, 0, 1, ""};
            vector<u8> def(n);
            for (size_t r = 0; r < n; r++) def[r] = o.top[r] >= 0;
            rle_levels(cols[5].page, def);
            for (size_t r = 0; r < n; r++)
                if (o.top[r] >= 0) {
                    const string& s = CATNAMES[o.top[r]];
                    put32(cols[5].page, (u32)s.size());
                    cols[5].page += s;
                }
            write_parquet(OUT_DIR + "/user_daily/part-" + to_string(t) + ".parquet", (i64)n, cols);
        });
    }
    {
        size_t cells = (size_t)NCAMP * NDAYS;
        vector<i64> cs(cells, 0), cps(cells, 0), crev(cells, 0), cas(cells, 0);
        for (auto& o : outs)
            for (size_t x = 0; x < cells; x++) {
                cs[x] += o.cs[x];
                cps[x] += o.cps[x];
                crev[x] += o.crev[x];
                cas[x] += o.cas[x];
            }
        write_campaign_daily(cs, cps, crev, cas);
    }
    tick("generic: write");
}

}  // namespace generic

// ===========================================================================
// fast: compact records, AVX2 scanning, block lists, bucketed grouping.
// Times are u32 milliseconds since CBASE = window_start - 7 days.
// ===========================================================================
namespace fast {

static const u32 BAD_TS = 0xFFFFFFFFu;
static i64 CBASE;
static u64 DAYBASE;  // day index = (ts + DAYBASE) / DAY_MS for event times
static int PE_BITS, NPE, PP_BITS, NPP;

struct EvRec {
    u64 eid;
    u64 user;
    i64 ingest;
    u32 ts;   // BAD_TS = invalid copy
    u32 aux;  // purchase: price | cat << 24; else 0
};
struct PRec {
    u64 person;
    u32 ts;
    u32 aux;
};
struct CRec {
    u64 person;
    u64 cid;
    u32 ts;
    u32 camp;
};
static_assert(sizeof(EvRec) == 32 && sizeof(PRec) == 16 && sizeof(CRec) == 24, "record sizes");

// ---- memory: per-thread slabs and block lists ----------------------------------
struct Slab {
    char* p = nullptr;
    size_t left = 0;
    vector<pair<char*, size_t>> regions;  // for releasing memory before exit
};
static void* slab_alloc(Slab& s, size_t n) {
    n = (n + 63) & ~(size_t)63;
    if (s.left < n) {
        size_t sz = max<size_t>(n, (size_t)64 << 20);
        size_t big = sz + ((size_t)2 << 20);
        void* m = mmap(nullptr, big, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) die("mmap failed");
        char* a = (char*)(((uintptr_t)m + ((2 << 20) - 1)) & ~(uintptr_t)((2 << 20) - 1));
        madvise(a, sz, MADV_HUGEPAGE);
        s.p = a;
        s.left = sz;
        s.regions.push_back({a, sz});
    }
    void* r = s.p;
    s.p += n;
    s.left -= n;
    return r;
}
#ifndef SS_STG
#define SS_STG 2
#endif
struct alignas(32 * SS_STG < 64 ? 64 : 32 * SS_STG) Blk {  // block data aligned to a whole staging buffer
    Blk* next;
    u32 n;
    u32 pad;
};
template <class R>
static inline R* bdata(Blk* b) {
    return (R*)(b + 1);
}
template <class R>
struct Scatter {
    int np = 0;
    u32 brec = 0;
    Slab* slab = nullptr;
    struct CE {
        R* cur;
        R* end;
    };
    vector<CE> ce;  // write pointer and block end, together in one line
    vector<Blk*> head, tail;
    // 32-byte records are staged per partition in a cache line and written to
    // the block a whole line at a time with a non-temporal store: no
    // read-for-ownership of the (thousands of scattered) destination lines.
    // cur advances per record; its parity (block data is line aligned) tells
    // which half of the staging line the record goes to.
    static constexpr bool NT = sizeof(R) == 32;
    static constexpr int S = SS_STG;  // records staged per partition, streamed together
    struct alignas(64) Line {
        R r[S];
    };
    vector<Line> stg;
    void init(int np_, Slab* s, u32 recs) {
        np = np_;
        slab = s;
        brec = recs;
        if (NT) stg.assign(np, Line{});
        ce.assign(np, CE{nullptr, nullptr});
        head.assign(np, nullptr);
        tail.assign(np, nullptr);
    }
    vector<Blk*> free_blocks;  // recycled blocks of exactly this block size
    __attribute__((noinline)) void grow(int p) {
        if (tail[p]) tail[p]->n = brec;
        Blk* b;
        if (!free_blocks.empty()) {
            b = free_blocks.back();
            free_blocks.pop_back();
        } else
            b = (Blk*)slab_alloc(*slab, sizeof(Blk) + (size_t)brec * sizeof(R));
        b->next = nullptr;
        b->n = 0;
        if (tail[p])
            tail[p]->next = b;
        else
            head[p] = b;
        tail[p] = b;
        ce[p].cur = bdata<R>(b);
        ce[p].end = ce[p].cur + brec;
    }
    inline void push(int p, const R& r) {
        if constexpr (NT) {
            R* c = ce[p].cur;
            const u32 i = (u32)((uintptr_t)c >> 5) & (S - 1);
            if (__builtin_expect(c == ce[p].end, 0)) {  // only ever at i == 0
                grow(p);
                c = ce[p].cur;
            }
            stg[p].r[i] = r;
            if (i == S - 1) {
                R* d = c - (S - 1);
#ifdef __AVX512F__
                for (int k = 0; k < S / 2; k++)
                    _mm512_stream_si512((__m512i*)(d + 2 * k), _mm512_load_si512((const void*)&stg[p].r[2 * k]));
#else
                for (int k = 0; k < S; k++)
                    _mm256_stream_si256((__m256i*)(d + k), _mm256_load_si256((const __m256i*)&stg[p].r[k]));
#endif
            }
            ce[p].cur = c + 1;
            return;
        }
        R* c = ce[p].cur;
        if (__builtin_expect(c == ce[p].end, 0)) {
            grow(p);
            c = ce[p].cur;
        }
        // the block's next lines are written soon: prefetch them for ownership
        // (thousands of scattered streams defeat the hardware prefetcher)
        __builtin_prefetch((const char*)c + 192, 1, 3);
        *c = r;
        ce[p].cur = c + 1;
    }
    void finish() {
        if constexpr (NT) {
            // a staged first half whose line was never streamed
            for (int p = 0; p < np; p++) {
                R* c = ce[p].cur;
                const u32 i = (u32)((uintptr_t)c >> 5) & (S - 1);
                for (u32 k = 0; k < i; k++) c[(int)k - (int)i] = stg[p].r[k];
            }
            _mm_sfence();
        }
        for (int p = 0; p < np; p++)
            if (tail[p]) tail[p]->n = (u32)(ce[p].cur - bdata<R>(tail[p]));
    }
};
template <class R>
static size_t gather(vector<Scatter<R>>& sc, int p, vector<R>& out) {
    size_t n = 0;
    for (auto& s : sc)
        for (Blk* b = s.head[p]; b; b = b->next) n += b->n;
    if (out.size() < n) out.resize(n + n / 4 + 16);
    R* o = out.data();
    for (auto& s : sc)
        for (Blk* b = s.head[p]; b; b = b->next) {
            memcpy(o, bdata<R>(b), (size_t)b->n * sizeof(R));
            o += b->n;
        }
    return n;
}

// ---- line scanning -------------------------------------------------------------
// Scans the line at p: returns its length (up to '\n' or bufend); records the
// offsets of up to 8 commas in cpos, the comma count in nc and the quote count in nq.
__attribute__((always_inline)) static inline size_t scan_line(const char* p, const char* bufend, u32* cpos, int& nc, int& nq) {
    nc = 0;
    nq = 0;

    const __m256i NLv = _mm256_set1_epi8('\n'), CMv = _mm256_set1_epi8(','), QTv = _mm256_set1_epi8('"');
    size_t off = 0;
    while (p + off + 32 <= bufend) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(p + off));
        u32 mn = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, NLv));
        u32 mc = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, CMv));
        u32 mq = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, QTv));
        u32 lim = mn ? ((mn & (0u - mn)) - 1) : 0xFFFFFFFFu;
        mc &= lim;
        nq += __builtin_popcount(mq & lim);
        while (mc) {
            u32 b = (u32)__builtin_ctz(mc);
            if (nc < 8) cpos[nc] = (u32)off + b;
            nc++;
            mc &= mc - 1;
        }
        if (mn) return off + (u32)__builtin_ctz(mn);
        off += 32;
    }
    for (; p + off < bufend; off++) {
        char c = p[off];
        if (c == '\n') return off;
        if (c == ',') {
            if (nc < 8) cpos[nc] = (u32)off;
            nc++;
        } else if (c == '"')
            nq++;
    }
    return off;
}
// Fields [fs, fe) from comma positions; strips quotes; false if malformed.
template <int NF>
static inline bool fields(const char* p, size_t len, const u32* cp, int nc, int nq, u32* fs, u32* fe) {
    if (nc != NF - 1) return false;
    fs[0] = 0;
    for (int i = 0; i < NF - 1; i++) fe[i] = cp[i], fs[i + 1] = cp[i] + 1;
    fe[NF - 1] = (u32)len;
    if (nq) {
        // Branchless: a field is either wrapped in quotes (2) or has none.
        // p[e-1] of an empty field reads the delimiter before it (data lines
        // always follow a '\n'), so it is never a quote.
        u32 bad = 0, contrib = 0;
        for (int i = 0; i < NF; i++) {
            u32 s = fs[i], e = fe[i];
            u32 ne = e > s;
            u32 qs = ne & (p[s] == '"');
            u32 qe = ne & (p[(ptrdiff_t)e - 1] == '"');
            bad |= (qs ^ qe) | (qs & (e - s < 2));
            fs[i] = s + qs;
            fe[i] = e - qe;
            contrib += qs + qe;
        }
        if (bad | (contrib != (u32)nq)) return false;
    }
    return true;
}
template <class F>
__attribute__((flatten)) static void for_lines_fast(const Task& t, F&& f) {
    const char* b = t.b;
    const char* bufend = b + t.size;
    size_t pos = first_line(t);
    u32 cpos[8];
    int nc, nq;
    while (pos < t.end) {
        size_t len = scan_line(b + pos, bufend, cpos, nc, nq);
        f(b + pos, len, cpos, nc, nq);
        pos += len + 1;
    }
}

// Event lines (7 fields): the whole line is classified from 128 bytes of
// masks at once. Calls g(p, fs, fe) for every well-formed line, with the field
// bounds already stripped of their quotes. Lines of 128+ bytes and lines near
// the end of the buffer take the general path (scan_line + fields).
typedef unsigned __int128 u128;
static inline u32 ctz128(u128 x) {
    const u64 lo = (u64)x;
    return lo ? (u32)__builtin_ctzll(lo) : 64 + (u32)__builtin_ctzll((u64)(x >> 64));
}
static inline u64 pxor64(u64 v) {
    v ^= v << 1;
    v ^= v << 2;
    v ^= v << 4;
    v ^= v << 8;
    v ^= v << 16;
    v ^= v << 32;
    return v;
}
template <class G>
__attribute__((flatten)) static void for_lines7(const Task& t, G&& g) {
    const char* b = t.b;
    const char* bufend = b + t.size;
    size_t pos = first_line(t);
    const __m256i NLv = _mm256_set1_epi8('\n'), CMv = _mm256_set1_epi8(','), QTv = _mm256_set1_epi8('"');
    u32 fs[7], fe[7];
    while (pos < t.end) {
        const char* p = b + pos;
        __builtin_prefetch(p + 8192);  // the input a few pages ahead: page cache pages are 4 KB, past the reach of the hardware prefetcher
        if (__builtin_expect(p + 128 <= bufend, 1)) {
#ifndef SS_YES512
#define SS_NO512
#endif
#if defined(__AVX512BW__) && !defined(SS_NO512)
            const __m512i w0 = _mm512_loadu_si512((const void*)p);
            const __m512i w1 = _mm512_loadu_si512((const void*)(p + 64));
#define SS_MASK64(a, c0, c1) ((u64)_mm512_cmpeq_epi8_mask(c0, _mm512_broadcast_i64x4(a)))
#define v0 w0
#define v1 w0
#define v2 w1
#define v3 w1
#else
            const __m256i v0 = _mm256_loadu_si256((const __m256i*)p);
            const __m256i v1 = _mm256_loadu_si256((const __m256i*)(p + 32));
            const __m256i v2 = _mm256_loadu_si256((const __m256i*)(p + 64));
            const __m256i v3 = _mm256_loadu_si256((const __m256i*)(p + 96));
#define SS_MASK64(a, c0, c1) \
    ((u64)(u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(c0, a)) | ((u64)(u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, a)) << 32))
#endif
            const u64 n0 = SS_MASK64(NLv, v0, v1), n1 = SS_MASK64(NLv, v2, v3);
            if (__builtin_expect((n0 | n1) != 0, 1)) {
                const u32 L = n0 ? (u32)__builtin_ctzll(n0) : 64 + (u32)__builtin_ctzll(n1);
                const u128 lim = ((u128)1 << L) - 1;
                const u128 cm = ((u128)SS_MASK64(CMv, v2, v3) << 64 | SS_MASK64(CMv, v0, v1)) & lim;
                pos += L + 1;
                const u64 cmlo = (u64)cm, cmhi = (u64)(cm >> 64);
                if (__builtin_popcountll(cmlo) + __builtin_popcountll(cmhi) != 6) continue;
                u128 x = cm;
                fs[0] = 0;
                for (int k = 0; k < 6; k++) {
                    const u32 c = ctz128(x);
                    x &= x - 1;
                    fe[k] = c;
                    fs[k + 1] = c + 1;
                }
                fe[6] = L;
                const u128 qt = ((u128)SS_MASK64(QTv, v2, v3) << 64 | SS_MASK64(QTv, v0, v1)) & lim;
                if (qt) {
                    // Quotes only at a field's first or last byte, and an even
                    // number of them before every separator and the line end.
                    const u128 FS = (cm << 1) | 1, FE = (cm >> 1) | ((u128)1 << (L - 1));
                    if (qt & ~(FS | FE)) continue;
                    const u64 qlo = (u64)qt, qhi = (u64)(qt >> 64);
                    const u64 ilo = pxor64(qlo), ihi = pxor64(qhi) ^ (0 - (u64)(__builtin_popcountll(qlo) & 1));
                    const u128 in = ((u128)ihi << 64) | ilo;
                    if (in & (cm | ((u128)1 << L))) continue;
                    for (int k = 0; k < 7; k++) {
                        const u32 q = (u32)(qt >> fs[k]) & 1;
                        fs[k] += q;
                        fe[k] -= q;
                    }
                }
#undef SS_MASK64
#if defined(__AVX512BW__) && !defined(SS_NO512)
#undef v0
#undef v1
#undef v2
#undef v3
#endif
                g(p, fs, fe);
                continue;
            }
        }
        u32 cpos[8];
        int nc, nq;
        const size_t len = scan_line(p, bufend, cpos, nc, nq);
        if (fields<7>(p, len, cpos, nc, nq, fs, fe)) g(p, (const u32*)fs, (const u32*)fe);
        pos += len + 1;
    }
}

// Short lines (clicks, links): the same idea as for_lines7 with a 64-byte
// window. Lines of 64+ bytes or near the buffer end take the general path.
template <int NF, class G>
__attribute__((flatten)) static void for_lines_short(const Task& t, G&& g) {
    const char* b = t.b;
    const char* bufend = b + t.size;
    size_t pos = first_line(t);
    const __m256i NLv = _mm256_set1_epi8('\n'), CMv = _mm256_set1_epi8(','), QTv = _mm256_set1_epi8('"');
    u32 fs[NF], fe[NF];
    while (pos < t.end) {
        const char* p = b + pos;
        __builtin_prefetch(p + 4096);
        if (__builtin_expect(p + 64 <= bufend, 1)) {
#if defined(__AVX512BW__) && !defined(SS_NO512)
            const __m512i w0 = _mm512_loadu_si512((const void*)p);
#define SS_MASK64(a, c0, c1) ((u64)_mm512_cmpeq_epi8_mask(w0, _mm512_broadcast_i64x4(a)))
#else
            const __m256i v0 = _mm256_loadu_si256((const __m256i*)p);
            const __m256i v1 = _mm256_loadu_si256((const __m256i*)(p + 32));
#define SS_MASK64(a, c0, c1) \
    ((u64)(u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(c0, a)) | ((u64)(u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(c1, a)) << 32))
#endif
            const u64 n0 = SS_MASK64(NLv, v0, v1);
            if (__builtin_expect(n0 != 0, 1)) {
                const u32 L = (u32)__builtin_ctzll(n0);
                const u64 lim = (n0 - 1) & ~n0;  // bits below L
                const u64 cm = SS_MASK64(CMv, v0, v1) & lim;
                pos += L + 1;
                if (__builtin_popcountll(cm) != NF - 1) continue;
                u64 x = cm;
                fs[0] = 0;
                for (int k = 0; k < NF - 1; k++) {
                    const u32 c = (u32)__builtin_ctzll(x);
                    x &= x - 1;
                    fe[k] = c;
                    fs[k + 1] = c + 1;
                }
                fe[NF - 1] = L;
                const u64 qt = SS_MASK64(QTv, v0, v1) & lim;
                if (qt) {
                    const u64 FS = (cm << 1) | 1, FE = (cm >> 1) | ((u64)1 << (L - 1));
                    if (qt & ~(FS | FE)) continue;
                    if (pxor64(qt) & (cm | ((u64)1 << L))) continue;
                    for (int k = 0; k < NF; k++) {
                        const u32 q = (u32)(qt >> fs[k]) & 1;
                        fs[k] += q;
                        fe[k] -= q;
                    }
                }
#undef SS_MASK64
                g(p, (const u32*)fs, (const u32*)fe);
                continue;
            }
        }
        u32 cpos[8];
        int nc, nq;
        const size_t len = scan_line(p, bufend, cpos, nc, nq);
        if (fields<NF>(p, len, cpos, nc, nq, fs, fe)) g(p, (const u32*)fs, (const u32*)fe);
        pos += len + 1;
    }
}

static inline u32 load4(const char* p) {
    u32 v;
    memcpy(&v, p, 4);
    return v;
}
static inline u64 load8(const char* p) {
    u64 v;
    memcpy(&v, p, 8);
    return v;
}

// One well-formed event line (field bounds fs/fe) -> r; false if event_id is empty.
__attribute__((always_inline)) static inline bool parse_event(const char* p, const u32* fs, const u32* fe,
                                                               const char* bufend, EvRec& r) {
    u32 l0 = fe[0] - fs[0];
    if (!l0) return false;
    r.eid = key_of_fast(p + fs[0], l0, bufend);
    i64 ing;
    if (!parse_ts_fast(p + fs[3], fe[3] - fs[3], ing)) ing = INT64_MAX;
    r.ingest = ing;
    r.user = 0;
    r.ts = BAD_TS;
    r.aux = 0;
    do {
        u32 l1 = fe[1] - fs[1];
        if (!l1) break;
        i64 ts;
        if (!parse_ts_fast(p + fs[2], fe[2] - fs[2], ts) || ts < WS || ts >= WE) break;
        u32 l4 = fe[4] - fs[4];
        const char* ty = p + fs[4];
        int type;
        if (l4 == 4) {
            u32 w = load4(ty);
            if (w == 0x77656976u)  // "view"
                type = 0;
            else if (w == 0x74726163u)  // "cart"
                type = 1;
            else
                break;
        } else if (l4 == 8 && load8(ty) == 0x6573616863727570ULL)  // "purchase"
            type = 2;
        else
            break;
        if (fe[5] == fs[5]) break;
        u32 price = 0;
        if (type == 2 && !parse_price(p + fs[6], fe[6] - fs[6], price)) break;
        if (ing - ts > DAY_MS) break;
        r.user = key_of_fast(p + fs[1], l1, bufend);
        r.ts = (u32)(ts - CBASE);
        if (type == 2) {
            const i64 hit = PIDX.find(p + fs[5], fe[5] - fs[5]);
            u32 cat;
            if (hit >= 0)
                cat = hit ? (u32)(hit - 1) : (u32)UNKNOWN_CAT;
            else {
                u64 pk = key_find(p + fs[5], fe[5] - fs[5]);
                const KV* pe = pk ? PRODUCTS.get(pk) : nullptr;
                cat = (u32)(pe ? pe->v : (u64)UNKNOWN_CAT);
            }
            r.aux = price | (cat << 24);
        }
    } while (0);
    return true;
}

// ---- per-thread state -----------------------------------------------------------
struct Thr {
    Slab slab;
    Scatter<EvRec> se;
    Scatter<CRec> sc;
    Scatter<PRec> sp;
    // pass 3 outputs
    vector<u64> uid;
    vector<i32> day;
    vector<i32> ses, evs;
    vector<i64> rev;
    vector<i16> top;
    vector<i64> cs, cps, crev, cas;
    vector<int> cnt, pos, active;
    vector<i64> catsum;
    vector<int> touched;
};

static void run() {
    const int NCAMP = (int)CAMP_ID.size();
    const int NCAT = (int)CATNAMES.size();
    CBASE = WS - WEEK_MS;
    DAYBASE = (u64)(CBASE - DAY0 * DAY_MS + 4 * WEEK_MS) ;  // shifted to stay non-negative
    // day index = (ts + DAYBASE) / DAY_MS - 28 (the 4-week shift)
    const u64 est_events = EV_BYTES / 85 + 1;
    auto bits_for = [](u64 n, u64 per, int lo, int hi) {
        int b = lo;
        while (b < hi && ((u64)1 << b) * per < n) b++;
        return b;
    };
    // at most 2048 event partitions: more makes the parse scatter (per-partition
    // staging lines and write pointers) spill out of L2, which costs more than
    // the larger dedup partitions
    PE_BITS = bits_for(est_events, getenv("SS_PE") ? (u64)atoi(getenv("SS_PE")) : 12000, 6, 11);
    NPE = 1 << PE_BITS;
    PP_BITS = bits_for(est_events, getenv("SS_PP") ? (u64)atoi(getenv("SS_PP")) : 64000, 6, 13);
    NPP = 1 << PP_BITS;

    vector<Thr> th(NT);
    for (auto& t : th) {
        t.se.init(NPE, &t.slab, 512);
        static_assert(sizeof(EvRec) * 512 == sizeof(PRec) * 1024, "recycled event blocks must fit person blocks");
        t.sc.init(NPP, &t.slab, 256);
        t.sp.init(NPP, &t.slab, 1024);
    }
    // ---- links build on their own thread, overlapped with the parse -------------
    static atomic<bool> links_ready(false);
    thread links_thread([] {
        PAR_NT = getenv("SS_LINK_NT") ? max(1, atoi(getenv("SS_LINK_NT"))) : 2;
        build_links();
        links_ready.store(true, memory_order_release);
        tick("links built (overlapped)");
    });
    // ---- pass 1: parse events and clicks ------------------------------------------
    {
        size_t total = EV_BYTES + CL_BYTES;
        size_t chunk = max<size_t>(256 << 10, min<size_t>(4 << 20, total / (NT * 32) + 1));
        if (getenv("SS_CHUNK")) chunk = (size_t)atoi(getenv("SS_CHUNK")) << 10;
        vector<Task> tasks;
        add_tasks(tasks, EVF, 0, chunk);
        add_tasks(tasks, CLF, 1, chunk);
        par_for((int)tasks.size(), [&](int ti, int tid) {
            const Task& t = tasks[ti];
            const char* const bufend = t.b + t.size;
            Thr& T = th[tid];
            if (t.kind == 0) {
                Scatter<EvRec>& se = T.se;
                const int sh = 64 - PE_BITS;
                for_lines7(t, [&](const char* p, const u32* fs, const u32* fe) {
                    EvRec r;
                    if (!parse_event(p, fs, fe, bufend, r)) return;
                    se.push((int)(mix64(r.eid) >> sh), r);
                });
                _mm_sfence();  // this thread's non-temporal stores are globally visible before the join
            } else {
                // Click tasks come after every event task, so the links are
                // normally built by now. Persons are resolved in batches of 32
                // with their table slots prefetched, then scattered by person.
                Scatter<CRec>& sc = T.sc;
                const int sh = 64 - PP_BITS;
                CRec pend[32];
                u32 np = 0;
                auto flush = [&]() {
                    if (!links_ready.load(memory_order_acquire))
                        while (!links_ready.load(memory_order_acquire)) sched_yield();
                    for (u32 i = 0; i < np; i++) person_prefetch(pend[i].person);
                    for (u32 i = 0; i < np; i++) {
                        CRec x = pend[i];
                        x.person = person_of(x.person);
                        sc.push((int)(mix64(x.person) >> sh), x);
                    }
                    np = 0;
                };
                for_lines_short<4>(t, [&](const char* p, const u32* fs, const u32* fe) {
                    u32 l0 = fe[0] - fs[0], l1 = fe[1] - fs[1], l2 = fe[2] - fs[2];
                    if (!l0 || !l1 || !l2) return;
                    i64 ts;
                    if (!parse_ts_fast(p + fs[3], fe[3] - fs[3], ts) || ts < CBASE || ts >= WE) return;
                    i64 camp = CIDX.find(p + fs[2], l2);
                    if (camp < 0) {
                        u64 ck = key_find(p + fs[2], l2);
                        const KV* ce = ck ? CAMPAIGNS.get(ck) : nullptr;
                        camp = ce ? (i64)ce->v : 0;
                    }
                    if (!camp) return;
                    CRec c;
                    c.person = key_of_fast(p + fs[1], l1, bufend);  // user key for now
                    c.cid = key_of_fast(p + fs[0], l0, bufend);
                    c.ts = (u32)(ts - CBASE);
                    c.camp = (u32)camp;
                    pend[np++] = c;
                    if (np == 32) flush();
                });
                flush();
            }
        });
        for (auto& t : th) t.se.finish();
        tick("parse events+clicks");
    }
    links_thread.join();
    tick("links joined");

    // ---- pass 2: dedup events, map to persons ---------------------------------------
    {
        vector<Scatter<EvRec>> ses;
        ses.reserve(NT);
        for (auto& t : th) ses.push_back(t.se);
        vector<vector<EvRec>> bufs(NT);
        vector<vector<u32>> tabs(NT);
        vector<vector<u8>> wins(NT);
        vector<vector<u32>> survs(NT);
        par_for(NPE, [&](int p, int tid) {
            auto& buf = bufs[tid];
            size_t n = gather(ses, p, buf);
            if (!n) return;
            for (auto& sv : ses)  // these event blocks are dead now: reuse them for person records
                for (Blk* b = sv.head[p]; b;) {
                    Blk* nx = b->next;
                    th[tid].sp.free_blocks.push_back(b);
                    b = nx;
                }
            size_t cap = 16;
            while (cap < n * 2) cap <<= 1;
            auto& tab = tabs[tid];
            if (tab.size() < cap) tab.resize(cap);
            memset(tab.data(), 0, cap * sizeof(u32));
            const u64 mask = cap - 1;
            const EvRec* E = buf.data();
            auto& win = wins[tid];
            if (win.size() < n) win.resize(n + n / 4 + 16);
            u8* W = win.data();
            for (u32 i = 0; i < (u32)n; i++) {
                const u64 k = E[i].eid;
                u64 s = mix64(k) & mask;
                W[i] = 1;
                for (;;) {
                    u32 x = tab[s];
                    if (!x) {
                        tab[s] = i + 1;
                        break;
                    }
                    if (E[x - 1].eid == k) {
                        if (E[i].ingest < E[x - 1].ingest) {
                            W[x - 1] = 0;
                            tab[s] = i + 1;
                        } else
                            W[i] = 0;
                        break;
                    }
                    s = (s + 1) & mask;
                }
            }
            Scatter<PRec>& sp = th[tid].sp;
            const int sh = 64 - PP_BITS;
            // compact the survivors, then resolve persons with a steady
            // 16-ahead prefetch pipeline
            auto& sv = survs[tid];
            if (sv.size() < n) sv.resize(n + n / 4 + 16);
            u32* S = sv.data();
            u32 ns = 0;
            for (u32 x = 0; x < (u32)n; x++) {
                S[ns] = x;
                ns += W[x] & (E[x].ts != BAD_TS);
            }
            const u32 D = 16;
            for (u32 i = 0; i < ns && i < D; i++) person_prefetch(E[S[i]].user);
            for (u32 i = 0; i < ns; i++) {
                if (i + D < ns) person_prefetch(E[S[i + D]].user);
                const EvRec& r = E[S[i]];
                PRec q;
                q.person = person_of(r.user);
                q.ts = r.ts;
                q.aux = r.aux;
                sp.push((int)(mix64(q.person) >> sh), q);
            }
        });
        for (auto& t : th) t.sp.finish(), t.sc.finish();
        tick("dedup");
    }

    // ---- pass 3: per person partition -------------------------------------------------
    for (auto& t : th) {
        size_t cells = (size_t)NCAMP * NDAYS;
        t.cs.assign(cells, 0);
        t.cps.assign(cells, 0);
        t.crev.assign(cells, 0);
        t.cas.assign(cells, 0);
        t.cnt.assign(NCAMP, 0);
        t.pos.assign(NCAMP, 0);
        t.catsum.assign(NCAT, 0);
        // output rows (person-days): about 1 per 8 events; reserved so the
        // columns do not grow by copying into freshly faulted pages
        const size_t est = EV_BYTES / 512 / NT + 1024;
        t.uid.reserve(est);
        t.day.reserve(est);
        t.ses.reserve(est);
        t.evs.reserve(est);
        t.rev.reserve(est);
        t.top.reserve(est);
    }
    {
        vector<Scatter<PRec>> sps;
        vector<Scatter<CRec>> scs;
        for (auto& t : th) sps.push_back(t.sp), scs.push_back(t.sc);
        vector<size_t> psize(NPP, 0);
        for (int p = 0; p < NPP; p++)
            for (auto& s : sps)
                for (Blk* b = s.head[p]; b; b = b->next) psize[p] += b->n;
        vector<int> order(NPP);
        for (int p = 0; p < NPP; p++) order[p] = p;
        sort(order.begin(), order.end(), [&](int a, int b) { return psize[a] > psize[b]; });
        // Per partition: persons get dense local indexes; events become u64 keys
        // (idx << 32 | ts) sorted as plain integers; purchases and clicks are
        // side lists sorted by the same key, swept in step with the events.
        struct PK {
            u64 key;
            u32 aux;
            u32 pad;
        };
        struct CK {
            u64 key;
            u64 cid;
            u32 camp;
            u32 pad;
        };
        struct Local {
            vector<u64> keys, tmp;
            vector<u32> cnt;
            vector<PK> pur;
            vector<CK> ck, ck2;
            vector<u64> pid;  // local idx -> person
            vector<u64> mk;   // person -> idx map (open addressing, key 0 = empty)
            vector<u32> mv;
            u64 mmask = 0;
        };
        vector<Local> locs(NT);
        const u64 DAYSHIFT = 4 * 7;  // matches the 4-week shift in DAYBASE
        par_for(NPP, [&](int oi, int tid) {
            const int p = order[oi];
            const size_t ne = psize[p];
            if (!ne) return;
#ifdef SS_PROF
            u64 _pl = __rdtsc();
#endif
            Thr& T = th[tid];
            Local& L = locs[tid];
            if (L.keys.size() < ne) L.keys.resize(ne + ne / 4), L.tmp.resize(ne + ne / 4);
            L.pur.clear();
            L.ck.clear();
            L.pid.clear();
            size_t cap = 64;
            while (cap < ne / 4) cap <<= 1;
            L.mk.assign(cap, 0);
            L.mv.resize(cap);
            L.mmask = cap - 1;
            auto rehash = [&]() {
                vector<u64> ok;
                ok.swap(L.mk);
                vector<u32> ov;
                ov.swap(L.mv);
                const size_t ncap = ok.size() * 2;
                L.mk.assign(ncap, 0);
                L.mv.resize(ncap);
                L.mmask = ncap - 1;
                for (size_t x = 0; x < ok.size(); x++)
                    if (ok[x]) {
                        u64 h = mix64(ok[x]) & L.mmask;
                        while (L.mk[h]) h = (h + 1) & L.mmask;
                        L.mk[h] = ok[x];
                        L.mv[h] = ov[x];
                    }
            };
            u64* K = L.keys.data();
            u64* KT = L.tmp.data();
            size_t i = 0;
            for (auto& sp : sps)
                for (Blk* b = sp.head[p]; b; b = b->next) {
                    const PRec* r = bdata<PRec>(b);
                    for (u32 k = 0; k < b->n; k++) {
                        const u64 person = r[k].person;
                        u64 h = mix64(person) & L.mmask;
                        u32 idx;
                        for (;;) {
                            const u64 m = L.mk[h];
                            if (m == person) {
                                idx = L.mv[h];
                                break;
                            }
                            if (!m) {
                                idx = (u32)L.pid.size();
                                L.pid.push_back(person);
                                L.mk[h] = person;
                                L.mv[h] = idx;
                                if (L.pid.size() * 2 > L.mk.size()) rehash();
                                break;
                            }
                            h = (h + 1) & L.mmask;
                        }
                        const u64 key = ((u64)idx << 32) | r[k].ts;
                        KT[i++] = key;
                        if (r[k].aux) L.pur.push_back(PK{key, r[k].aux, 0});
                    }
                }
            PROF(0);
            // Sort by (person index, ts): a stable LSD radix sort on ts (11-bit
            // digits; a digit that is the same for every key is skipped), then a
            // stable counting pass by person index.
            {
                const size_t npers = L.pid.size();
                L.cnt.assign(npers + 1, 0);
                u32* cn = L.cnt.data();
                u32 h[3][2048];
                memset(h, 0, sizeof h);
                for (size_t x = 0; x < ne; x++) {
                    const u64 v = KT[x];
                    const u32 t = (u32)v;
                    h[0][t & 2047]++;
                    h[1][(t >> 11) & 2047]++;
                    h[2][t >> 22]++;
                    cn[(v >> 32) + 1]++;
                }
                PROF(1);
                u64* src = KT;
                u64* dst = K;
                for (int d = 0; d < 3; d++) {
                    u32* hd = h[d];
                    const int sh = 11 * d;
                    if (hd[((u32)src[0] >> sh) & 2047] == (u32)ne) continue;
                    u32 sum = 0;
                    for (int b = 0; b < 2048; b++) {
                        const u32 c = hd[b];
                        hd[b] = sum;
                        sum += c;
                    }
                    for (size_t x = 0; x < ne; x++) {
                        const u64 v = src[x];
                        dst[hd[((u32)v >> sh) & 2047]++] = v;
                    }
                    swap(src, dst);
                }
                for (size_t x = 0; x < npers; x++) cn[x + 1] += cn[x];
                for (size_t x = 0; x < ne; x++) {
                    const u64 v = src[x];
                    dst[cn[v >> 32]++] = v;
                }
                K = dst;
            }
            PROF(2);
            sort(L.pur.begin(), L.pur.end(), [](const PK& x, const PK& y) { return x.key < y.key; });
            PROF(3);
            for (auto& sc : scs)
                for (Blk* b = sc.head[p]; b; b = b->next) {
                    const CRec* c = bdata<CRec>(b);
                    for (u32 k = 0; k < b->n; k++) {
                        const u64 person = c[k].person;
                        u64 h = mix64(person) & L.mmask;
                        for (;;) {
                            const u64 m = L.mk[h];
                            if (m == person) {
                                L.ck.push_back(CK{((u64)L.mv[h] << 32) | c[k].ts, c[k].cid, c[k].camp, 0});
                                break;
                            }
                            if (!m) break;  // a person with clicks but no events
                            h = (h + 1) & L.mmask;
                        }
                    }
                }
            // clicks: sorted by (person index, ts) the same way as the events, then
            // equal keys by click_id descending, so the smallest click_id comes last
            PROF(4);
            const size_t nc = L.ck.size();
            if (L.ck2.size() < nc) L.ck2.resize(nc + nc / 4 + 16);
            const CK* C = L.ck.data();
            if (nc) {
                const size_t npers = L.pid.size();
                L.cnt.assign(npers + 1, 0);
                u32* cn = L.cnt.data();
                CK* src = L.ck.data();
                CK* dst = L.ck2.data();
                u32 h[3][2048];
                memset(h, 0, sizeof h);
                for (size_t x = 0; x < nc; x++) {
                    const u64 v = src[x].key;
                    const u32 t = (u32)v;
                    h[0][t & 2047]++;
                    h[1][(t >> 11) & 2047]++;
                    h[2][t >> 22]++;
                    cn[(v >> 32) + 1]++;
                }
                for (int d = 0; d < 3; d++) {
                    u32* hd = h[d];
                    const int sh = 11 * d;
                    if (hd[((u32)src[0].key >> sh) & 2047] == (u32)nc) continue;
                    u32 sum = 0;
                    for (int b = 0; b < 2048; b++) {
                        const u32 c = hd[b];
                        hd[b] = sum;
                        sum += c;
                    }
                    for (size_t x = 0; x < nc; x++) dst[hd[((u32)src[x].key >> sh) & 2047]++] = src[x];
                    swap(src, dst);
                }
                for (size_t x = 0; x < npers; x++) cn[x + 1] += cn[x];
                for (size_t x = 0; x < nc; x++) dst[cn[src[x].key >> 32]++] = src[x];
                auto cl = [](const CK& x, const CK& y) {
                    if (x.key != y.key) return x.key < y.key;
                    if (x.cid == y.cid) return false;
                    return key_less(y.cid, x.cid);
                };
                // runs of equal keys are short: insertion sort them in place
                for (size_t x = 1; x < nc; x++)
                    if (dst[x].key == dst[x - 1].key) {
                        const CK v = dst[x];
                        size_t y = x;
                        while (y > 0 && cl(v, dst[y - 1])) dst[y] = dst[y - 1], y--;
                        dst[y] = v;
                    }
                C = dst;
            }
            PROF(5);
            const PK* P = L.pur.data();
            const size_t npur = L.pur.size();
            size_t pp = 0, cp = 0;
            i = 0;
            while (i < ne) {
                const u64 idx = K[i] >> 32;
                size_t j = i + 1;
                while (j < ne && (K[j] >> 32) == idx) j++;
                const u64 person = L.pid[idx];
                while (cp < nc && (C[cp].key >> 32) < idx) cp++;
                const size_t cA = cp;
                while (cp < nc && (C[cp].key >> 32) == idx) cp++;
                const size_t cB = cp;
                size_t lo = cA, hi = cA;
                u32 cur_day = 0xFFFFFFFFu;
                i32 d_ses = 0, d_evs = 0;
                i64 d_rev = 0;
                auto flush_day = [&]() {
                    if (cur_day == 0xFFFFFFFFu) return;
                    int best = -1;
                    i64 bestv = -1;
                    for (int c : T.touched) {
                        i64 v = T.catsum[c];
                        if (v > bestv || (v == bestv && c < best)) bestv = v, best = c;
                        T.catsum[c] = 0;
                    }
                    T.touched.clear();
                    T.uid.push_back(person);
                    T.day.push_back((i32)(DAY0 + cur_day));
                    T.ses.push_back(d_ses);
                    T.evs.push_back(d_evs);
                    T.rev.push_back(d_rev);
                    T.top.push_back((i16)best);
                    d_ses = d_evs = 0;
                    d_rev = 0;
                };
                size_t s = i;
                while (s < j) {
                    const u32 start = (u32)K[s];
                    size_t e = s + 1;
                    while (e < j && (u32)K[e] - (u32)K[e - 1] <= (u32)GAP_MS) e++;
                    const u32 day = (u32)((start + DAYBASE) / (u64)DAY_MS - DAYSHIFT);
                    if (day != cur_day) {
                        flush_day();
                        cur_day = day;
                    }
                    i64 rev = 0;
                    u32 purch = 0;
                    const u64 last = K[e - 1];
                    while (pp < npur && P[pp].key <= last) {
                        const u32 aux = P[pp].aux;
                        const u32 price = aux & 0xFFFFFFu, c = aux >> 24;
                        purch = 1;
                        rev += price;
                        if (T.catsum[c] == 0) T.touched.push_back((int)c);
                        T.catsum[c] += price;
                        pp++;
                    }
                    d_ses += 1;
                    d_evs += (i32)(e - s);
                    d_rev += rev;
                    while (hi < cB && (u32)C[hi].key <= start) {
                        u32 c = C[hi].camp;
                        if (T.cnt[c]++ == 0) T.pos[c] = (int)T.active.size(), T.active.push_back((int)c);
                        hi++;
                    }
                    while (lo < hi && (u64)(u32)C[lo].key + (u64)WEEK_MS < (u64)start) {
                        u32 c = C[lo].camp;
                        if (--T.cnt[c] == 0) {
                            int at = T.pos[c], lastc = T.active.back();
                            T.active[at] = lastc;
                            T.pos[lastc] = at;
                            T.active.pop_back();
                        }
                        lo++;
                    }
                    u32 credited = hi > lo ? C[hi - 1].camp : 0;
                    size_t cell = (size_t)credited * NDAYS + day;
                    T.cs[cell] += 1;
                    T.cps[cell] += purch;
                    T.crev[cell] += rev;
                    for (int c : T.active) T.cas[(size_t)c * NDAYS + day] += 1;
                    s = e;
                }
                flush_day();
                for (int c : T.active) T.cnt[c] = 0;
                T.active.clear();
                i = j;
            }
            PROF(6);
        });
        tick("sessions");
#ifdef SS_PROF
        {
            const char* nm[] = {"map+keys", "count sort", "person sort", "pur sort", "click map", "click sort", "sweep"};
            for (int q = 0; q < 7; q++) fprintf(stderr, "    %-12s %8.1f Mcyc\n", nm[q], PROFC[q] / 1e6);
        }
#endif
    }

    // ---- write ------------------------------------------------------------------------
    {
        size_t total_rows = 0;
        for (auto& t : th) total_rows += t.uid.size();
        // After the NT writer tasks, release memory in parallel (MADV_DONTNEED takes
        // the mmap lock shared) so the single-threaded teardown at exit is short.
        vector<pair<const char*, size_t>> rel;
        for (auto& t : th)
            for (auto& r : t.slab.regions) rel.push_back(r);
        for (auto* v : {&EVF, &CLF, &LKF})
            for (auto& m : *v)
                for (size_t o = 0; o < m.n; o += (size_t)256 << 20)
                    rel.push_back({m.p + o, min(m.n - o, (size_t)256 << 20)});
        par_for(NT + (int)rel.size(), [&](int ti, int) {
            if (ti >= NT) {
                const auto& r = rel[ti - NT];
                madvise((void*)r.first, r.second, MADV_DONTNEED);
                return;
            }
            Thr& T = th[ti];
            size_t n = T.uid.size();
            if (n == 0 && !(ti == 0 && total_rows == 0)) return;
            vector<ColData> cols(6);
            cols[0] = {"user_id", PT_BYTES, false, 0, 1, ""};
            cols[0].page.reserve(n * 14);
            for (size_t r = 0; r < n; r++) put_key(cols[0].page, T.uid[r]);
            cols[1] = {"day", PT_INT32, false, 6, 6, ""};
            cols[1].page.append((const char*)T.day.data(), n * 4);
            cols[2] = {"sessions", PT_INT32, false, -1, 0, ""};
            cols[2].page.append((const char*)T.ses.data(), n * 4);
            cols[3] = {"events", PT_INT32, false, -1, 0, ""};
            cols[3].page.append((const char*)T.evs.data(), n * 4);
            cols[4] = {"revenue_cents", PT_INT64, false, -1, 0, ""};
            cols[4].page.append((const char*)T.rev.data(), n * 8);
            cols[5] = {"top_category", PT_BYTES, true, 0, 1, ""};
            vector<u8> def(n);
            for (size_t r = 0; r < n; r++) def[r] = T.top[r] >= 0;
            rle_levels(cols[5].page, def);
            for (size_t r = 0; r < n; r++)
                if (T.top[r] >= 0) {
                    const string& s = CATNAMES[T.top[r]];
                    put32(cols[5].page, (u32)s.size());
                    cols[5].page += s;
                }
            write_parquet(OUT_DIR + "/user_daily/part-" + to_string(ti) + ".parquet", (i64)n, cols);
        });
    }
    {
        size_t cells = (size_t)NCAMP * NDAYS;
        vector<i64> cs(cells, 0), cps(cells, 0), crev(cells, 0), cas(cells, 0);
        for (auto& t : th)
            for (size_t x = 0; x < cells; x++) {
                cs[x] += t.cs[x];
                cps[x] += t.cps[x];
                crev[x] += t.crev[x];
                cas[x] += t.cas[x];
            }
        write_campaign_daily(cs, cps, crev, cas);
    }
    tick("write");
}

}  // namespace fast

// Links -> LMF, single-threaded:
//  1. parse every link line into a flat pair array (AVX2 line scan, SIMD keys)
//  2. LMF assigns dense ids to the keys (prefetched lookups in input order)
//  3. union-find over the dense ids
//  4. each id's person = the smallest key of its component
//  5. one pass over LMF replaces ids by persons
static void build_links() {
    vector<u64> A;  // A[2j], A[2j+1]: the two keys of link j
    A.reserve(LK_BYTES / 9 + 16);
    for (auto& m : LKF) {
        if (!m.n) continue;
        Task t{m.p, m.n, 0, m.n, 2};
        const char* const bufend = m.p + m.n;
        fast::for_lines_short<2>(t, [&](const char* p, const u32* fs, const u32* fe) {
            const u32 l0 = fe[0] - fs[0], l1 = fe[1] - fs[1];
            if (!l0 || !l1) return;
            A.push_back(key_of_fast(p + fs[0], l0, bufend));
            A.push_back(key_of_fast(p + fs[1], l1, bufend));
        });
    }
    const size_t M = A.size();
    // 2. LMF doubles as the key -> dense id table while the components are
    // found: v holds the id first and the person at the end. Keys are looked
    // up in input order with the table slot prefetched 16 keys ahead.
    size_t cap = 16;
    while (cap < M + M / 2) cap <<= 1;
    LMASK = cap - 1;
    LMF.n = cap;
    LMF.p = huge_zeroed<KV>(cap);
    KV* T = LMF.p;
    vector<u32> ids(M);
    vector<u64> keys;  // id -> key
    keys.reserve(M / 2 + 16);
    {
        const size_t D = 16;
        for (size_t j = 0; j < M && j < D; j++) __builtin_prefetch(T + (mix64(A[j]) & LMASK), 1);
        for (size_t j = 0; j < M; j++) {
            if (j + D < M) __builtin_prefetch(T + (mix64(A[j + D]) & LMASK), 1);
            const u64 k = A[j];
            u64 s = mix64(k) & LMASK;
            for (;;) {
                if (T[s].k == k) {
                    ids[j] = (u32)T[s].v;
                    break;
                }
                if (!T[s].k) {
                    T[s].k = k;
                    T[s].v = keys.size();
                    ids[j] = (u32)keys.size();
                    keys.push_back(k);
                    break;
                }
                s = (s + 1) & LMASK;
            }
        }
    }
    const u32 N = (u32)keys.size();
    // 3. union-find (path halving; the smaller id becomes the root)
    vector<u32> par(N);
    for (u32 i = 0; i < N; i++) par[i] = i;
    auto find = [&](u32 x) {
        while (par[x] != x) {
            par[x] = par[par[x]];
            x = par[x];
        }
        return x;
    };
    for (size_t j = 0; j < M; j += 2) {
        u32 a = find(ids[j]), b = find(ids[j + 1]);
        if (a == b) continue;
        if (a < b)
            par[b] = a;
        else
            par[a] = b;
    }
    // 4. each id's person: the smallest key of its component
    vector<u64> nm(keys);  // root -> smallest key of its component
    for (u32 i = 0; i < N; i++) {
        const u32 r = find(i);
        par[i] = r;
        if (r != i && key_less(keys[i], nm[r])) nm[r] = keys[i];
    }
    vector<u64> pv(N);
    for (u32 i = 0; i < N; i++) pv[i] = nm[par[i]];
    // 5. ids -> persons, in one sequential pass over the table
    for (size_t s = 0; s < cap; s++)
        if (T[s].k) T[s].v = pv[T[s].v];
}

// SS_SELFTEST=1: differential tests of the fast helpers against the plain ones.
static int selftest() {
    u64 seed = 88172645463325252ULL;
    auto rnd = [&]() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };
    const char alpha[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz \t\r\"-:.+Z\x80\xd9\xff/~`";
    long bad = 0;
    char buf[64];
    for (int it = 0; it < 5000000; it++) {
        u32 n = 1 + (u32)(rnd() % 14);
        for (int i = 0; i < 32; i++) buf[i] = alpha[rnd() % (sizeof(alpha) - 1)];
        if (rnd() % 2)
            for (u32 i = 0; i < n; i++) buf[i] = alpha[rnd() % 63];
        if (rnd() % 4 == 0) buf[rnd() % n] = (char)(rnd() % 256);
        if (rnd() % 8 == 0) buf[rnd() % n] = (char)(rnd() % 2 ? 0 : 127);
        u64 a = 0, b = 0;
        bool x = short_code(buf, n, a), y = short_code_simd(buf, n, b);
        if (x != y || (x && a != b)) {
            if (bad++ < 5) fprintf(stderr, "key mismatch n=%u\n", n);
        }
        // order preservation against memcmp on a second random short string
        if (x) {
            char b2[32];
            u32 n2 = 1 + (u32)(rnd() % 9);
            for (u32 i = 0; i < n2; i++) b2[i] = (char)(1 + rnd() % 127);
            if (rnd() % 2) memcpy(b2, buf, min(n, n2));
            u64 c;
            if (short_code(b2, n2, c)) {
                int cmp = memcmp(buf, b2, min(n, n2));
                bool lt = cmp < 0 || (cmp == 0 && n < n2);
                if ((a < c) != lt && !(a == c && n == n2 && cmp == 0)) {
                    if (bad++ < 5) fprintf(stderr, "key order mismatch\n");
                }
            }
        }
    }
    // timestamps around a random window
    for (int w = 0; w < 40; w++) {
        i64 wsd = days_from_civil(2024, 1, 1) + (i64)(rnd() % 2000);
        init_dates(wsd * DAY_MS, (wsd + 14) * DAY_MS);
        for (int it = 0; it < 200000; it++) {
            i64 ms = (wsd - 90 + (i64)(rnd() % 200)) * DAY_MS + (i64)(rnd() % DAY_MS);
            i64 days = floordiv(ms, DAY_MS), sod = (ms - days * DAY_MS) / 1000;
            i64 z = days + 719468, era = (z >= 0 ? z : z - 146096) / 146097, doe = z - era * 146097;
            i64 yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365, y = yoe + era * 400;
            i64 doy = doe - (365 * yoe + yoe / 4 - yoe / 100), mp = (5 * doy + 2) / 153;
            i64 dd = doy - (153 * mp + 2) / 5 + 1, m = mp < 10 ? mp + 3 : mp - 9;
            y += m <= 2;
            int form = (int)(rnd() % 3);
            int len;
            if (form == 0)
                len = snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02dZ", (int)y, (int)m, (int)dd,
                               (int)(sod / 3600), (int)(sod / 60 % 60), (int)(sod % 60));
            else if (form == 1)
                len = snprintf(buf, sizeof buf, "%04d-%02d-%02dT%02d:%02d:%02d%c%02d:%02d", (int)y, (int)m, (int)dd,
                               (int)(sod / 3600), (int)(sod / 60 % 60), (int)(sod % 60), rnd() % 2 ? '+' : '-',
                               (int)(rnd() % 26), (int)(rnd() % 62));
            else
                len = snprintf(buf, sizeof buf, "%lld", (long long)ms);
            int muts = (int)(rnd() % 3);
            for (int k = 0; k < muts; k++) {
                int at = (int)(rnd() % (u64)len);
                switch (rnd() % 4) {
                    case 0: buf[at] = alpha[rnd() % (sizeof(alpha) - 1)]; break;
                    case 1: buf[at] = (char)('0' + rnd() % 10); break;
                    case 2: buf[at] = (char)('0' + rnd() % 4); break;
                    default: buf[at] = (char)(rnd() % 256); break;
                }
            }
            i64 a = 0, b = 0;
            bool x = parse_ts(buf, (size_t)len, a), yv = parse_ts_fast(buf, (size_t)len, b);
            if (x != yv || (x && a != b)) {
                if (bad++ < 10) fprintf(stderr, "ts mismatch: %.*s  %d/%lld vs %d/%lld\n", len, buf, x, (long long)a, yv, (long long)b);
            }
        }
    }
    fprintf(stderr, "selftest: %ld mismatches\n", bad);
    return bad ? 1 : 0;
}

int main(int argc, char** argv) {
#ifdef SS_DISPATCH_TO
    // This build targets x86-64-v3 (AVX2). If the CPU has everything x86-64-v4
    // needs (AVX-512 F/BW/CD/DQ/VL), hand over to the AVX-512 build. If that
    // binary is missing or exec fails, keep running this one.
    __builtin_cpu_init();
    if (!getenv("SS_NO_V4") && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512cd") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("avx512vl"))
        execv(SS_DISPATCH_TO, argv);
#endif
    T0 = chrono::steady_clock::now();
    DBG = getenv("SS_DEBUG") != nullptr;
    init_codes();
    if (getenv("SS_SELFTEST")) return selftest();
    string DATA = env_or("DATA_DIR", "/data");
    OUT_DIR = env_or("OUTPUT_DIR", "/output");
    if (argc >= 3) DATA = argv[argc - 2], OUT_DIR = argv[argc - 1];
    {
        int n = atoi(env_or("NCPU", "0").c_str());
        if (n <= 0) n = (int)thread::hardware_concurrency();
        NT = max(1, min(n, 64));
    }
    mkdir(OUT_DIR.c_str(), 0777);
    mkdir((OUT_DIR + "/user_daily").c_str(), 0777);
    mkdir((OUT_DIR + "/campaign_daily").c_str(), 0777);

    load_inputs(DATA);
    tick("lookups");
    // The fast path stores times as u32 ms since window_start - 7 days and
    // a category in 8 bits.
    bool fast_ok = WE > WS && (WE - (WS - WEEK_MS)) < (i64)0xFFFFFFF0LL && CATNAMES.size() <= 256 &&
                   getenv("SS_GENERIC") == nullptr;
    if (fast_ok)
        fast::run();  // builds links itself, overlapped with its parse
    else {
        build_links();
        tick("links");
        generic::run();
    }
    fflush(stdout);
    _exit(0);  // skip freeing large structures
}
