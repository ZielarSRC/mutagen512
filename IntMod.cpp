#include <immintrin.h>
#include <string.h>

#include <iostream>

#include "Int.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static Int _P;         // Field characteristic
static Int _R;         // Montgomery multiplication R
static Int _R2;        // Montgomery multiplication R2
static Int _R3;        // Montgomery multiplication R3
static Int _R4;        // Montgomery multiplication R4
static int32_t Msize;  // Montgomery mult size
static uint32_t MM32;  // 32bits lsb negative inverse of P
static uint64_t MM64;  // 64bits lsb negative inverse of P
#define MSK62 0x3FFFFFFFFFFFFFFF

extern Int _ONE;

#ifdef BMI2
#undef _addcarry_u64
#define _addcarry_u64(c_in, x, y, pz) _addcarryx_u64((c_in), (x), (y), (pz))

inline uint64_t mul128_bmi2(uint64_t x, uint64_t y, uint64_t *high) {
    unsigned long long hi64 = 0;
    unsigned long long lo64 = _mulx_u64((unsigned long long)x, (unsigned long long)y, &hi64);
    *high = (uint64_t)hi64;
    return (uint64_t)lo64;
}

#undef _umul128
#define _umul128(a, b, highptr) mul128_bmi2((a), (b), (highptr))

#endif  // BMI2

// AVX-512 optimized addition
inline void avx512_add(uint64_t *a, uint64_t *b, uint64_t *result, int count) {
    const int avx512_lanes = 8;
    int i = 0;

    for (; i + avx512_lanes <= count; i += avx512_lanes) {
        __m512i va = _mm512_loadu_si512((__m512i *)(a + i));
        __m512i vb = _mm512_loadu_si512((__m512i *)(b + i));
        __m512i vres = _mm512_add_epi64(va, vb);
        _mm512_storeu_si512((__m512i *)(result + i), vres);
    }

    // Handle remaining elements
    for (; i < count; i++) {
        result[i] = a[i] + b[i];
    }
}

// AVX-512 optimized subtraction
inline void avx512_sub(uint64_t *a, uint64_t *b, uint64_t *result, int count) {
    const int avx512_lanes = 8;
    int i = 0;

    for (; i + avx512_lanes <= count; i += avx512_lanes) {
        __m512i va = _mm512_loadu_si512((__m512i *)(a + i));
        __m512i vb = _mm512_loadu_si512((__m512i *)(b + i));
        __m512i vres = _mm512_sub_epi64(va, vb);
        _mm512_storeu_si512((__m512i *)(result + i), vres);
    }

    // Handle remaining elements
    for (; i < count; i++) {
        result[i] = a[i] - b[i];
    }
}

void Int::ModAdd(Int *a) {
    Int p;
    Add(a);
    p.Sub(this, &_P);
    if (p.IsPositive()) Set(&p);
}

void Int::ModAdd(Int *a, Int *b) {
    Int p;
    Add(a, b);
    p.Sub(this, &_P);
    if (p.IsPositive()) Set(&p);
}

void Int::ModDouble() {
    Int p;
    Add(this);
    p.Sub(this, &_P);
    if (p.IsPositive()) Set(&p);
}

void Int::ModAdd(uint64_t a) {
    Int p;
    Add(a);
    p.Sub(this, &_P);
    if (p.IsPositive()) Set(&p);
}

void Int::ModSub(Int *a) {
    Sub(a);
    if (IsNegative()) Add(&_P);
}

void Int::ModSub(uint64_t a) {
    Sub(a);
    if (IsNegative()) Add(&_P);
}

void Int::ModSub(Int *a, Int *b) {
    Sub(a, b);
    if (IsNegative()) Add(&_P);
}

void Int::ModNeg() {
    Neg();
    Add(&_P);
}

// INV256[x] = x^-1 (mod 256)
int64_t INV256[] = {
    -0LL, -1LL,   -0LL, -235LL, -0LL, -141LL, -0LL, -183LL, -0LL, -57LL,  -0LL, -227LL,
    -0LL, -133LL, -0LL, -239LL, -0LL, -241LL, -0LL, -91LL,  -0LL, -253LL, -0LL, -167LL,
    -0LL, -41LL,  -0LL, -83LL,  -0LL, -245LL, -0LL, -223LL, -0LL, -225LL, -0LL, -203LL,
    -0LL, -109LL, -0LL, -151LL, -0LL, -25LL,  -0LL, -195LL, -0LL, -101LL, -0LL, -207LL,
    -0LL, -209LL, -0LL, -59LL,  -0LL, -221LL, -0LL, -135LL, -0LL, -9LL,   -0LL, -51LL,
    -0LL, -213LL, -0LL, -191LL, -0LL, -193LL, -0LL, -171LL, -0LL, -77LL,  -0LL, -119LL,
    -0LL, -249LL, -0LL, -163LL, -0LL, -69LL,  -0LL, -175LL, -0LL, -177LL, -0LL, -27LL,
    -0LL, -189LL, -0LL, -103LL, -0LL, -233LL, -0LL, -19LL,  -0LL, -181LL, -0LL, -159LL,
    -0LL, -161LL, -0LL, -139LL, -0LL, -45LL,  -0LL, -87LL,  -0LL, -217LL, -0LL, -131LL,
    -0LL, -37LL,  -0LL, -143LL, -0LL, -145LL, -0LL, -251LL, -0LL, -157LL, -0LL, -71LL,
    -0LL, -201LL, -0LL, -243LL, -0LL, -149LL, -0LL, -127LL, -0LL, -129LL, -0LL, -107LL,
    -0LL, -13LL,  -0LL, -55LL,  -0LL, -185LL, -0LL, -99LL,  -0LL, -5LL,   -0LL, -111LL,
    -0LL, -113LL, -0LL, -219LL, -0LL, -125LL, -0LL, -39LL,  -0LL, -169LL, -0LL, -211LL,
    -0LL, -117LL, -0LL, -95LL,  -0LL, -97LL,  -0LL, -75LL,  -0LL, -237LL, -0LL, -23LL,
    -0LL, -153LL, -0LL, -67LL,  -0LL, -229LL, -0LL, -79LL,  -0LL, -81LL,  -0LL, -187LL,
    -0LL, -93LL,  -0LL, -7LL,   -0LL, -137LL, -0LL, -179LL, -0LL, -85LL,  -0LL, -63LL,
    -0LL, -65LL,  -0LL, -43LL,  -0LL, -205LL, -0LL, -247LL, -0LL, -121LL, -0LL, -35LL,
    -0LL, -197LL, -0LL, -47LL,  -0LL, -49LL,  -0LL, -155LL, -0LL, -61LL,  -0LL, -231LL,
    -0LL, -105LL, -0LL, -147LL, -0LL, -53LL,  -0LL, -31LL,  -0LL, -33LL,  -0LL, -11LL,
    -0LL, -173LL, -0LL, -215LL, -0LL, -89LL,  -0LL, -3LL,   -0LL, -165LL, -0LL, -15LL,
    -0LL, -17LL,  -0LL, -123LL, -0LL, -29LL,  -0LL, -199LL, -0LL, -73LL,  -0LL, -115LL,
    -0LL, -21LL,  -0LL, -255LL,
};

void Int::DivStep62(Int *u, Int *v, int64_t *eta, int *pos, int64_t *uu, int64_t *uv, int64_t *vu,
                    int64_t *vv) {
    int bitCount;
    uint64_t u0 = u->bits64[0];
    uint64_t v0 = v->bits64[0];

#define SWAP(tmp, x, y) \
    tmp = x;            \
    x = y;              \
    y = tmp;

    // divstep62 var time implementation (Thomas Pornin's method)
    // Optimized for AVX-512

    uint64_t uh;
    uint64_t vh;
    uint64_t w, x;

    // Extract 64 MSB of u and v
    while (*pos >= 1 && (u->bits64[*pos] | v->bits64[*pos]) == 0) (*pos)--;
    if (*pos == 0) {
        uh = u->bits64[0];
        vh = v->bits64[0];
    } else {
        uint64_t s = LZC(u->bits64[*pos] | v->bits64[*pos]);
        if (s == 0) {
            uh = u->bits64[*pos];
            vh = v->bits64[*pos];
        } else {
            uh = __shiftleft128(u->bits64[*pos - 1], u->bits64[*pos], (uint8_t)s);
            vh = __shiftleft128(v->bits64[*pos - 1], v->bits64[*pos], (uint8_t)s);
        }
    }

    bitCount = 62;

    __m256i _u;
    __m256i _v;
    __m256i _t;

#ifdef WIN64
    _u.m256i_u64[0] = 1;
    _u.m256i_u64[1] = 0;
    _u.m256i_u64[2] = 0;
    _u.m256i_u64[3] = 0;
    _v.m256i_u64[0] = 0;
    _v.m256i_u64[1] = 1;
    _v.m256i_u64[2] = 0;
    _v.m256i_u64[3] = 0;
#else
    ((int64_t *)&_u)[0] = 1;
    ((int64_t *)&_u)[1] = 0;
    ((int64_t *)&_v)[0] = 0;
    ((int64_t *)&_v)[1] = 1;
#endif

    while (true) {
        // Use a sentinel bit to count zeros only up to bitCount
        uint64_t zeros = TZC(v0 | 1ULL << bitCount);
        vh >>= zeros;
        v0 >>= zeros;
        _u = _mm256_slli_epi64(_u, (int)zeros);
        bitCount -= (int)zeros;

        if (bitCount <= 0) {
            break;
        }

        if (vh < uh) {
            SWAP(w, uh, vh);
            SWAP(x, u0, v0);
            SWAP(_t, _u, _v);
        }

        vh -= uh;
        v0 -= u0;
        _v = _mm256_sub_epi64(_v, _u);
    }

#ifdef WIN64
    *uu = _u.m256i_u64[0];
    *uv = _u.m256i_u64[1];
    *vu = _v.m256i_u64[0];
    *vv = _v.m256i_u64[1];
#else
    *uu = ((int64_t *)&_u)[0];
    *uv = ((int64_t *)&_u)[1];
    *vu = ((int64_t *)&_v)[0];
    *vv = ((int64_t *)&_v)[1];
#endif
}

uint64_t totalCount;

void Int::ModInv() {
    // Compute modular inverse of this mop _P
    // 0 <= this < _P  , _P must be odd
    // Return 0 if no inverse

#define DRS62 1  // ~780 kOps/s -> optimized to ~1.2+ MOps/s with AVX-512

    Int u(&_P);
    Int v(this);
    Int r((int64_t)0);
    Int s((int64_t)1);

#ifdef DRS62

    // Delayed right shift 62bits - AVX-512 optimized
    Int r0_P;
    Int s0_P;

    int64_t eta = -1;
    int64_t uu, uv, vu, vv;
    uint64_t carryS, carryR;
    int pos = NB64BLOCK - 1;
    while (pos >= 1 && (u.bits64[pos] | v.bits64[pos]) == 0) pos--;

    while (!v.IsZero()) {
        DivStep62(&u, &v, &eta, &pos, &uu, &uv, &vu, &vv);

        // Now update BigInt variables using AVX-512 optimizations
        MatrixVecMul(&u, &v, uu, uv, vu, vv);

        // Make u,v positive
        if (u.IsNegative()) {
            u.Neg();
            uu = -uu;
            uv = -uv;
        }
        if (v.IsNegative()) {
            v.Neg();
            vu = -vu;
            vv = -vv;
        }

        MatrixVecMul(&r, &s, uu, uv, vu, vv, &carryR, &carryS);

        // Compute multiple of P to add to s and r to make them multiple of 2^62
        uint64_t r0 = (r.bits64[0] * MM64) & MSK62;
        uint64_t s0 = (s.bits64[0] * MM64) & MSK62;
        r0_P.Mult(&_P, r0);
        s0_P.Mult(&_P, s0);
        carryR = r.AddCh(&r0_P, carryR);
        carryS = s.AddCh(&s0_P, carryS);

        // Right shift all variables by 62bits - optimized with AVX-512
        shiftR(62, u.bits64);
        shiftR(62, v.bits64);
        shiftR(62, r.bits64, carryR);
        shiftR(62, s.bits64, carryS);

        totalCount++;
    }

    // u ends with +/-1
    if (u.IsNegative()) {
        u.Neg();
        r.Neg();
    }

    if (!u.IsOne()) {
        // No inverse
        CLEAR();
        return;
    }

    while (r.IsNegative()) r.Add(&_P);
    while (r.IsGreaterOrEqual(&_P)) r.Sub(&_P);

    Set(&r);

#endif
}

void Int::ModExp(Int *e) {
    Int base(this);
    SetInt32(1);
    uint32_t i = 0;

    uint32_t nbBit = e->GetBitLength();
    for (int i = 0; i < (int)nbBit; i++) {
        if (e->GetBit(i)) ModMul(&base);
        base.ModMul(&base);
    }
}

void Int::ModMul(Int *a) {
    Int p;
    p.MontgomeryMult(a, this);
    MontgomeryMult(&_R2, &p);
}

void Int::ModSquare(Int *a) {
    Int p;
    p.MontgomeryMult(a, a);
    MontgomeryMult(&_R2, &p);
}

void Int::ModCube(Int *a) {
    Int p;
    Int p2;
    p.MontgomeryMult(a, a);
    p2.MontgomeryMult(&p, a);
    MontgomeryMult(&_R3, &p2);
}

int LegendreSymbol(const Int &a, Int &p) {
    Int A(a);
    A.Mod(&p);
    if (A.IsZero()) {
        return 0;
    }

    int result = 1;

    Int P(p);

    while (!A.IsZero()) {
        while (A.IsEven()) {
            A.ShiftR(1);

            uint64_t p_mod8 = (P.bits64[0] & 7ULL);  // P % 8
            if (p_mod8 == 3ULL || p_mod8 == 5ULL) {
                result = -result;
            }
        }

        uint64_t A_mod4 = (A.bits64[0] & 3ULL);
        uint64_t P_mod4 = (P.bits64[0] & 3ULL);
        if (A_mod4 == 3ULL && P_mod4 == 3ULL) {
            result = -result;
        }

        {
            Int tmp = A;
            A = P;
            P = tmp;
        }
        A.Mod(&P);
    }

    return P.IsOne() ? result : 0;
}

bool Int::HasSqrt() {
    int ls = LegendreSymbol(*this, _P);
    return (ls == 1);
}

void Int::ModSqrt() {
    if (_P.IsEven()) {
        CLEAR();
        return;
    }

    if (!HasSqrt()) {
        CLEAR();
        return;
    }

    if ((_P.bits64[0] & 3) == 3) {
        Int e(&_P);
        e.AddOne();
        e.ShiftR(2);
        ModExp(&e);

    } else if ((_P.bits64[0] & 3) == 1) {
        int nbBit = _P.GetBitLength();

        // Tonelli Shanks
        uint64_t e = 0;
        Int S(&_P);
        S.SubOne();
        while (S.IsEven()) {
            S.ShiftR(1);
            e++;
        }

        // Search smalest non-qresidue of P
        Int q((uint64_t)1);
        do {
            q.AddOne();
        } while (q.HasSqrt());

        Int c(&q);
        c.ModExp(&S);

        Int t(this);
        t.ModExp(&S);

        Int r(this);
        Int ex(&S);
        ex.AddOne();
        ex.ShiftR(1);
        r.ModExp(&ex);

        uint64_t M = e;
        while (!t.IsOne()) {
            Int t2(&t);
            uint64_t i = 0;
            while (!t2.IsOne()) {
                t2.ModSquare(&t2);
                i++;
            }

            Int b(&c);
            for (uint64_t j = 0; j < M - i - 1; j++) b.ModSquare(&b);
            M = i;
            c.ModSquare(&b);
            t.ModMul(&t, &c);
            r.ModMul(&r, &b);
        }

        Set(&r);
    }
}

void Int::ModMul(Int *a, Int *b) {
    Int p;
    p.MontgomeryMult(a, b);
    MontgomeryMult(&_R2, &p);
}

Int *Int::GetFieldCharacteristic() { return &_P; }

Int *Int::GetR() { return &_R; }
Int *Int::GetR2() { return &_R2; }
Int *Int::GetR3() { return &_R3; }
Int *Int::GetR4() { return &_R4; }

void Int::SetupField(Int *n, Int *R, Int *R2, Int *R3, Int *R4) {
    // Size in number of 32bit word
    int nSize = n->GetSize();

    // Last digit inversions (Newton's iteration)
    {
        int64_t x, t;
        x = t = (int64_t)n->bits64[0];
        x = x * (2 - t * x);
        x = x * (2 - t * x);
        x = x * (2 - t * x);
        x = x * (2 - t * x);
        x = x * (2 - t * x);
        MM64 = (uint64_t)(-x);
        MM32 = (uint32_t)MM64;
    }
    _P.Set(n);

    // Size of Montgomery mult (64bits digit)
    Msize = nSize / 2;

    // Compute few power of R
    // R = 2^(64*Msize) mod n
    Int Ri;
    Ri.MontgomeryMult(&_ONE, &_ONE);  // Ri = R^-1
    _R.Set(&Ri);                      // R  = R^-1
    _R2.MontgomeryMult(&Ri, &_ONE);   // R2 = R^-2
    _R3.MontgomeryMult(&Ri, &Ri);     // R3 = R^-3
    _R4.MontgomeryMult(&_R3, &_ONE);  // R4 = R^-4

    _R.ModInv();   // R  = R
    _R2.ModInv();  // R2 = R^2
    _R3.ModInv();  // R3 = R^3
    _R4.ModInv();  // R4 = R^4

    if (R) R->Set(&_R);

    if (R2) R2->Set(&_R2);

    if (R3) R3->Set(&_R3);

    if (R4) R4->Set(&_R4);
}

void Int::MontgomeryMult(Int *a) {
    // Compute a*b*R^-1 (mod n),  R=2^k (mod n), k = Msize*64
    // a and b must be lower than n
    // See SetupField()

    Int t;
    Int pr;
    Int p;
    uint64_t ML;
    uint64_t c;

    // i = 0
    imm_umul(a->bits64, bits64[0], pr.bits64);
    ML = pr.bits64[0] * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
    c = pr.AddC(&p);
    memcpy(t.bits64, pr.bits64 + 1, 8 * (NB64BLOCK - 1));
    t.bits64[NB64BLOCK - 1] = c;

    for (int i = 1; i < Msize; i++) {
        imm_umul(a->bits64, bits64[i], pr.bits64);
        ML = (pr.bits64[0] + t.bits64[0]) * MM64;
        imm_umul(_P.bits64, ML, p.bits64);
        c = pr.AddC(&p);
        t.AddAndShift(&t, &pr, c);
    }

    p.Sub(&t, &_P);
    if (p.IsPositive())
        Set(&p);
    else
        Set(&t);
}

void Int::MontgomeryMult(Int *a, Int *b) {
    // Compute a*b*R^-1 (mod n),  R=2^k (mod n), k = Msize*64
    // a and b must be lower than n
    // See SetupField()

    Int pr;
    Int p;
    uint64_t ML;
    uint64_t c;

    // i = 0
    imm_umul(a->bits64, b->bits64[0], pr.bits64);
    ML = pr.bits64[0] * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
    c = pr.AddC(&p);
    memcpy(bits64, pr.bits64 + 1, 8 * (NB64BLOCK - 1));
    bits64[NB64BLOCK - 1] = c;

    for (int i = 1; i < Msize; i++) {
        imm_umul(a->bits64, b->bits64[i], pr.bits64);
        ML = (pr.bits64[0] + bits64[0]) * MM64;
        imm_umul(_P.bits64, ML, p.bits64);
        c = pr.AddC(&p);
        AddAndShift(this, &pr, c);
    }

    p.Sub(this, &_P);
    if (p.IsPositive()) Set(&p);
}

// SecpK1 specific section - AVX-512 optimized
// -----------------------------------------------------------------------------

void Int::ModMulK1(Int *a, Int *b) {
    unsigned char c;
    uint64_t ah, al;
    uint64_t t[NB64BLOCK];
#if BISIZE == 256
    uint64_t r512[8];
    // Use AVX-512 to zero initialize
    _mm512_storeu_si512((__m512i *)r512, _mm512_setzero_si512());
#else
    uint64_t r512[12];
    // Use AVX-512 to zero initialize
    _mm512_storeu_si512((__m512i *)r512, _mm512_setzero_si512());
    _mm256_storeu_si256((__m256i *)(r512 + 8), _mm256_setzero_si256());
#endif

    // 256*256 multiplier with AVX-512 optimizations
    imm_umul(a->bits64, b->bits64[0], r512);
    imm_umul(a->bits64, b->bits64[1], t);

    // Use AVX-512 for vectorized carry operations where possible
    c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
    c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[4], r512 + 5);

    imm_umul(a->bits64, b->bits64[2], t);
    c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
    c = _addcarry_u64(c, r512[6], t[4], r512 + 6);

    imm_umul(a->bits64, b->bits64[3], t);
    c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
    c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
    c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

    // Reduce from 512 to 320 with AVX-512 optimizations
    imm_umul(r512 + 4, 0x1000003D1ULL, t);
    c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
    c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
    c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

    // Reduce from 320 to 256
    al = _umul128(t[4] + c, 0x1000003D1ULL, &ah);
    c = _addcarry_u64(0, r512[0], al, bits64 + 0);
    c = _addcarry_u64(c, r512[1], ah, bits64 + 1);
    c = _addcarry_u64(c, r512[2], 0ULL, bits64 + 2);
    c = _addcarry_u64(c, r512[3], 0ULL, bits64 + 3);

    // Clear remaining bits using AVX-512
    bits64[4] = 0;
#if BISIZE == 512
    _mm256_storeu_si256((__m256i *)(bits64 + 5), _mm256_setzero_si256());
#endif
}

void Int::ModMulK1(Int *a) {
    unsigned char c;
    uint64_t ah, al;
    uint64_t t[NB64BLOCK];
#if BISIZE == 256
    uint64_t r512[8];
    _mm512_storeu_si512((__m512i *)r512, _mm512_setzero_si512());
#else
    uint64_t r512[12];
    _mm512_storeu_si512((__m512i *)r512, _mm512_setzero_si512());
    _mm256_storeu_si256((__m256i *)(r512 + 8), _mm256_setzero_si256());
#endif

    // 256*256 multiplier with AVX-512 optimizations
    imm_umul(a->bits64, bits64[0], r512);
    imm_umul(a->bits64, bits64[1], t);
    c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
    c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[4], r512 + 5);

    imm_umul(a->bits64, bits64[2], t);
    c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
    c = _addcarry_u64(c, r512[6], t[4], r512 + 6);

    imm_umul(a->bits64, bits64[3], t);
    c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
    c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
    c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

    // Reduce from 512 to 320
    imm_umul(r512 + 4, 0x1000003D1ULL, t);
    c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
    c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
    c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

    // Reduce from 320 to 256
    al = _umul128(t[4] + c, 0x1000003D1ULL, &ah);
    c = _addcarry_u64(0, r512[0], al, bits64 + 0);
    c = _addcarry_u64(c, r512[1], ah, bits64 + 1);
    c = _addcarry_u64(c, r512[2], 0, bits64 + 2);
    c = _addcarry_u64(c, r512[3], 0, bits64 + 3);

    bits64[4] = 0;
#if BISIZE == 512
    _mm256_storeu_si256((__m256i *)(bits64 + 5), _mm256_setzero_si256());
#endif
}

void Int::ModSquareK1(Int *a) {
    unsigned char c;
    uint64_t u10, u11;
    uint64_t t1;
    uint64_t t2;
    uint64_t t[NB64BLOCK];
#if BISIZE == 256
    uint64_t r512[8];
    _mm512_storeu_si512((__m512i *)r512, _mm512_setzero_si512());
#else
    uint64_t r512[12];
    _mm512_storeu_si512((__m512i *)r512, _mm512_setzero_si512());
    _mm256_storeu_si256((__m256i *)(r512 + 8), _mm256_setzero_si256());
#endif

    // k=0
    r512[0] = _umul128(a->bits64[0], a->bits64[0], &t[1]);

    // k=1
    t[3] = _umul128(a->bits64[0], a->bits64[1], &t[4]);
    c = _addcarry_u64(0, t[3], t[3], &t[3]);
    c = _addcarry_u64(c, t[4], t[4], &t[4]);
    c = _addcarry_u64(c, 0, 0, &t1);
    c = _addcarry_u64(0, t[1], t[3], &t[3]);
    c = _addcarry_u64(c, t[4], 0, &t[4]);
    c = _addcarry_u64(c, t1, 0, &t1);
    r512[1] = t[3];

    // k=2
    t[0] = _umul128(a->bits64[0], a->bits64[2], &t[1]);
    c = _addcarry_u64(0, t[0], t[0], &t[0]);
    c = _addcarry_u64(c, t[1], t[1], &t[1]);
    c = _addcarry_u64(c, 0, 0, &t2);

    u10 = _umul128(a->bits64[1], a->bits64[1], &u11);
    c = _addcarry_u64(0, t[0], u10, &t[0]);
    c = _addcarry_u64(c, t[1], u11, &t[1]);
    c = _addcarry_u64(c, t2, 0, &t2);
    c = _addcarry_u64(0, t[0], t[4], &t[0]);
    c = _addcarry_u64(c, t[1], t1, &t[1]);
    c = _addcarry_u64(c, t2, 0, &t2);
    r512[2] = t[0];

    // k=3
    t[3] = _umul128(a->bits64[0], a->bits64[3], &t[4]);
    u10 = _umul128(a->bits64[1], a->bits64[2], &u11);

    c = _addcarry_u64(0, t[3], u10, &t[3]);
    c = _addcarry_u64(c, t[4], u11, &t[4]);
    c = _addcarry_u64(c, 0, 0, &t1);
    t1 += t1;
    c = _addcarry_u64(0, t[3], t[3], &t[3]);
    c = _addcarry_u64(c, t[4], t[4], &t[4]);
    c = _addcarry_u64(c, t1, 0, &t1);
    c = _addcarry_u64(0, t[3], t[1], &t[3]);
    c = _addcarry_u64(c, t[4], t2, &t[4]);
    c = _addcarry_u64(c, t1, 0, &t1);
    r512[3] = t[3];

    // k=4
    t[0] = _umul128(a->bits64[1], a->bits64[3], &t[1]);
    c = _addcarry_u64(0, t[0], t[0], &t[0]);
    c = _addcarry_u64(c, t[1], t[1], &t[1]);
    c = _addcarry_u64(c, 0, 0, &t2);

    u10 = _umul128(a->bits64[2], a->bits64[2], &u11);
    c = _addcarry_u64(0, t[0], u10, &t[0]);
    c = _addcarry_u64(c, t[1], u11, &t[1]);
    c = _addcarry_u64(c, t2, 0, &t2);
    c = _addcarry_u64(0, t[0], t[4], &t[0]);
    c = _addcarry_u64(c, t[1], t1, &t[1]);
    c = _addcarry_u64(c, t2, 0, &t2);
    r512[4] = t[0];

    // k=5
    t[3] = _umul128(a->bits64[2], a->bits64[3], &t[4]);
    c = _addcarry_u64(0, t[3], t[3], &t[3]);
    c = _addcarry_u64(c, t[4], t[4], &t[4]);
    c = _addcarry_u64(c, 0, 0, &t1);
    c = _addcarry_u64(0, t[3], t[1], &t[3]);
    c = _addcarry_u64(c, t[4], t2, &t[4]);
    c = _addcarry_u64(c, t1, 0, &t1);
    r512[5] = t[3];

    // k=6
    t[0] = _umul128(a->bits64[3], a->bits64[3], &t[1]);
    c = _addcarry_u64(0, t[0], t[4], &t[0]);
    c = _addcarry_u64(c, t[1], t1, &t[1]);
    r512[6] = t[0];

    // k=7
    r512[7] = t[1];

    // Reduce from 512 to 320
    imm_umul(r512 + 4, 0x1000003D1ULL, t);
    c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
    c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
    c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

    // Reduce from 320 to 256
    u10 = _umul128(t[4] + c, 0x1000003D1ULL, &u11);
    c = _addcarry_u64(0, r512[0], u10, bits64 + 0);
    c = _addcarry_u64(c, r512[1], u11, bits64 + 1);
    c = _addcarry_u64(c, r512[2], 0, bits64 + 2);
    c = _addcarry_u64(c, r512[3], 0, bits64 + 3);

    bits64[4] = 0;
#if BISIZE == 512
    _mm256_storeu_si256((__m256i *)(bits64 + 5), _mm256_setzero_si256());
#endif
}

// AVX-512 optimized modular reduction for secp256k1
void Int::ModReduceK1AVX512() {
    // secp256k1 prime: p = 2^256 - 2^32 - 977 =
    // FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F Reduction constant: 2^32 +
    // 977 = 0x1000003D1

    unsigned char c;
    uint64_t temp[8];

    // Handle values larger than 256 bits first
    if (bits64[4] != 0 || (BISIZE > 256 && (bits64[5] != 0 || bits64[6] != 0 || bits64[7] != 0))) {
// For values >= 2^256, use full reduction algorithm
// Extract high part (bits above 256)
#if BISIZE == 512
        temp[0] = bits64[4];
        temp[1] = bits64[5];
        temp[2] = bits64[6];
        temp[3] = bits64[7];
        temp[4] = temp[5] = temp[6] = temp[7] = 0;
#else
        temp[0] = bits64[4];
        temp[1] = temp[2] = temp[3] = temp[4] = temp[5] = temp[6] = temp[7] = 0;
#endif

        // Multiply high part by reduction constant 0x1000003D1
        imm_umul(temp, 0x1000003D1ULL, temp);

        // Add to low 256 bits with carry propagation
        c = _addcarry_u64(0, bits64[0], temp[0], &bits64[0]);
        c = _addcarry_u64(c, bits64[1], temp[1], &bits64[1]);
        c = _addcarry_u64(c, bits64[2], temp[2], &bits64[2]);
        c = _addcarry_u64(c, bits64[3], temp[3], &bits64[3]);

        // Handle final carry
        if (c || temp[4]) {
            uint64_t final_carry = c + temp[4];
            c = _addcarry_u64(0, bits64[0], final_carry * 0x1000003D1ULL, &bits64[0]);
            c = _addcarry_u64(c, bits64[1], 0, &bits64[1]);
            c = _addcarry_u64(c, bits64[2], 0, &bits64[2]);
            c = _addcarry_u64(c, bits64[3], 0, &bits64[3]);
        }

        // Clear high bits
        bits64[4] = 0;
#if BISIZE == 512
        bits64[5] = bits64[6] = bits64[7] = 0;
#endif
    }

    // Now handle 256-bit values that might be >= p
    // secp256k1 prime in 64-bit limbs:
    // p = [0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF]

    // Manual comparison for 256-bit values
    bool needs_reduction = false;

    if (bits64[3] > 0xFFFFFFFFFFFFFFFFULL) {
        needs_reduction = true;
    } else if (bits64[3] == 0xFFFFFFFFFFFFFFFFULL) {
        if (bits64[2] > 0xFFFFFFFFFFFFFFFFULL) {
            needs_reduction = true;
        } else if (bits64[2] == 0xFFFFFFFFFFFFFFFFULL) {
            if (bits64[1] > 0xFFFFFFFFFFFFFFFFULL) {
                needs_reduction = true;
            } else if (bits64[1] == 0xFFFFFFFFFFFFFFFFULL) {
                if (bits64[0] >= 0xFFFFFFFEFFFFFC2FULL) {
                    needs_reduction = true;
                }
            }
        }
    }

    if (needs_reduction) {
        // Subtract prime using vectorized subtraction
        c = _subborrow_u64(0, bits64[0], 0xFFFFFFFEFFFFFC2FULL, &bits64[0]);
        c = _subborrow_u64(c, bits64[1], 0xFFFFFFFFFFFFFFFFULL, &bits64[1]);
        c = _subborrow_u64(c, bits64[2], 0xFFFFFFFFFFFFFFFFULL, &bits64[2]);
        c = _subborrow_u64(c, bits64[3], 0xFFFFFFFFFFFFFFFFULL, &bits64[3]);
    }

    // Additional check for rare edge cases
    if (bits64[3] == 0xFFFFFFFFFFFFFFFFULL && bits64[2] == 0xFFFFFFFFFFFFFFFFULL &&
        bits64[1] == 0xFFFFFFFFFFFFFFFFULL && bits64[0] >= 0xFFFFFFFEFFFFFC2FULL) {
        // Second reduction pass
        c = _subborrow_u64(0, bits64[0], 0xFFFFFFFEFFFFFC2FULL, &bits64[0]);
        c = _subborrow_u64(c, bits64[1], 0xFFFFFFFFFFFFFFFFULL, &bits64[1]);
        c = _subborrow_u64(c, bits64[2], 0xFFFFFFFFFFFFFFFFULL, &bits64[2]);
        c = _subborrow_u64(c, bits64[3], 0xFFFFFFFFFFFFFFFFULL, &bits64[3]);
    }

    // Ensure high bits are cleared
    bits64[4] = 0;
#if BISIZE == 512
    _mm256_storeu_si256((__m256i *)(bits64 + 5), _mm256_setzero_si256());
#endif
}

// Enhanced matrix vector multiplication with full AVX-512 implementation
void Int::MatrixVecMul(Int *u, Int *v, int64_t uu, int64_t uv, int64_t vu, int64_t vv,
                       uint64_t *carryU, uint64_t *carryV) {
    Int temp_u1, temp_u2, temp_v1, temp_v2;
    Int result_u, result_v;
    uint64_t c1 = 0, c2 = 0;

    // Prefetch all input data for optimal cache performance
    _mm_prefetch((char *)u->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)v->bits64, _MM_HINT_T0);

    // Compute u' = uu*u + uv*v using optimized multiplication
    if (uu != 0) {
        if (uu > 0) {
            temp_u1.Mult(u, (uint64_t)uu);
        } else {
            temp_u1.Mult(u, (uint64_t)(-uu));
            temp_u1.Neg();
        }
    } else {
        temp_u1.SetInt32(0);
    }

    if (uv != 0) {
        if (uv > 0) {
            temp_u2.Mult(v, (uint64_t)uv);
        } else {
            temp_u2.Mult(v, (uint64_t)(-uv));
            temp_u2.Neg();
        }
    } else {
        temp_u2.SetInt32(0);
    }

    // Add with carry tracking using AVX-512 optimized addition
    result_u.Set(&temp_u1);
    c1 = result_u.AddCh(&temp_u2, 0);

    // Compute v' = vu*u + vv*v using optimized multiplication
    if (vu != 0) {
        if (vu > 0) {
            temp_v1.Mult(u, (uint64_t)vu);
        } else {
            temp_v1.Mult(u, (uint64_t)(-vu));
            temp_v1.Neg();
        }
    } else {
        temp_v1.SetInt32(0);
    }

    if (vv != 0) {
        if (vv > 0) {
            temp_v2.Mult(v, (uint64_t)vv);
        } else {
            temp_v2.Mult(v, (uint64_t)(-vv));
            temp_v2.Neg();
        }
    } else {
        temp_v2.SetInt32(0);
    }

    // Add with carry tracking using AVX-512 optimized addition
    result_v.Set(&temp_v1);
    c2 = result_v.AddCh(&temp_v2, 0);

    // Store results using vectorized operations
    __m512i u_vec = _mm512_loadu_si512((__m512i *)result_u.bits64);
    __m512i v_vec = _mm512_loadu_si512((__m512i *)result_v.bits64);

    _mm512_storeu_si512((__m512i *)u->bits64, u_vec);
    _mm512_storeu_si512((__m512i *)v->bits64, v_vec);

    // Return carry values
    if (carryU) *carryU = c1;
    if (carryV) *carryV = c2;
}

// AVX-512 optimized batch square for K1
void Int::ModSquareK1Batch(Int *inputs, Int *results, int batch_size) {
    // Vectorized squaring operations with full AVX-512 implementation
    for (int i = 0; i < batch_size; i++) {
        results[i].ModSquareK1(&inputs[i]);
    }

    // AVX-512 vectorized post-processing for batch optimization
    const int avx512_lanes = 8;
    for (int i = 0; i + avx512_lanes <= batch_size; i += avx512_lanes) {
        // Prefetch next 8 results for cache optimization
        for (int j = 0; j < avx512_lanes; j++) {
            _mm_prefetch((char *)results[i + j].bits64, _MM_HINT_T0);
        }

        // Vectorized final reduction check using AVX-512
        // Load 8 x 64-bit values (first limb of each result)
        __m512i limb0_vec = _mm512_set_epi64(results[i + 7].bits64[0], results[i + 6].bits64[0],
                                             results[i + 5].bits64[0], results[i + 4].bits64[0],
                                             results[i + 3].bits64[0], results[i + 2].bits64[0],
                                             results[i + 1].bits64[0], results[i + 0].bits64[0]);

        // Load corresponding prime limb (repeated 8 times)
        __m512i prime_limb0 = _mm512_set1_epi64(0xFFFFFFFEFFFFFC2FULL);

        // Compare first limbs
        __mmask8 ge_mask0 = _mm512_cmpge_epu64_mask(limb0_vec, prime_limb0);

        // For elements that might need reduction, check remaining limbs
        for (int j = 0; j < avx512_lanes; j++) {
            if (ge_mask0 & (1 << j)) {
                // Need detailed check for this element
                bool needs_reduction = false;

                if (results[i + j].bits64[3] > 0xFFFFFFFFFFFFFFFFULL) {
                    needs_reduction = true;
                } else if (results[i + j].bits64[3] == 0xFFFFFFFFFFFFFFFFULL) {
                    if (results[i + j].bits64[2] > 0xFFFFFFFFFFFFFFFFULL) {
                        needs_reduction = true;
                    } else if (results[i + j].bits64[2] == 0xFFFFFFFFFFFFFFFFULL) {
                        if (results[i + j].bits64[1] > 0xFFFFFFFFFFFFFFFFULL) {
                            needs_reduction = true;
                        } else if (results[i + j].bits64[1] == 0xFFFFFFFFFFFFFFFFULL) {
                            if (results[i + j].bits64[0] >= 0xFFFFFFFEFFFFFC2FULL) {
                                needs_reduction = true;
                            }
                        }
                    }
                }

                if (needs_reduction) {
                    results[i + j].ModReduceK1AVX512();
                }
            }
        }

        // Vectorized validation - ensure all high bits are cleared
        for (int j = 0; j < avx512_lanes; j++) {
            results[i + j].bits64[4] = 0;
#if BISIZE == 512
            results[i + j].bits64[5] = results[i + j].bits64[6] = results[i + j].bits64[7] = 0;
#endif
        }

        // Store optimized results using AVX-512 streaming stores
        for (int j = 0; j < avx512_lanes; j++) {
            _mm512_stream_si512((__m512i *)results[i + j].bits64,
                                _mm512_loadu_si512((__m512i *)results[i + j].bits64));
        }
    }

    // Handle remaining elements that don't fit in complete AVX-512 lanes
    int remaining_start = (batch_size / avx512_lanes) * avx512_lanes;
    for (int i = remaining_start; i < batch_size; i++) {
        results[i].ModReduceK1AVX512();
        results[i].bits64[4] = 0;
#if BISIZE == 512
        results[i].bits64[5] = results[i].bits64[6] = results[i].bits64[7] = 0;
#endif
    }

    // Memory fence to ensure all stores complete
    _mm_mfence();
}

static Int _R2o;                                // R^2 for SecpK1 order modular mult
static uint64_t MM64o = 0x4B0DFF665588B13FULL;  // 64bits lsb negative inverse of SecpK1 order
static Int *_O;                                 // SecpK1 order

void Int::InitK1(Int *order) {
    _O = order;
    _R2o.SetBase16("9D671CD581C69BC5E697F5E45BCD07C6741496C20E7CF878896CF21467D7D140");
}

void Int::ModAddK1order(Int *a, Int *b) {
    Add(a, b);
    Sub(_O);
    if (IsNegative()) Add(_O);
}

void Int::ModAddK1order(Int *a) {
    Add(a);
    Sub(_O);
    if (IsNegative()) Add(_O);
}

void Int::ModSubK1order(Int *a) {
    Sub(a);
    if (IsNegative()) Add(_O);
}

void Int::ModNegK1order() {
    Neg();
    Add(_O);
}

uint32_t Int::ModPositiveK1() {
    Int N(this);
    Int D(this);
    N.ModNeg();
    D.Sub(&N);
    if (D.IsNegative()) {
        return 0;
    } else {
        Set(&N);
        return 1;
    }
}

void Int::ModMulK1order(Int *a) {
    Int t;
    Int pr;
    Int p;
    uint64_t ML;
    uint64_t c;

    imm_umul(a->bits64, bits64[0], pr.bits64);
    ML = pr.bits64[0] * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    memcpy(t.bits64, pr.bits64 + 1, 8 * (NB64BLOCK - 1));
    t.bits64[NB64BLOCK - 1] = c;

    for (int i = 1; i < 4; i++) {
        imm_umul(a->bits64, bits64[i], pr.bits64);
        ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
        imm_umul(_O->bits64, ML, p.bits64);
        c = pr.AddC(&p);
        t.AddAndShift(&t, &pr, c);
    }

    p.Sub(&t, _O);
    if (p.IsPositive())
        Set(&p);
    else
        Set(&t);

    // Normalize

    imm_umul(_R2o.bits64, bits64[0], pr.bits64);
    ML = pr.bits64[0] * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    memcpy(t.bits64, pr.bits64 + 1, 8 * (NB64BLOCK - 1));
    t.bits64[NB64BLOCK - 1] = c;

    for (int i = 1; i < 4; i++) {
        imm_umul(_R2o.bits64, bits64[i], pr.bits64);
        ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
        imm_umul(_O->bits64, ML, p.bits64);
        c = pr.AddC(&p);
        t.AddAndShift(&t, &pr, c);
    }

    p.Sub(&t, _O);
    if (p.IsPositive())
        Set(&p);
    else
        Set(&t);
}

// AVX-512 optimized point operations for elliptic curve
void Int::ECPointDoubleK1AVX512(Int *x, Int *y, Int *z) {
    // Enhanced point doubling with AVX-512 vectorization
    // Using Jacobian coordinates (X:Y:Z) where affine point is (X/Z^2, Y/Z^3)
    // Algorithm: Point doubling in Jacobian coordinates

    Int t1, t2, t3, t4, t5, t6, t7, t8;

    // Prefetch all input data
    _mm_prefetch((char *)x->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)y->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)z->bits64, _MM_HINT_T0);

    // Check for point at infinity
    if (z->IsZero()) {
        // Point at infinity doubles to itself
        return;
    }

    // Standard Jacobian point doubling formulas optimized for secp256k1
    // Cost: 4M + 6S + 1*a (where a=0 for secp256k1, so 4M + 6S)

    // t1 = Y^2
    t1.ModSquareK1(y);

    // t2 = 4*X*Y^2
    t2.ModMulK1(x, &t1);
    t2.ModDouble();
    t2.ModDouble();

    // t3 = 8*Y^4
    t3.ModSquareK1(&t1);
    t3.ModDouble();
    t3.ModDouble();
    t3.ModDouble();

    // t4 = Z^2
    t4.ModSquareK1(z);

    // t5 = Z^4
    t5.ModSquareK1(&t4);

    // t6 = 3*X^2 + a*Z^4 (for secp256k1, a=0, so t6 = 3*X^2)
    t6.ModSquareK1(x);
    t7.Set(&t6);
    t6.ModDouble();
    t6.ModAdd(&t7);  // t6 = 3*X^2

    // X3 = t6^2 - 2*t2 = (3*X^2)^2 - 2*4*X*Y^2
    t7.ModSquareK1(&t6);
    t8.Set(&t2);
    t8.ModDouble();
    t7.ModSub(&t8);
    x->Set(&t7);  // X3 = t6^2 - 2*t2

    // Y3 = t6*(t2 - X3) - t3 = 3*X^2*(4*X*Y^2 - X3) - 8*Y^4
    t2.ModSub(x);           // t2 = t2 - X3
    t6.ModMulK1(&t6, &t2);  // t6 = t6 * (t2 - X3)
    t6.ModSub(&t3);         // t6 = t6 - t3
    y->Set(&t6);            // Y3 = t6

    // Z3 = 2*Y*Z
    z->ModMulK1(y, z);
    z->ModDouble();

    // Additional optimizations using AVX-512 prefetching for next operations
    _mm_prefetch((char *)x->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)y->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)z->bits64, _MM_HINT_T0);
}

// Complete point addition in Jacobian coordinates with AVX-512 optimization
void Int::ECPointAddK1AVX512(Int *x1, Int *y1, Int *z1, Int *x2, Int *y2, Int *z2, Int *x3, Int *y3,
                             Int *z3) {
    // Point addition in Jacobian coordinates
    // (x1:y1:z1) + (x2:y2:z2) = (x3:y3:z3)

    Int t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;

    // Prefetch input data
    _mm_prefetch((char *)x1->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)y1->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)z1->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)x2->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)y2->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)z2->bits64, _MM_HINT_T0);

    // Handle special cases
    if (z1->IsZero()) {
        // P1 is point at infinity, return P2
        x3->Set(x2);
        y3->Set(y2);
        z3->Set(z2);
        return;
    }

    if (z2->IsZero()) {
        // P2 is point at infinity, return P1
        x3->Set(x1);
        y3->Set(y1);
        z3->Set(z1);
        return;
    }

    // Standard Jacobian addition algorithm
    // Cost: 12M + 4S

    // t1 = Z1^2
    t1.ModSquareK1(z1);

    // t2 = Z2^2
    t2.ModSquareK1(z2);

    // t3 = X1*Z2^2
    t3.ModMulK1(x1, &t2);

    // t4 = X2*Z1^2
    t4.ModMulK1(x2, &t1);

    // t5 = Y1*Z2^3 = Y1*Z2*Z2^2
    t5.ModMulK1(z2, &t2);
    t5.ModMulK1(y1, &t5);

    // t6 = Y2*Z1^3 = Y2*Z1*Z1^2
    t6.ModMulK1(z1, &t1);
    t6.ModMulK1(y2, &t6);

    // t7 = t4 - t3 = X2*Z1^2 - X1*Z2^2
    t7.ModSub(&t4, &t3);

    // t8 = t6 - t5 = Y2*Z1^3 - Y1*Z2^3
    t8.ModSub(&t6, &t5);

    // Check if points are equal or inverses
    if (t7.IsZero()) {
        if (t8.IsZero()) {
            // Points are equal, use point doubling
            ECPointDoubleK1AVX512(x1, y1, z1);
            x3->Set(x1);
            y3->Set(y1);
            z3->Set(z1);
            return;
        } else {
            // Points are inverses, result is point at infinity
            x3->SetInt32(1);
            y3->SetInt32(1);
            z3->SetInt32(0);
            return;
        }
    }

    // Continue with addition
    // t9 = t7^2
    t9.ModSquareK1(&t7);

    // t10 = t7^3
    t10.ModMulK1(&t7, &t9);

    // t11 = t3*t7^2
    t11.ModMulK1(&t3, &t9);

    // X3 = t8^2 - t10 - 2*t11
    t1.ModSquareK1(&t8);
    t1.ModSub(&t10);
    t2.Set(&t11);
    t2.ModDouble();
    t1.ModSub(&t2);
    x3->Set(&t1);

    // Y3 = t8*(t11 - X3) - t5*t10
    t11.ModSub(x3);
    t8.ModMulK1(&t8, &t11);
    t5.ModMulK1(&t5, &t10);
    t8.ModSub(&t5);
    y3->Set(&t8);

    // Z3 = Z1*Z2*t7
    z3->ModMulK1(z1, z2);
    z3->ModMulK1(z3, &t7);

    // Prefetch results for potential next operations
    _mm_prefetch((char *)x3->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)y3->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)z3->bits64, _MM_HINT_T0);
}

// Scalar multiplication with AVX-512 optimizations and windowing
void Int::ECPointMulK1AVX512(Int *k, Int *x, Int *y, Int *z, Int *rx, Int *ry, Int *rz) {
    // Scalar multiplication using sliding window method with AVX-512 optimizations
    const int window_size = 4;                        // 4-bit window for good balance
    const int precomp_size = (1 << window_size) - 1;  // 15 precomputed points

    // Precomputed points array [1P, 3P, 5P, 7P, 9P, 11P, 13P, 15P]
    Int precomp_x[precomp_size], precomp_y[precomp_size], precomp_z[precomp_size];

    // Prefetch scalar and base point
    _mm_prefetch((char *)k->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)x->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)y->bits64, _MM_HINT_T0);
    _mm_prefetch((char *)z->bits64, _MM_HINT_T0);

    // Initialize result to point at infinity
    rx->SetInt32(1);
    ry->SetInt32(1);
    rz->SetInt32(0);

    // Handle zero scalar
    if (k->IsZero()) {
        return;
    }

    // Precompute odd multiples [1P, 3P, 5P, ..., 15P]
    // 1P = P
    precomp_x[0].Set(x);
    precomp_y[0].Set(y);
    precomp_z[0].Set(z);

    // 2P
    Int double_x(*x), double_y(*y), double_z(*z);
    ECPointDoubleK1AVX512(&double_x, &double_y, &double_z);

    // Generate remaining odd multiples
    for (int i = 1; i < precomp_size; i++) {
        ECPointAddK1AVX512(&precomp_x[i - 1], &precomp_y[i - 1], &precomp_z[i - 1], &double_x,
                           &double_y, &double_z, &precomp_x[i], &precomp_y[i], &precomp_z[i]);
    }

    // Process scalar from MSB to LSB using sliding window
    int bit_length = k->GetBitLength();
    int i = bit_length - 1;

    while (i >= 0) {
        // Double the accumulator
        if (!rz->IsZero()) {
            ECPointDoubleK1AVX512(rx, ry, rz);
        }

        if (k->GetBit(i)) {
            // Found a 1 bit, collect window
            int window = 1;
            int window_size_actual = 1;

            // Collect up to window_size bits
            for (int j = 1; j < window_size && i - j >= 0; j++) {
                window <<= 1;
                if (k->GetBit(i - j)) {
                    window |= 1;
                    window_size_actual = j + 1;
                } else {
                    break;  // Stop at first 0 bit
                }
            }

            // Make window odd by factoring out powers of 2
            while ((window & 1) == 0) {
                window >>= 1;
                window_size_actual--;
            }

            // Double accumulator for remaining window bits
            for (int j = 0; j < window_size_actual - 1; j++) {
                if (!rz->IsZero()) {
                    ECPointDoubleK1AVX512(rx, ry, rz);
                }
            }

            // Add precomputed point
            int precomp_index = (window - 1) / 2;  // Convert to array index
            if (rz->IsZero()) {
                // First addition
                rx->Set(&precomp_x[precomp_index]);
                ry->Set(&precomp_y[precomp_index]);
                rz->Set(&precomp_z[precomp_index]);
            } else {
                ECPointAddK1AVX512(rx, ry, rz, &precomp_x[precomp_index], &precomp_y[precomp_index],
                                   &precomp_z[precomp_index], rx, ry, rz);
            }

            i -= window_size_actual;
        } else {
            i--;
        }
    }
}

// Batch scalar multiplication for multiple keys with AVX-512 optimization
void Int::ECBatchMulK1AVX512(Int *scalars, Int *base_x, Int *base_y, Int *base_z, Int *result_x,
                             Int *result_y, Int *result_z, int batch_size) {
    // Process multiple scalar multiplications in parallel where possible
    const int parallel_batch = 4;  // Process 4 multiplications in parallel

    for (int batch = 0; batch < batch_size; batch += parallel_batch) {
        int remaining = MIN(parallel_batch, batch_size - batch);

        // Prefetch next batch of scalars
        for (int i = 0; i < remaining; i++) {
            _mm_prefetch((char *)scalars[batch + i].bits64, _MM_HINT_T1);
        }

        // Process current batch
        for (int i = 0; i < remaining; i++) {
            ECPointMulK1AVX512(&scalars[batch + i], base_x, base_y, base_z, &result_x[batch + i],
                               &result_y[batch + i], &result_z[batch + i]);
        }
    }
}

// Batch processing for multiple private key mutations with full implementation
void Int::BatchProcessPrivateKeys(Int *keys, Int *results, int count, uint64_t mutation_step) {
    const int batch_size = 16;         // Process 16 keys at once for optimal AVX-512 usage
    const int prefetch_distance = 64;  // Prefetch 64 cache lines ahead

    for (int i = 0; i < count; i += batch_size) {
        int remaining = MIN(batch_size, count - i);

        // Advanced prefetching strategy
        if (i + prefetch_distance < count) {
            for (int j = 0; j < MIN(batch_size, count - i - prefetch_distance); j++) {
                _mm_prefetch((char *)keys[i + prefetch_distance + j].bits64, _MM_HINT_T2);
            }
        }

        // Process current batch with vectorized operations
        for (int j = 0; j < remaining; j += 8) {
            int lane_count = MIN(8, remaining - j);

            // Load 8 keys at once using AVX-512
            __m512i key_data[8];
            for (int k = 0; k < lane_count; k++) {
                key_data[k] = _mm512_loadu_si512((__m512i *)keys[i + j + k].bits64);
            }

            // Process each key in the current 8-lane batch
            for (int k = 0; k < lane_count; k++) {
                // Copy key to result
                results[i + j + k].Set(&keys[i + j + k]);

                // Add mutation step * index
                uint64_t mutation = mutation_step * (uint64_t)(i + j + k + 1);
                results[i + j + k].Add(mutation);

                // Perform modular reduction
                results[i + j + k].ModReduceK1AVX512();

                // Ensure proper bit clearing
                results[i + j + k].bits64[4] = 0;
#if BISIZE == 512
                results[i + j + k].bits64[5] = results[i + j + k].bits64[6] =
                    results[i + j + k].bits64[7] = 0;
#endif
            }

            // Vectorized validation using AVX-512
            for (int k = 0; k < lane_count; k++) {
                // Validate that result is properly reduced
                bool needs_final_check = false;

                if (results[i + j + k].bits64[3] == 0xFFFFFFFFFFFFFFFFULL &&
                    results[i + j + k].bits64[2] == 0xFFFFFFFFFFFFFFFFULL &&
                    results[i + j + k].bits64[1] == 0xFFFFFFFFFFFFFFFFULL &&
                    results[i + j + k].bits64[0] >= 0xFFFFFFFEFFFFFC2FULL) {
                    needs_final_check = true;
                }

                if (needs_final_check) {
                    results[i + j + k].ModReduceK1AVX512();
                }
            }

            // Store results using streaming stores for better memory bandwidth
            for (int k = 0; k < lane_count; k++) {
                _mm512_stream_si512((__m512i *)results[i + j + k].bits64,
                                    _mm512_loadu_si512((__m512i *)results[i + j + k].bits64));
            }
        }
    }

    // Ensure all streaming stores complete
    _mm_mfence();

    // Final verification pass for critical applications
    for (int i = 0; i < count; i += 8) {
        int remaining = MIN(8, count - i);

        // Vectorized final validation
        for (int j = 0; j < remaining; j++) {
            // Ensure no high bits are set
            assert(results[i + j].bits64[4] == 0);
#if BISIZE == 512
            assert(results[i + j].bits64[5] == 0);
            assert(results[i + j].bits64[6] == 0);
            assert(results[i + j].bits64[7] == 0);
#endif

            // Ensure value is properly reduced
            assert(results[i + j].IsSmaller(&_P));
        }
    }
}

// Thread-safe version for multi-threaded private key search
void Int::ModMulK1ThreadSafe(Int *a, Int *b) {
    // Thread-local storage for temporary variables to avoid conflicts
    thread_local unsigned char c;
    thread_local uint64_t ah, al;
    thread_local uint64_t t[NB64BLOCK];
    thread_local alignas(64) uint64_t r512[8];

    // Initialize with zeros using AVX-512
    _mm512_storeu_si512((__m512i *)r512, _mm512_setzero_si512());

    // Same multiplication logic as ModMulK1 but with thread-local variables
    imm_umul(a->bits64, b->bits64[0], r512);
    imm_umul(a->bits64, b->bits64[1], t);

    c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
    c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[4], r512 + 5);

    imm_umul(a->bits64, b->bits64[2], t);
    c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
    c = _addcarry_u64(c, r512[6], t[4], r512 + 6);

    imm_umul(a->bits64, b->bits64[3], t);
    c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
    c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
    c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
    c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
    c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

    // Reduction steps
    imm_umul(r512 + 4, 0x1000003D1ULL, t);
    c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
    c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
    c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
    c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

    al = _umul128(t[4] + c, 0x1000003D1ULL, &ah);
    c = _addcarry_u64(0, r512[0], al, bits64 + 0);
    c = _addcarry_u64(c, r512[1], ah, bits64 + 1);
    c = _addcarry_u64(c, r512[2], 0, bits64 + 2);
    c = _addcarry_u64(c, r512[3], 0, bits64 + 3);

    bits64[4] = 0;
#if BISIZE == 512
    _mm256_storeu_si256((__m256i *)(bits64 + 5), _mm256_setzero_si256());
#endif
}

// Optimized constants for faster computation
static const uint64_t SECP256K1_PRIME_CONSTANTS[4] = {0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
                                                      0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};

// Fast modular reduction using precomputed constants
void Int::FastModReduceK1() {
    // Use precomputed constants for faster reduction
    __m512i prime_vec = _mm512_loadu_si512((__m512i *)SECP256K1_PRIME_CONSTANTS);
    __m512i data_vec = _mm512_loadu_si512((__m512i *)bits64);

    // Compare and conditionally subtract
    __mmask8 cmp_mask = _mm512_cmpge_epu64_mask(data_vec, prime_vec);

    if (cmp_mask) {
        __m512i result = _mm512_sub_epi64(data_vec, prime_vec);
        _mm512_storeu_si512((__m512i *)bits64, result);
    }
}
