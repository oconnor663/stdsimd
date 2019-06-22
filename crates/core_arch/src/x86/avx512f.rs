use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::{self, transmute},
    ptr,
};

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Computes the absolute values of packed 32-bit integers in `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990,33&text=_mm512_abs_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsd))]
pub unsafe fn _mm512_abs_epi32(a: __m512i) -> __m512i {
    let a = a.as_i32x16();
    // all-0 is a properly initialized i32x16
    let zero: i32x16 = mem::zeroed();
    let sub = simd_sub(zero, a);
    let cmp: i32x16 = simd_gt(a, zero);
    transmute(simd_select(cmp, a, sub))
}

/// Computes the absolute value of packed 32-bit integers in `a`, and store the
/// unsigned results in `dst` using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990,33&text=_mm512_abs_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsd))]
pub unsafe fn _mm512_mask_abs_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi32(a).as_i32x16();
    transmute(simd_select_bitmask(k, abs, src.as_i32x16()))
}

/// Computes the absolute value of packed 32-bit integers in `a`, and store the
/// unsigned results in `dst` using zeromask `k` (elements are zeroed out when
/// the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990,33,34,35,35&text=_mm512_maskz_abs_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsd))]
pub unsafe fn _mm512_maskz_abs_epi32(k: __mmask16, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi32(a).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Returns vector of type `__m512i` with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setzero_si512)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm512_setzero_si512() -> __m512i {
    // All-0 is a properly initialized __m512i
    mem::zeroed()
}

/// Sets packed 32-bit integers in `dst` with the supplied values in reverse
/// order.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_setr_epi32(
    e15: i32,
    e14: i32,
    e13: i32,
    e12: i32,
    e11: i32,
    e10: i32,
    e9: i32,
    e8: i32,
    e7: i32,
    e6: i32,
    e5: i32,
    e4: i32,
    e3: i32,
    e2: i32,
    e1: i32,
    e0: i32,
) -> __m512i {
    let r = i32x16(
        e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0,
    );
    transmute(r)
}

/// Broadcast 64-bit integer `a` to all elements of `dst`.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_set1_epi64(a: i64) -> __m512i {
    transmute(i64x8::splat(a))
}

#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))] // FIXME vmovdqu32 expected
pub unsafe fn _mm512_loadu_si512(mem_addr: *const __m512i) -> __m512i {
    ptr::read_unaligned(mem_addr)
}

#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))] // FIXME vmovdqu32 expected
pub unsafe fn _mm512_storeu_si512(mem_addr: *mut __m512i, a: __m512i) {
    ptr::write_unaligned(mem_addr, a)
}

#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_add_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i32x16(), b.as_i32x16()))
}

#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_add_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i64x8(), b.as_i64x8()))
}

#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmpeqd))]
pub unsafe fn _mm512_cmpeq_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute::<i32x16, _>(simd_eq(a.as_i32x16(), b.as_i32x16()))
}

#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmpeqq))]
pub unsafe fn _mm512_cmpeq_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute::<i64x8, _>(simd_eq(a.as_i64x8(), b.as_i64x8()))
}

#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpandq))]
pub unsafe fn _mm512_and_si512(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_and(a.as_i64x8(), b.as_i64x8()))
}

#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vandnps))]
pub unsafe fn _mm512_andnot_si512(a: __m512i, b: __m512i) -> __m512i {
    let mut all_ones: __m512i = mem::uninitialized();
    ptr::write_bytes(&mut all_ones, 0xff, 1);
    transmute(simd_and(
        simd_xor(a.as_i64x8(), all_ones.as_i64x8()),
        b.as_i64x8(),
    ))
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
#[cfg_attr(test, assert_instr(vprord, imm8 = 3))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm256_ror_epi32(a: __m256i, imm8: i32) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            vprord_256(a, $imm8)
        };
    }
    constify_imm8!(imm8, call)
}

#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
#[cfg_attr(test, assert_instr(vprorq, imm8 = 3))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm256_ror_epi64(a: __m256i, imm8: i32) -> __m256i {
    macro_rules! call {
        ($imm8:expr) => {
            vprorq_256(a, $imm8)
        };
    }
    constify_imm8!(imm8, call)
}

#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprord, imm8 = 3))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_ror_epi32(a: __m512i, imm8: i32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vprord_512(a, $imm8)
        };
    }
    constify_imm8!(imm8, call)
}

#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vprorq, imm8 = 3))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_ror_epi64(a: __m512i, imm8: i32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vprorq_512(a, $imm8)
        };
    }
    constify_imm8!(imm8, call)
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.pror.d.256"]
    fn vprord_256(a: __m256i, imm8: i32) -> __m256i;
    #[link_name = "llvm.x86.avx512.pror.q.256"]
    fn vprorq_256(a: __m256i, imm8: i32) -> __m256i;
    #[link_name = "llvm.x86.avx512.pror.d.512"]
    fn vprord_512(a: __m512i, imm8: i32) -> __m512i;
    #[link_name = "llvm.x86.avx512.pror.q.512"]
    fn vprorq_512(a: __m512i, imm8: i32) -> __m512i;
}

#[cfg(test)]
mod tests {
    use std;
    use stdsimd_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_abs_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, std::i32::MAX,
            std::i32::MIN, 100, -100, -32,
            0, 1, -1, std::i32::MAX,
            std::i32::MIN, 100, -100, -32,
        );
        let r = _mm512_abs_epi32(a);
        let e = _mm512_setr_epi32(
            0,
            1,
            1,
            std::i32::MAX,
            std::i32::MAX.wrapping_add(1),
            100,
            100,
            32,
            0,
            1,
            1,
            std::i32::MAX,
            std::i32::MAX.wrapping_add(1),
            100,
            100,
            32,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_abs_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, std::i32::MAX,
            std::i32::MIN, 100, -100, -32,
            0, 1, -1, std::i32::MAX,
            std::i32::MIN, 100, -100, -32,
        );
        let r = _mm512_mask_abs_epi32(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_abs_epi32(a, 0b11111111, a);
        let e = _mm512_setr_epi32(
            0,
            1,
            1,
            std::i32::MAX,
            std::i32::MAX.wrapping_add(1),
            100,
            100,
            32,
            0,
            1,
            -1,
            std::i32::MAX,
            std::i32::MIN,
            100,
            -100,
            -32,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_abs_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, std::i32::MAX,
            std::i32::MIN, 100, -100, -32,
            0, 1, -1, std::i32::MAX,
            std::i32::MIN, 100, -100, -32,
        );
        let r = _mm512_maskz_abs_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_abs_epi32(0b11111111, a);
        let e = _mm512_setr_epi32(
            0,
            1,
            1,
            std::i32::MAX,
            std::i32::MAX.wrapping_add(1),
            100,
            100,
            32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    SOMETHING ABOUT THESE ROTATIONS ISNT WORKING -- the 32 bit ones seem to get the 64 bit instructions
    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_ror_epi32() {
        let a: [u32; 8] = [0, 1, 2, 4, 8, 16, 32, 64];
        let b: [u32; 8] = transmute(_mm256_ror_epi32(transmute(a), 1));
        assert_eq!(b, [0, 2147483648, 1, 2, 4, 8, 16, 32]);
    }

    #[simd_test(enable = "avx512f,avx512vl")]
    unsafe fn test_mm256_ror_epi64() {
        let a: [u64; 4] = [0, 1, 2, 4];
        let b: [u64; 4] = transmute(_mm256_ror_epi64(transmute(a), 1));
        assert_eq!(b, [0, 9223372036854775808, 1, 2]);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_ror_epi32() {
        let a: [u32; 16] = [
            0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
        ];
        let b: [u32; 16] = transmute(_mm512_ror_epi32(transmute(a), 1));
        assert_eq!(
            b,
            [0, 2147483648, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        );
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_ror_epi64() {
        let a: [u64; 8] = [0, 1, 2, 4, 8, 16, 32, 64];
        let b: [u64; 8] = transmute(_mm512_ror_epi64(transmute(a), 1));
        assert_eq!(b, [0, 9223372036854775808, 1, 2, 4, 8, 16, 32]);
    }
}
