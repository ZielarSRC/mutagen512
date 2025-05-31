// Update state
state[0] = _mm512_add_epi32(state[0], a);
state[1] = _mm512_add_epi32(state[1], b);
state[2] = _mm512_add_epi32(state[2], c);
state[3] = _mm512_add_epi32(state[3], d);
state[4] = _mm512_add_epi32(state[4], e);
state[5] = _mm512_add_epi32(state[5], f);
state[6] = _mm512_add_epi32(state[6], g);
state[7] = _mm512_add_epi32(state[7], h);
}

}  // namespace _sha256avx512

// Process 16 message blocks in parallel with AVX-512
void sha256avx512_16B(
    const uint8_t* data0, const uint8_t* data1, const uint8_t* data2,
    const uint8_t* data3, const uint8_t* data4, const uint8_t* data5,
    const uint8_t* data6, const uint8_t* data7, const uint8_t* data8,
    const uint8_t* data9, const uint8_t* data10, const uint8_t* data11,
    const uint8_t* data12, const uint8_t* data13, const uint8_t* data14,
    const uint8_t* data15, unsigned char* hash0, unsigned char* hash1,
    unsigned char* hash2, unsigned char* hash3, unsigned char* hash4,
    unsigned char* hash5, unsigned char* hash6, unsigned char* hash7,
    unsigned char* hash8, unsigned char* hash9, unsigned char* hash10,
    unsigned char* hash11, unsigned char* hash12, unsigned char* hash13,
    unsigned char* hash14, unsigned char* hash15) {
  ALIGN64 __m512i state[8];
  const uint8_t* data[16] = {data0,  data1,  data2,  data3, data4,  data5,
                             data6,  data7,  data8,  data9, data10, data11,
                             data12, data13, data14, data15};
  unsigned char* hashArray[16] = {hash0,  hash1,  hash2,  hash3, hash4,  hash5,
                                  hash6,  hash7,  hash8,  hash9, hash10, hash11,
                                  hash12, hash13, hash14, hash15};

  // Initialize the state with the initial hash values
  _sha256avx512::Initialize(state);

  // Process the data blocks
  _sha256avx512::Transform(state, data);

  // Extract and store hash values
  ALIGN64 uint32_t digest[8][16];  // digest[state_index][element_index]

  for (int i = 0; i < 8; ++i) {
    _mm512_store_si512((__m512i*)digest[i], state[i]);
  }

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 8; ++j) {
      uint32_t word = digest[j][i];
#ifdef _MSC_VER
      word = _byteswap_ulong(word);
#else
      word = __builtin_bswap32(word);
#endif
      memcpy(hashArray[i] + j * 4, &word, 4);
    }
  }
}
