//go:build !amd64 || nosimd

package fishtable

import (
	"encoding/binary"
	"math"
	"math/bits"
)

type (
	fingerprintProbe uint64
)

func makeHashProbe(fingerprint fingerprint) fingerprintProbe {
	mask := (math.MaxUint64 / 255) * uint64(fingerprint&0b1111_1111)
	return fingerprintProbe(mask)
}

type bucketFinder struct {
	fingerprintsle0 uint64
	fingerprintsle1 uint64
}

func bucketFinderFrom(fingerprints *[bucketSize]byte) bucketFinder {
	return bucketFinder{
		fingerprintsle0: binary.LittleEndian.Uint64(fingerprints[0:]),
		fingerprintsle1: binary.LittleEndian.Uint64(fingerprints[8:]),
	}
}

func findZeros64(v uint64) uint64 {
	const c1 = (math.MaxUint64 / 255) * 0b0111_1111
	const topbit = (math.MaxUint64 / 255) * 0b1000_0000
	return ^((v&c1 + c1) | v) & topbit
}

func (b *bucketFinder) ProbeHashMatches(probe fingerprintProbe) (hashes matchiter) {
	hashMatches0 := findZeros64(b.fingerprintsle0 ^ uint64(probe))
	hashMatches1 := findZeros64(b.fingerprintsle1 ^ uint64(probe))

	// 64-bit MOVMSKB by doing a 128bit mul
	// have: 0b10000000_00000000_10000000_00000000_10000000_00000000_10000000_00000000
	// want: 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_10101010
	const mul = (1 << (8*8 - 7)) |
		(1 << (7*8 - 6)) |
		(1 << (6*8 - 5)) |
		(1 << (5*8 - 4)) |
		(1 << (4*8 - 3)) |
		(1 << (3*8 - 2)) |
		(1 << (2*8 - 1)) |
		(1 << (1*8 - 0))

	m1, _ := bits.Mul64(hashMatches0, mul)
	m2, _ := bits.Mul64(hashMatches1, mul)
	hashes.hashMatches = uint32(m1&0b1111_1111) | uint32((m2&0b1111_1111)<<8)
	return

	// Alternate algo: Slightly slower cuz branches
	// // PAIN. Pack the [0, 16) matches into a 16-bit integer. Luckily we usually only have one hit.
	// // hashMatches0 are packed to bit range [0, 8)
	// // hashMatches1 are packed to bit range [8, 16)

	// // ALGORITHM: We first make it so that TrailingZeroes will return an odd number
	// // for hashMatches0 and an even number for hashMatches1 so that based on that property
	// // we know if the bit belongs to the [8, 16) bit range
	// matches := hashMatches0>>1 | hashMatches1
	// for ; matches != 0; matches = matches & (matches - 1) {
	// 	bit := bits.TrailingZeros64(matches)
	// 	adjust := 8 * (bit & 1)
	// 	hashes.hashMatches |= (1 << (bit/8 + adjust))
	// }
}
