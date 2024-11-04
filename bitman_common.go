package fishtable

import (
	"math"
	"math/bits"
)

// TODO: Use bucket size [16, 32) such that sizeof(bucket) is nice for the cachelines/allocator/page_size/etc
const bucketSize = 16

type matchiter struct{ hashMatches uint32 }

func (m *matchiter) HasCurrent() bool {
	return m.hashMatches != 0
}

func (m *matchiter) Current() int {
	return bits.TrailingZeros32(m.hashMatches)
}

func (m *matchiter) Advance() {
	// unset the lowest set bit
	// BLSR — Reset Lowest Set Bit - works only on 32 or 64 bit integers (and
	// requires compiling with GOAMD64=v3)
	m.hashMatches = m.hashMatches & (m.hashMatches - 1)
}

func (m *matchiter) Count() int {
	return bits.OnesCount32(m.hashMatches)
}

func (m *matchiter) AsBitmask() bucketBitmask {
	return bucketBitmask(m.hashMatches)
}

func (m *matchiter) Keep(mask bucketBitmask) matchiter {
	return matchiter{hashMatches: uint32(m.hashMatches) & uint32(mask)}
}

func (m *matchiter) RotateRight(n int) matchiter {
	res := bits.RotateLeft16(uint16(m.hashMatches), -n)
	return matchiter{hashMatches: uint32(res)}
}

type bucketBitmask uint32

func (bm bucketBitmask) IsAllSet() bool {
	return bm == math.MaxUint16
}

func (bm bucketBitmask) IsEmpty() bool {
	return bm == 0
}

func (bm bucketBitmask) Toggle() bucketBitmask {
	return bucketBitmask(^uint16(bm))
}

func (bm bucketBitmask) IsMarked(slot int) bool {
	// NOTE: Converting bm to uint32 makes the compiler use the BT instruction here
	// which makes it go faster
	return uint32(bm)&(1<<(slot%32)) != 0
}

func (bm bucketBitmask) Count() int {
	return bits.OnesCount32(uint32(bm))
}

func (bm *bucketBitmask) Mark(slot int) {
	*bm |= 1 << (slot % 32)
}

func (bm *bucketBitmask) MaybeMark(slot int, bit uintptr) {
	*bm |= bucketBitmask(bit << slot)
}

func (bm *bucketBitmask) Unmark(slot int) {
	*bm &^= 1 << (slot % 32)
}

func (bm *bucketBitmask) UnmarkAll(mask bucketBitmask) {
	*bm &^= mask
}

func (bm *bucketBitmask) MarkFirst(n int) {
	*bm |= (math.MaxUint16) >> (bucketSize - n)
}

func (bm bucketBitmask) FirstUnmarkedSlot() int {
	return bits.TrailingZeros32(^uint32(bm))
}

func (bm bucketBitmask) AsMatchiter() matchiter {
	return matchiter{hashMatches: uint32(bm)}
}
