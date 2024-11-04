package fishtable

import (
	"math/bits"
	"unsafe"
)

type Kv[K comparable, V any] struct {
	Key   K
	Value V
}

func New[K comparable, V any]() *Map[K, V] {
	return &Map[K, V]{
		hasher: getRuntimeHasher[K](),
		// hasher: func(key unsafe.Pointer, seed uintptr) uintptr {
		// 	// random ass hash function for cheating in the benchmarks
		// 	k := *(*uint64)(key)
		// 	k *= uint64(seed)
		// 	k, k2 := bits.Mul64(k, 11400714819323198486)
		// 	return uintptr(bits.RotateLeft64(k+k2, 37))
		// },
		trie: sarray[*bucketMap[K, V]]{shift: 64},
		seed: uintptr(runtime_fastrand64()),
	}
}

func NewWithUnsafeHasher[K comparable, V any](hasher func(key unsafe.Pointer, seed uintptr) uintptr) *Map[K, V] {
	return &Map[K, V]{
		hasher: hasher,
		trie:   sarray[*bucketMap[K, V]]{shift: 64},
		seed:   uintptr(runtime_fastrand64()),
	}
}

type fingerprint = uint64

func fixFingerprint(hash uint64) fingerprint {
	// NOTE: We could reserve fingerprint value of 0 to indicate a slot
	// that is empty. This would allow us to avoid having to maintain
	// the presents bitset for each bucket.
	return hash
}

// arrays are just like slices but they don't have the third and useless capacity field
// also its size is pow2
type sarray[V any] struct {
	first *V
	shift uintptr
}

func asSArray[T any](s []T) sarray[T] {
	if len(s) == 0 {
		return sarray[T]{nil, 64}
	}
	return sarray[T]{&s[0], uintptr(64 - (bits.Len(uint(len(s))) - 1))}
}

func (a *sarray[V]) IsEmpty() bool {
	return a.shift >= 64
}

func (a *sarray[V]) NotEmpty() bool {
	return a.shift < 64
}

func (a *sarray[V]) Len() int {
	if a.shift >= 64 {
		return 0
	}
	return 1 << (64 - a.shift)
}

func (a *sarray[V]) Slice() []V {
	return unsafe.Slice(a.first, a.Len())
}

func (a *sarray[V]) Get(i uintptr) V {
	return *(*V)(unsafe.Add(unsafe.Pointer(a.first), unsafe.Sizeof(*a.first)*i))
}

func (a *sarray[V]) GetLocation(i uintptr) *V {
	return (*V)(unsafe.Add(unsafe.Pointer(a.first), unsafe.Sizeof(*a.first)*i))
}

func (a *sarray[V]) Set(i uintptr, v V) {
	*(*V)(unsafe.Add(unsafe.Pointer(a.first), unsafe.Sizeof(*a.first)*i)) = v
}

type Map[K comparable, V any] struct {
	hasher func(key unsafe.Pointer, seed uintptr) uintptr
	trie   sarray[*bucketMap[K, V]]
	// the initial map that we use before the first split() and the trie creation. If the trie exists
	// then this field is used as the head of a free-list
	map0 *bucketMap[K, V]
	seed uintptr
	len  int

	// Real languages have bulk free() so freeing all of the the little maps
	// should be fast.

	// NOTE: How to get the struct size down to 32 bytes.
	// 	*) Inline the hash function
	// 	*) Pack the trie shift into the 6 top bits of seed
}

type bucketMap[K comparable, V any] struct {
	fingerprints [bucketSize]uint8
	present      bucketBitmask
	depth        uint8 // <= 64
	cap          uint8 // {2, 4, 8, 16}
	overflow     *bucketMap[K, V]

	// NOTE: Then comes a flexible array member that contains the kvs. We access it with unsafe
	// kvs        [cap]Kv[K, V]

	// TODO: It would be mildly interesting to test buckets of 32 entries or perhaps even 64 entries.
	// It would surely require changing to use 16-bit tophashes to keep collisions at bay.

	// TODO: If we wanted to be nasty and abuse the fact that for good hash functions overflows
	// are rare (~1/40 buckets have overflow when Len() >= 100k) then we could have a dedicated structs for the cases
	// when there is an overflow and when there is not. Initially we allocate object without the field
	// but if it's needed then we allocate object with that field and move over to using that one instead.
	// Requires adding one bool field.
	//
	// Maybe there should be a centralized associative data structure that keeps track of the overflow buckets
}

func (b *bucketMap[K, V]) GetKv(i int) *Kv[K, V] {
	return (*Kv[K, V])(unsafe.Add(unsafe.Pointer(b.first()), unsafe.Sizeof(*b.first())*uintptr(i)))
}

func (b *bucketMap[K, V]) SetKv(i int, v Kv[K, V]) {
	*(*Kv[K, V])(unsafe.Add(unsafe.Pointer(b.first()), unsafe.Sizeof(*b.first())*uintptr(i))) = v
}

func (b *bucketMap[K, V]) Kvs() []Kv[K, V] {
	return unsafe.Slice(b.first(), b.cap)
}

// returns the pointer to the first kv entry
func (b *bucketMap[K, V]) first() *Kv[K, V] {
	// find the harmony in newBucketMap()
	// TODO: This is nasty and assumes things that might not be true in Go
	p := (*struct {
		m   bucketMap[K, V]
		kvs [1]Kv[K, V]
	})(unsafe.Pointer(b))
	return &p.kvs[0]
}

func newBucketMap[K comparable, V any](cap uint8) *bucketMap[K, V] {
	switch cap {
	case 2:
		p := new(struct {
			m   bucketMap[K, V]
			kvs [2]Kv[K, V]
		})
		p.m.cap = cap
		return &p.m
	case 4:
		p := new(struct {
			m   bucketMap[K, V]
			kvs [4]Kv[K, V]
		})
		p.m.cap = cap
		return &p.m
	case 8:
		p := new(struct {
			m   bucketMap[K, V]
			kvs [8]Kv[K, V]
		})
		p.m.cap = cap
		return &p.m
	case 16:
		p := new(struct {
			m   bucketMap[K, V]
			kvs [16]Kv[K, V]
		})
		p.m.cap = cap
		return &p.m
	default:
		println(cap)
		panic("bad cap")
	}
}

func (m *Map[K, V]) Get(k K) (v V, ok bool) {
	hash := m.hasher(noescape(unsafe.Pointer(&k)), m.seed)
	fingerprint := fixFingerprint(uint64(hash))
	probe := makeHashProbe(fingerprint)

	bucket := m.map0
	if trie := m.trie; trie.shift < 64 {
		bucket = trie.Get(hash >> trie.shift)
	}

	for ; bucket != nil; bucket = bucket.overflow {
		finder := bucketFinderFrom(&bucket.fingerprints)

		matches := finder.ProbeHashMatches(probe)
		matches = matches.Keep(bucket.present)
		for ; matches.HasCurrent(); matches.Advance() {
			slotInBucket := matches.Current()
			kv := bucket.GetKv(slotInBucket)
			if kv.Key == k {
				return kv.Value, true
			}
		}
	}
	return
}

func (m *Map[K, V]) Delete(k K) {
	hash := m.hasher(noescape(unsafe.Pointer(&k)), m.seed)
	fingerprint := fixFingerprint(uint64(hash))
	probe := makeHashProbe(fingerprint)

	bucket := m.map0
	if trie := m.trie; trie.shift < 64 {
		bucket = trie.Get(hash >> trie.shift)
	}

	for ; bucket != nil; bucket = bucket.overflow {
		finder := bucketFinderFrom(&bucket.fingerprints)

		matches := finder.ProbeHashMatches(probe)
		matches = matches.Keep(bucket.present)
		for ; matches.HasCurrent(); matches.Advance() {
			slotInBucket := matches.Current()
			kv := bucket.GetKv(slotInBucket)
			if kv.Key == k {
				*kv = Kv[K, V]{} // for GC
				bucket.present.Unmark(slotInBucket)
				m.len--
				// NOTE: Releasing empty overflow buckets is deferred until the next Put()
				// into this trie bucket
				return
			}
		}
	}
}

func (m *Map[K, V]) Put(k K, v V) {
	hash := m.hasher(noescape(unsafe.Pointer(&k)), m.seed)
	fingerprint := fixFingerprint(uint64(hash))
	probe := makeHashProbe(fingerprint)

	bucket := m.map0
	trie := m.trie
	if trie.shift < 64 {
		bucket = trie.Get(hash >> trie.shift)
	}

	for {
		// bucket is nil for empty maps
		if bucket != nil {
			finder := bucketFinderFrom(&bucket.fingerprints)

			// Any matching hashes?
			matches := finder.ProbeHashMatches(probe)
			matches = matches.Keep(bucket.present) // is there an entry actually present there?
			for ; matches.HasCurrent(); matches.Advance() {
				slotInBucket := matches.Current()
				if kv := bucket.GetKv(slotInBucket); kv.Key == k {
					kv.Value = v
					return
				}
			}

			// If we reached the end of the linked list we know that the Key is not in this map.
			// We can try inserting then if there's free space to do so.
			if bucket.overflow == nil {
				nextFree := bucket.present.FirstUnmarkedSlot()
				if nextFree < int(bucket.cap) {
					bucket.SetKv(nextFree, Kv[K, V]{Key: k, Value: v})
					bucket.fingerprints[nextFree] = uint8(fingerprint)
					bucket.present.Mark(nextFree)
					m.len++
					return
				}
			}
		}

		trie, bucket = m.makeSpaceForHash(trie, bucket, hash)
	}
}

type Entry[K comparable, V any] struct {
	bucket       *bucketMap[K, V]
	m            *Map[K, V]
	slotInBucket int
}

func (e *Entry[K, V]) IsPresent() bool {
	return e.bucket.present.IsMarked(e.slotInBucket)
}

func (e *Entry[K, V]) Get() (V, bool) {
	return e.bucket.GetKv(e.slotInBucket).Value, e.bucket.present.IsMarked(e.slotInBucket)
}

func (e *Entry[K, V]) Set(v V) {
	e.bucket.GetKv(e.slotInBucket).Value = v

	// Increment map len if the entry was not previously present
	wasPresent := e.bucket.present.IsMarked(e.slotInBucket)
	var add int
	if !wasPresent {
		add = 1
	}
	e.m.len += add

	e.bucket.present.Mark(e.slotInBucket)
}

// Returns an Entry that allows inspecting the current state for the given key in the map.
//
// If the key doesn't already exist it doesn't insert the key into the map but will ensure enough space for it
// to be inserted.
//
// WARNING: DO NOT modify the map in any other way other than through the entry object while inspecting the entry
func (m *Map[K, V]) Entry(k K) Entry[K, V] {
	hash := m.hasher(noescape(unsafe.Pointer(&k)), m.seed)
	fingerprint := fixFingerprint(uint64(hash))
	probe := makeHashProbe(fingerprint)

	bucket := m.map0
	trie := m.trie
	if trie.shift < 64 {
		bucket = trie.Get(hash >> trie.shift)
	}

	for {
		// bucket is nil for empty maps
		if bucket != nil {
			finder := bucketFinderFrom(&bucket.fingerprints)

			// Any matching hashes?
			matches := finder.ProbeHashMatches(probe)
			matches = matches.Keep(bucket.present) // is there an entry actually present there?
			for ; matches.HasCurrent(); matches.Advance() {
				slotInBucket := matches.Current()
				if kv := bucket.GetKv(slotInBucket); kv.Key == k {
					return Entry[K, V]{m: m, bucket: bucket, slotInBucket: slotInBucket}
				}
			}

			// If we reached the end of the linked list we know that the Key is not in this map.
			// We can try inserting then if there's free space to do so.
			if bucket.overflow == nil {
				nextFree := bucket.present.FirstUnmarkedSlot()
				if nextFree < int(bucket.cap) {
					// We can store the key now into the bucket even if we don't end up
					// inserting anything into the entry. But this may leak memory the
					// key is pointing at.
					bucket.GetKv(nextFree).Key = k
					bucket.fingerprints[nextFree] = uint8(fingerprint)
					return Entry[K, V]{m: m, bucket: bucket, slotInBucket: nextFree}
				}
			}
		}

		trie, bucket = m.makeSpaceForHash(trie, bucket, hash)
	}
}

// figures out a good bucket for an entry with the given hash.
func (m *Map[K, V]) makeSpaceForHash(trie sarray[*bucketMap[K, V]], bucket *bucketMap[K, V], hash uintptr) (sarray[*bucketMap[K, V]], *bucketMap[K, V]) {
	// bucket/map0 is nil if the map is empty
	if bucket == nil {
		if trie.NotEmpty() {
			panic("bad")
		}
		m.map0 = newBucketMap[K, V](4) // start at cap 4
		return trie, m.map0
	}

	// Are we still in the lookup phase trying to just check if the linked list of buckets contains the key
	if bucket.overflow != nil {
		// Also use this as an opportunity to remove useless overflow buckets.
		//
		// After doing many deletes we may have ended up with useless overflow buckets. Unless the hash function is shit
		// the linked list of buckets isn't longer than 2 buckets (initial + overflow1).
		if bucket.present.IsAllSet() {
			// Nothing to do, continue to the overflow
			return trie, bucket.overflow
		}

		// Could the overflow bucket fit into this bucket?
		freeSpace := int(bucket.cap) - bucket.present.Count()
		if freeSpace < bucket.overflow.present.Count() {
			// Not enough free space to merge the two buckets. Continue to overflow
			return trie, bucket.overflow
		}

		// Move the overflow bucket's contents into this bucket
		for presents := bucket.overflow.present.AsMatchiter(); presents.HasCurrent(); presents.Advance() {
			slotInBucket := presents.Current()

			nextFree := bucket.present.FirstUnmarkedSlot()
			if nextFree >= int(bucket.cap) {
				panic("bad")
			}

			bucket.SetKv(nextFree, *bucket.overflow.GetKv(slotInBucket))
			bucket.fingerprints[nextFree] = bucket.overflow.fingerprints[slotInBucket]
			bucket.present.Mark(nextFree)
		}

		// Drop the overflow
		bucket.overflow = bucket.overflow.overflow

		// Retry looking from the bucket as it may now contain the key
		// that the user is trying to update
		return trie, bucket
	}

	// Grow the bucket if it's not at full capacity yet
	if bucket.cap < bucketSize {
		newCap := bucket.cap * 2
		if trie.IsEmpty() {
			// initial bucket grows 2x
		} else if trie.Get(hash>>trie.shift) != bucket {
			// overflow buckets grow 2x
		} else {
			// trie buckets jump to full size
			newCap = bucketSize
		}

		newbucket := m.map0
		if trie.NotEmpty() || newbucket == nil || newbucket.cap != newCap {
			newbucket = newBucketMap[K, V](newCap)
		} else {
			m.map0 = nil
		}

		newbucket.fingerprints = bucket.fingerprints
		newbucket.present = bucket.present
		newbucket.depth = bucket.depth
		copy(newbucket.Kvs(), bucket.Kvs())

		// Replace the old bucket with the new one in its place
		if trie.IsEmpty() {
			m.map0 = newbucket
			return trie, newbucket
		}

		// First find the correct place where to do the swap to the new bucket
		trieSlot := hash >> trie.shift
		place := trie.GetLocation(trieSlot)
		for *place != bucket {
			place = &((*place).overflow)
		}

		// Replace it
		if place == trie.GetLocation(trieSlot) {
			// replacing a trie entry requires more work
			shift := trie.shift % 64
			globalDepth := 64 - shift
			oldCount := uintptr(1) << uintptr(globalDepth-uintptr(bucket.depth))
			startpos := uintptr(hash>>shift) &^ (oldCount - 1)
			for i := startpos; i < startpos+oldCount; i++ {
				trie.Set(i, newbucket)
			}
		} else {
			// replacing overflow bucket is simple as they exist only in one location
			*place = newbucket
		}

		return trie, newbucket
	}

	globalDepth := (64 - trie.shift) % 64

	// Allocate overflow if we would cause the trie to grow too much
	// TODO: We could make it so that if we are about to allocate the 4th entry to the overflow chain
	// then we just split(). This would give us the property of being a perfect hash table because in the worst
	// case lookups would need to look at most into 3 buckets thus making it O(1) = perfect hash table
	if trie.shift < 64 && uintptr(bucket.depth) >= globalDepth {
		// How big should the trie be considering the total number of entries we are storing?
		// NOTE: This here is basically _the_ load-factor setting of this hash map
		// NOTE: The idea here is that as the map grows to a larger size (in the tens of thousands) we start to
		// aim for a higher load-factor to keep more of our data in our cpu caches and thus we are faster.
		// As expected this will cause our linked list of overflow buckets to grow longer and longer
		// but when 99.999% of entries can be reached through the first 2 buckets it doesn't matter
		// that we have 3rd and 4th overflow buckets. It's exceptionally rare to even have the 4th
		// bucket and I have never seen a 5th bucket.
		expectedTrieSize := m.len/bucketSize + 32<<(bits.Len(uint(m.len))/2)
		if trie.Len() >= expectedTrieSize {
			if !bucket.present.IsAllSet() {
				panic("bad")
			}

			// If we are close to allowing trie resizing then allocate smaller overflow bucket
			// assuming that in the mean time this bucket will not get too many inserts.
			sizeIndicatingSoonApproachingTrieResize := m.len/bucketSize + 32<<(bits.Len(uint(m.len))/2)*2
			if sizeIndicatingSoonApproachingTrieResize >= trie.Len() {
				// Assume less inserts for this bucket
				bucket.overflow = newBucketMap[K, V](8)
				bucket.overflow.depth = bucket.depth
				return trie, bucket.overflow
			}

			bucket.overflow = m.map0
			m.map0 = nil
			if bucket.overflow == nil {
				// Allocating our first overflow. Assume some inserts
				bucket.overflow = newBucketMap[K, V](bucketSize)
				bucket.overflow.depth = bucket.depth
			}
			bucket.overflow.depth = bucket.depth
			return trie, bucket.overflow
		}
	}

	// split it
	firstBucket := m.map0
	if trie.NotEmpty() {
		firstBucket = trie.Get(hash >> trie.shift)
	}

	firstBucket.depth++

	// Before inserting "newBucket" to the trie check if it needs to grow as well...
	if uintptr(firstBucket.depth) > globalDepth {
		// TODO: Do realloc for the trie. But then we have to do the moves in reverse order starting from the
		// back of the array.
		oldTrie := trie.Slice()
		newTrie := make([]*bucketMap[K, V], 1<<firstBucket.depth)

		// If it's our first trie then we need to remember to put the
		// map0 into it
		if len(oldTrie) == 0 {
			newTrie[0] = firstBucket
			newTrie[1] = firstBucket
			m.map0 = nil // map0 stores the free-list from now on
		} else {
			if len(newTrie) != len(oldTrie)*2 {
				panic("bad")
			}

			// NOTE: If we did the oldtrie+newtrie management like the concurrent map does
			// then we would have even more incrementality into the growth!
			// But this is simpler.
			// have: [b0 b0 b1 b2]
			// want: [b0 b0 b0 b0 b1 b1 b2 b2]
			for i, bucket := range oldTrie {
				newTrie[i*2+1] = bucket
				newTrie[i*2] = bucket
			}
		}

		trie = asSArray(newTrie)
		m.trie = trie
		globalDepth++
	}

	// Do the split moves
	newBucket, swapped := m.split(firstBucket)

	// Insert "newBucket" into the trie
	{
		var didNotSwap uintptr // is "newBucket" on the left or on the right?
		if !swapped {
			didNotSwap = 1
		}

		shift := trie.shift % 64
		oldCount := uintptr(1) << uintptr(globalDepth-uintptr(newBucket.depth-1))
		startpos := uintptr(hash>>shift) &^ (oldCount - 1)
		if oldCount < 2 {
			panic("bad")
		}

		// TODO: 99% the time (oldCount/2) is <= 2. Do something with it.

		startpos += didNotSwap * (oldCount / 2)
		for i := startpos; i < startpos+oldCount/2; i++ {
			trie.Set(i, newBucket)
		}
	}

	{
		// The overflow entries from the original first bucket need to be split
		// too.
		overflow := firstBucket.overflow
		if overflow != nil {
			firstBucket.overflow = nil
			leftb, rightb := maybeSwap(firstBucket, newBucket, swapped)
			m.reinsertOverflowBuckets(leftb, rightb, overflow)
		}
	}

	// return the correct bucket of the two for the hash
	return trie, trie.Get(hash >> (trie.shift % 64))
}

func maybeSwap[T any](left, right T, swap bool) (T, T) {
	pick := left
	unpick := right
	if swap {
		unpick = left
	}
	if swap {
		pick = right
	}
	return pick, unpick
}

func pickForHash[T any](left, right T, hash uintptr, depth uint8) (T, T) {
	pick := left
	unpick := right
	swap := hash&(1<<((64-depth)%64)) > 0
	if swap {
		unpick = left
	}
	if swap {
		pick = right
	}
	return pick, unpick
}

// reinsertOverflowBucket inserts the entries from an overflow bucket into the map
func (m *Map[K, V]) reinsertOverflowBuckets(leftb, rightb, firstOverflow *bucketMap[K, V]) {
	// TODO: When we start to have (N > 100k) entries some of our overflow buckets
	// have so many entries that the main buckets rarely have enough room, requiring
	// overflow buckets to be allocated. The situation just gets worse as we start
	// to have millions of entries. Optimize this function with that in mind by
	// trying to re-use the given overflow bucket as the new overflow bucket and not
	// doing any moves unless necessary. Consider that at ~200k entries our overflow
	// bucket starts to have an overflow bucket of its own.
	next := firstOverflow
	for overflow := firstOverflow; overflow != nil; overflow = next {
		next = overflow.overflow
		overflow.overflow = nil

		for presents := overflow.present.AsMatchiter(); presents.HasCurrent(); presents.Advance() {
			slotInBucket := presents.Current()

			kv := overflow.GetKv(slotInBucket)
			hash := m.hasher(unsafe.Pointer(&kv.Key), m.seed)
			tophash8 := fixFingerprint(uint64(hash))

			bucket, _ := pickForHash(leftb, rightb, hash, leftb.depth)
			for {
				// We can try inserting then if there's free space to do so. Unlike in Put() we know that the later buckets
				// in the linked list can't already contain this key.
				nextFree := bucket.present.FirstUnmarkedSlot()
				if nextFree < int(bucket.cap) {
					bucket.SetKv(nextFree, *kv)
					bucket.fingerprints[nextFree] = uint8(tophash8)
					bucket.present.Mark(nextFree)
					break
				}

				if bucket.overflow == nil {
					bucket.overflow = newBucketMap[K, V](bucketSize)
					bucket.overflow.depth = bucket.depth
				}

				// make leftb and rightb point to the last entry in their linked list so we don't have to always
				// find it again.
				if bucket == leftb {
					leftb = bucket.overflow
				} else {
					rightb = bucket.overflow
				}
				bucket = bucket.overflow
			}
		}

		// Add the bucket to the free-list
		if m.map0 != nil && m.map0.cap >= overflow.cap {
			continue
		}

		// We removed all entries and will add this to the free-list
		overflow.present = 0

		// Also clear the entries for GC
		clear(overflow.Kvs())

		// It's a single bucket free-list to keep it simple.
		m.map0 = overflow
	}
}

func (h *Map[K, V]) split(left *bucketMap[K, V]) (*bucketMap[K, V], bool) {
	if left.depth >= 64 || left.depth <= 0 {
		// 64-bit hash values have only 64 bits :-(
		panic("depth overflow")
	}
	if left.cap < bucketSize {
		panic("bad")
	}
	if !left.present.IsAllSet() {
		panic("bad")
	}

	oldDepth := left.depth - 1
	hashShift := (63 - oldDepth) % 64

	// TODO: Prefetch the keys before hashing them?

	// Move righties to the right map
	// NOTE: Doing it in 2 steps is much faster. One step version would have just a single
	// loop that has a branch in the body that checks if the entry should go to the right.
	// But it causes too many branch-misses.

	// Figure out which slots contain entries that will be moved to the "right" map
	var rightGoers bucketBitmask

	// NOTE: We know that the bucket is always full when we split()
	kvs := left.Kvs()[:bucketSize]
	for slotInBucket := range kvs {
		hash := h.hasher(unsafe.Pointer(&kvs[slotInBucket].Key), h.seed)

		goesRight := (hash >> hashShift) & 1
		rightGoers.MaybeMark(slotInBucket, goesRight)
	}

	// Simpler fast-path when K and V don't have pointers where we right is initialized
	// to be exact copy of left but then we just adjust "present" bitmask appropriately.
	// Also overflow linked list for right is zeroed. This speeds up insert benchmarks
	// by ~5-10% and could be a bigger improvement if the Go compiler didn't unnecessarily
	// have to zero the newly allocated fullmap.
	const splitShouldZeroEmptySlotsForGc = true
	if !splitShouldZeroEmptySlotsForGc {
		type fullmap struct {
			m   bucketMap[K, V]
			kvs [bucketSize]Kv[K, V]
		}

		// allocate the right map
		// NOTE: There is a way to write this function so that we can re-use the overflow bucket
		// as the 'right' bucket, avoiding unnecessarily allocating/freeing in such case. But
		// it's such a mess to write that I don't want to do it.
		right := (*fullmap)(unsafe.Pointer(h.map0))
		leftfull := (*fullmap)(unsafe.Pointer(left))
		if right != nil && right.m.cap == bucketSize {
			h.map0 = nil
			*right = *leftfull // just copy the whole thing
			right.m.overflow = nil
		} else {
			right = new(fullmap)
			*right = *leftfull // just copy the whole thing
			right.m.overflow = nil
		}

		left.present.UnmarkAll(rightGoers)
		right.m.present = rightGoers

		return &right.m, false
	}

	// We swap left and right around so that we minimize the amount of moves that we need to do.
	var swap bool
	if cnt := rightGoers.Count(); bucketSize-cnt < cnt {
		swap = true
	}

	if swap {
		old := rightGoers
		rightGoers = left.present
		rightGoers.UnmarkAll(old)
	}

	// allocate the right map
	// NOTE: There is a way to write this function such that we can re-use the overflow bucket
	// as the 'right' bucket, avoiding unnecessarily allocating/freeing in such case. But
	// it's such a mess to write that I don't want to do it.
	right := h.map0
	if right != nil && right.cap == bucketSize {
		h.map0 = nil
	} else {
		right = newBucketMap[K, V](bucketSize)
	}
	right.depth = left.depth

	// Do the moving

	// Outside the loop with SIMD
	left.present.UnmarkAll(rightGoers)
	right.present.MarkFirst(rightGoers.Count())
	// TODO: Move tophash8 with SIMD

	// NOTE: We fill from array start so that more of our kvs share their cache lines. Speeds up lookups
	var nextFreeInRight int
	for presents := rightGoers.AsMatchiter(); presents.HasCurrent(); presents.Advance() {
		slotInBucket := presents.Current()

		lhs := left.GetKv(slotInBucket)
		right.SetKv(nextFreeInRight, *lhs)
		right.fingerprints[nextFreeInRight] = left.fingerprints[slotInBucket]
		nextFreeInRight++

		*lhs = Kv[K, V]{} // for the GC, release pointers
	}

	return right, swap
}

func (m *Map[K, V]) iterateMaps(iter func(*bucketMap[K, V])) {
	// Oh... god... modifying iterators... pain...
	if m.trie.IsEmpty() {
		iter(m.map0)
		return
	}

	trie := m.trie.Slice()
	globalDepth := (64 - m.trie.shift) % 64
	for i := 0; i < len(trie); {
		triehash := i
		bucket := trie[triehash]

		i += 1 << (globalDepth - uintptr(bucket.depth%64))

		iter(bucket)
	}
}

func (m *Map[K, V]) Len() int {
	return m.len
}

// Not allowed to modify the map beyond the following:
// - updating the value for an existing key.
func (m *Map[K, V]) All(yield func(K, V) bool) {
	var trie []*bucketMap[K, V]
	if m.trie.IsEmpty() {
		if m.map0 == nil {
			return
		}
		trie = []*bucketMap[K, V]{m.map0}
	} else {
		trie = m.trie.Slice()
	}

	startPos := uintptr(runtime_fastrand64())
	bucketRotation := uint((startPos >> 32)) % bucketSize
	trieMask := uintptr(len(trie) - 1)
	globalDepth := (64 - m.trie.shift) % 64

	// align startPos so that we aren't in the middle of a bucket-repeat. This
	// ensures that inside the loop we can always jump forward past the repeats
	// to the next bucket
	{
		bucket := trie[startPos&trieMask]
		startPos &^= (1 << (globalDepth - uintptr(bucket.depth))) - 1
	}

	for i := 0; i < len(trie); {
		triehash := startPos + uintptr(i)
		bucket := trie[triehash&trieMask]

		i += 1 << ((globalDepth - uintptr(bucket.depth)) % 64)

		// TODO: prefetch the next bucket

		for bucket != nil {
			presents := bucket.present.AsMatchiter()

			// Rotate to random starting point
			// TODO: Make every bucket have a different rotation?
			presents = presents.RotateRight(int(bucketRotation))

			kvs := bucket.Kvs()
			for ; presents.HasCurrent(); presents.Advance() {
				slotInBucket := (bucketRotation + uint(presents.Current())) % bucketSize
				kv := &kvs[slotInBucket]
				if ok := yield(kv.Key, kv.Value); !ok {
					return
				}
			}
			bucket = bucket.overflow
		}
	}
}

// empties the map without releasing the main chunks of memory.
func (m *Map[K, V]) Clear() {
	m.iterateMaps(func(bucket *bucketMap[K, V]) {
		// zero the KVs, necessary for GC. Because we tend to have a high load-factor just clearing
		// the whole memory region is faster than iterating through all the present
		// slots and individually zeroing them.
		clear(bucket.Kvs())

		// mark all slots as empty
		bucket.present = 0

		// Overflow buckets we just set to nil and free them.
		bucket.overflow = nil
	})

	m.seed = uintptr(runtime_fastrand64())
	m.len = 0
}

func (m *Map[K, V]) Reserve(space int) {
	if m.len != 0 {
		panic("reserve called after first Put")
	}

	// NOTE: We seem to get a load factor of ~0.66 basically at any size.
	//
	// The strategy here is to allocate the trie into the full expected capacity
	// but only pre-allocate half of the expected maps to accommodate for the fact
	// that entries are not distributed perfectly evenly.

	adjusted := (space * 7 / 4) / bucketSize

	triesize := max(2, 1<<bits.Len(uint(adjusted)))

	newTrie := make([]*bucketMap[K, V], triesize)

	// then create the maps
	globalDepth := uint8(bits.Len(uint(triesize)))
	mapdepth := uint8(max(1, globalDepth-2))

	maps := make([]struct {
		m   bucketMap[K, V]
		kvs [bucketSize]Kv[K, V]
	}, 1<<(mapdepth-1))
	repeats := 1 << (globalDepth - mapdepth)
	for i := range maps {
		bucket := &maps[i].m
		bucket.depth = mapdepth - 1
		bucket.cap = uint8(cap(maps[i].kvs))

		for j := range repeats {
			newTrie[i*repeats+j] = bucket
		}
	}

	m.trie = asSArray(newTrie)
}

//go:linkname runtime_fastrand64 runtime.fastrand64
func runtime_fastrand64() uint64

// noescape hides a pointer from escape analysis.  noescape is
// the identity function but escape analysis doesn't think the
// output depends on the input.  noescape is inlined and currently
// compiles down to zero instructions.
// USE CAREFULLY!
//
// src/runtime/stubs.go
//
//go:nosplit
func noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x)
}

// getRuntimeHasher figures out the hash function that the built-in map would
// use to hash the key K.
func getRuntimeHasher[K comparable]() func(unsafe.Pointer, uintptr) uintptr {
	a := any((map[K]struct{})(nil))
	return (*rtEface)(unsafe.Pointer(&a)).typ.Hasher
}

// From runtime/runtime2.go:eface
type rtEface struct {
	typ  *rtMapType
	data unsafe.Pointer
}

// From internal/abi/type.go:MapType
type rtMapType struct {
	rtType
	Key    *rtType
	Elem   *rtType
	Bucket *rtType // internal type representing a hash bucket
	// function for hashing keys (ptr to key, seed) -> hash
	Hasher     func(unsafe.Pointer, uintptr) uintptr
	KeySize    uint8  // size of key slot
	ValueSize  uint8  // size of elem slot
	BucketSize uint16 // size of bucket
	Flags      uint32
}

type rtTFlag uint8
type rtNameOff int32
type rtTypeOff int32

// From internal/abi/type.go:Type
type rtType struct {
	Size_       uintptr
	PtrBytes    uintptr // number of (prefix) bytes in the type that can contain pointers
	Hash        uint32  // hash of type; avoids computation in hash tables
	TFlag       rtTFlag // extra type information flags
	Align_      uint8   // alignment of variable with this type
	FieldAlign_ uint8   // alignment of struct field with this type
	Kind_       uint8   // enumeration for C
	// function for comparing objects of this type
	// (ptr to object A, ptr to object B) -> ==?
	Equal func(unsafe.Pointer, unsafe.Pointer) bool
	// GCData stores the GC type data for the garbage collector.
	// If the KindGCProg bit is set in kind, GCData is a GC program.
	// Otherwise it is a ptrmask bitmap. See mbitmap.go for details.
	GCData    *byte
	Str       rtNameOff // string form
	PtrToThis rtTypeOff // type for pointer to this type, may be zero
}
