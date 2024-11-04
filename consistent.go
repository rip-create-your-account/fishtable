package fishtable

import (
	"cmp"
	"math"
	"math/bits"
	"slices"
	"unsafe"
)

func NewConsistent[K comparable, V any]() *ConsistentMap[K, V] {
	return &ConsistentMap[K, V]{
		hasher: getRuntimeHasher[K](),
		trie:   sarray[*consistentBucket[K, V]]{shift: 64},
		seed:   uintptr(runtime_fastrand64()),
	}
}

func NewConsistentWithUnsafeHasher[K comparable, V any](hasher func(key unsafe.Pointer, seed uintptr) uintptr) *ConsistentMap[K, V] {
	return &ConsistentMap[K, V]{
		hasher: hasher,
		trie:   sarray[*consistentBucket[K, V]]{shift: 64},
		seed:   uintptr(runtime_fastrand64()),
	}
}

type ConsistentMap[K comparable, V any] struct {
	hasher func(key unsafe.Pointer, seed uintptr) uintptr
	trie   sarray[*consistentBucket[K, V]]
	// the initial map that we use before the first split() and the trie creation
	map0 *consistentBucket[K, V]
	seed uintptr
	len  int

	// Real languages have bulk free() so freeing all of the the little maps
	// should be fast.
}

type consistentBucket[K comparable, V any] struct {
	hashes  []uintptr  // sorted by ascending hash, len usually <= bucketSize
	entries []Kv[K, V] // sorted by ascending hash, len usually <= bucketSize

	depth uint8 // <= 64
}

func newConsistentBucket[K comparable, V any]() *consistentBucket[K, V] {
	return &consistentBucket[K, V]{}
}

// binary search that takes advantage of the fact that len(x) is usually <= bucketSize
func bucketBinarySearch(x []uintptr, target uintptr) int {
	n := len(x)
	// Define x[-1] < target and x[n] >= target.
	// Invariant: x[i-1] < target, x[j] >= target.
	i, j := 0, n
	if i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		if cmp.Less(x[h], target) {
			i = h + 1 // preserves x[i-1] < target
		} else {
			j = h // preserves x[j] >= target
		}
	} else {
		return i
	}
	if i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		if cmp.Less(x[h], target) {
			i = h + 1 // preserves x[i-1] < target
		} else {
			j = h // preserves x[j] >= target
		}
	} else {
		return i
	}
	if i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		if cmp.Less(x[h], target) {
			i = h + 1 // preserves x[i-1] < target
		} else {
			j = h // preserves x[j] >= target
		}
	} else {
		return i
	}
	if i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		if cmp.Less(x[h], target) {
			i = h + 1 // preserves x[i-1] < target
		} else {
			j = h // preserves x[j] >= target
		}
	} else {
		return i
	}
	if i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		if cmp.Less(x[h], target) {
			i = h + 1 // preserves x[i-1] < target
		} else {
			j = h // preserves x[j] >= target
		}
	} else {
		return i
	}
	// give up and loop
	for i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		if cmp.Less(x[h], target) {
			i = h + 1 // preserves x[i-1] < target
		} else {
			j = h // preserves x[j] >= target
		}
	}
	// i == j, x[i-1] < target, and x[j] (= x[i]) >= target  =>  answer is i.
	return i
}

// Finds the value for an entry whose hash value is closest to that of k.
func (m *ConsistentMap[K, V]) GetClosest(k K) (vk K, v V, ok bool) {
	hash := m.hasher(noescape(unsafe.Pointer(&k)), m.seed)

	bucket := m.map0
	trie := m.trie
	if trie.shift < 64 {
		bucket = trie.Get(hash >> trie.shift)
	}

	if bucket != nil {
		bestIndex := bucketBinarySearch(bucket.hashes, hash)
		bestIndex = min(bestIndex, len(bucket.hashes)-1)
		bestHash := bucket.hashes[bestIndex]

		// Check if the entry next to our current best is actually closer.
		// TODO: The code below is shit and likely incorrect.
		if hash < bestHash {
			var prevKv *Kv[K, V]
			var prevHash uintptr
			if bestIndex == 0 {
				shift := trie.shift % 64
				globalDepth := (64 - shift)
				count := uintptr(1) << uintptr(globalDepth-uintptr(bucket.depth))
				startPos := uintptr(hash>>shift) &^ (count - 1)

				prevBucket := trie.Get((startPos - 1) << trie.shift >> trie.shift)
				prevHash = prevBucket.hashes[len(prevBucket.entries)-1]
				prevKv = &prevBucket.entries[len(prevBucket.entries)-1]
			} else {
				prevHash = bucket.hashes[bestIndex-1]
				prevKv = &bucket.entries[bestIndex-1]
			}

			if (hash - prevHash) < (bestHash - hash) {
				return prevKv.Key, prevKv.Value, true
			}
		} else if hash > bestHash {
			var nextKv *Kv[K, V]
			var nextHash uintptr
			if bestIndex == len(bucket.hashes)-1 {
				shift := trie.shift % 64
				globalDepth := (64 - shift)
				count := uintptr(1) << uintptr(globalDepth-uintptr(bucket.depth))
				startPos := uintptr(hash>>shift) &^ (count - 1)

				nextBucket := trie.Get((startPos + count) << trie.shift >> trie.shift)
				nextHash = nextBucket.hashes[0]
				nextKv = &nextBucket.entries[0]
			} else {
				nextHash = bucket.hashes[bestIndex+1]
				nextKv = &bucket.entries[bestIndex+1]
			}

			if (nextHash - hash) < (hash - bestHash) {
				return nextKv.Key, nextKv.Value, true
			}
		}

		kv := &bucket.entries[bestIndex]
		return kv.Key, kv.Value, true
	}
	return
}

func (m *ConsistentMap[K, V]) Delete(k K) {
	hash := m.hasher(noescape(unsafe.Pointer(&k)), m.seed)

	bucket := m.map0
	trie := m.trie
	if trie.shift < 64 {
		bucket = trie.Get(hash >> trie.shift)
	}

	// bucket is nil for empty maps
	if bucket != nil {
		i, found := slices.BinarySearch(bucket.hashes, hash)
		if found && bucket.entries[i].Key == k {
			bucket.hashes = slices.Delete(bucket.hashes, i, i+1)
			bucket.entries = slices.Delete(bucket.entries, i, i+1)
			m.len--

			if len(bucket.entries) == 0 {
				// To keep GetClosest simple and O(1) we must ensure that all
				// buckets have at least one entry.
				// NOTE: This is ass. Not gonna bother implementing it.
				panic("TODO: merge empty bucket with sibling")
			}
		}
	}
}

func (m *ConsistentMap[K, V]) Put(k K, v V) {
	hash := m.hasher(noescape(unsafe.Pointer(&k)), m.seed)

	bucket := m.map0
	trie := m.trie
	if trie.shift < 64 {
		bucket = trie.Get(hash >> trie.shift)
	}

	for {
		// bucket is nil for empty maps
		if bucket != nil {
			i, found := slices.BinarySearch(bucket.hashes, hash)
			if found && bucket.entries[i].Key == k {
				bucket.entries[i].Value = v
				return
			}

			// NOTE: we may want to split instead if we are full
			if len(bucket.entries) < cap(bucket.entries) {
				bucket.hashes = slices.Insert(bucket.hashes, i, hash)
				bucket.entries = slices.Insert(bucket.entries, i, Kv[K, V]{Key: k, Value: v})
				m.len++
				return
			}
		}

		trie, bucket = m.makeSpaceForHash(trie, bucket, hash)
	}
}

// figures out a good bucket for an entry with the given hash.
func (m *ConsistentMap[K, V]) makeSpaceForHash(trie sarray[*consistentBucket[K, V]], bucket *consistentBucket[K, V], hash uintptr) (sarray[*consistentBucket[K, V]], *consistentBucket[K, V]) {
	// bucket/map0 is nil if the map is empty
	if bucket == nil {
		if trie.NotEmpty() {
			panic("bad")
		}
		m.map0 = newConsistentBucket[K, V]()
		m.map0.hashes = slices.Grow(m.map0.hashes, bucketSize)
		m.map0.entries = slices.Grow(m.map0.entries, bucketSize)
		return trie, m.map0
	}

	globalDepth := (64 - trie.shift) % 64

	// Allocate overflow if we would cause the trie to grow too much
	if trie.shift < 64 && uintptr(bucket.depth) >= globalDepth {
		// How big should the trie be considering the total number of entries we are storing?
		// NOTE: This here is basically _the_ load-factor setting of this hash map
		expectedTrieSize := m.len/bucketSize + 32<<(bits.Len(uint(m.len))/2)
		if trie.Len() >= expectedTrieSize {
			// double the size
			bucket.hashes = slices.Grow(bucket.hashes, len(bucket.hashes))
			bucket.entries = slices.Grow(bucket.entries, len(bucket.entries))
			return trie, bucket
		}
	}

	// split it
	bucket.depth++

	// Do the split moves
	newBucket := m.split(hash, bucket)

	// We require that splitting must never produce an empty bucket. So splitting can fail
	if newBucket == nil {
		bucket.depth--
		// double the size
		bucket.hashes = slices.Grow(bucket.hashes, len(bucket.hashes))
		bucket.entries = slices.Grow(bucket.entries, len(bucket.entries))
		return trie, bucket
	}

	// Before inserting "newBucket" to the trie check if it needs to grow as well...
	if uintptr(bucket.depth) > globalDepth {
		// TODO: Do realloc for the trie. But then we have to do the moves in reverse order starting from the
		// back of the array.
		oldTrie := trie.Slice()
		newTrie := make([]*consistentBucket[K, V], 1<<bucket.depth)

		// If it's our first trie then we need to remember to put the
		// map0 into it
		if len(oldTrie) == 0 {
			newTrie[0] = bucket
			newTrie[1] = bucket
			m.map0 = nil
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

	// Insert "newBucket" into the trie
	{
		shift := trie.shift % 64
		oldCount := uintptr(1) << uintptr(globalDepth-uintptr(newBucket.depth-1))
		startpos := uintptr(hash>>shift) &^ (oldCount - 1)
		if oldCount < 2 {
			panic("bad")
		}

		// TODO: 99% the time (oldCount/2) is <= 2. Do something with it.

		startpos += oldCount / 2
		for i := startpos; i < startpos+oldCount/2; i++ {
			trie.Set(i, newBucket)
		}
	}

	// return the correct bucket of the two for the hash
	return trie, trie.Get(hash >> (trie.shift % 64))
}

func (h *ConsistentMap[K, V]) split(hash uintptr, left *consistentBucket[K, V]) *consistentBucket[K, V] {
	if left.depth >= 64 || left.depth <= 0 {
		// 64-bit hash values have only 64 bits :-(
		panic("depth overflow")
	}

	oldDepth := left.depth - 1
	hashShift := (63 - oldDepth) % 64

	// Figure out which slots contain entries that will be moved to the "right" map
	maxLeftGoing := (hash &^ (1 << hashShift)) | (math.MaxUint64 >> left.depth)
	rightGoersStartAt, _ := slices.BinarySearch(left.hashes, maxLeftGoing+1)

	// We must never produce an empty bucket for either side
	if rightGoersStartAt == 0 || rightGoersStartAt >= len(left.hashes) {
		return nil
	}

	right := newConsistentBucket[K, V]()
	right.depth = left.depth

	right.hashes = make([]uintptr, 0, max(bucketSize, len(left.hashes[rightGoersStartAt:])))
	right.entries = make([]Kv[K, V], 0, max(bucketSize, len(left.hashes[rightGoersStartAt:])))

	right.hashes = append(right.hashes, left.hashes[rightGoersStartAt:]...)
	right.entries = append(right.entries, left.entries[rightGoersStartAt:]...)

	left.hashes = left.hashes[:rightGoersStartAt]
	left.entries = left.entries[:rightGoersStartAt]

	return right
}

func (m *ConsistentMap[K, V]) iterateMaps(iter func(*consistentBucket[K, V])) {
	// Oh... god... modifying iterators... pain...
	if m.trie.IsEmpty() {
		iter(m.map0)
		return
	}

	trie := m.trie.Slice()
	globalDepth := (64 - m.trie.shift) % 64
	for i := 0; i < len(trie); {
		triehash := i
		sm := trie[triehash]

		i += 1 << (globalDepth - uintptr(sm.depth%64))

		iter(sm)
	}
}

func (m *ConsistentMap[K, V]) Len() int {
	return m.len
}

// Not allowed to modify the map beyond the following:
// - updating the value for an existing key.
func (m *ConsistentMap[K, V]) All(yield func(K, V) bool) {
	var trie []*consistentBucket[K, V]
	if m.trie.IsEmpty() {
		if m.map0 == nil {
			return
		}
		trie = []*consistentBucket[K, V]{m.map0}
	} else {
		trie = m.trie.Slice()
	}

	startPos := uintptr(runtime_fastrand64())
	trieMask := uintptr(len(trie) - 1)
	globalDepth := (64 - m.trie.shift) % 64

	// align startPos so that we aren't in the middle of a bucket-repeat. This
	// ensures that inside the loop we can always jump forward past the repeats
	// to the next bucket
	{
		sm := trie[startPos&trieMask]
		startPos &^= (1 << (globalDepth - uintptr(sm.depth))) - 1
	}

	for i := 0; i < len(trie); {
		triehash := startPos + uintptr(i)
		bucket := trie[triehash&trieMask]

		i += 1 << ((globalDepth - uintptr(bucket.depth)) % 64)

		// TODO: prefetch the next bucket

		for slotInBucket := range bucket.entries {
			kv := &bucket.entries[slotInBucket]
			if ok := yield(kv.Key, kv.Value); !ok {
				return
			}
		}
	}
}
